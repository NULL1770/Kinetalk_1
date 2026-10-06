"""Resumable ARKit adapter training and native-length development scoring."""
from __future__ import annotations
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.faceformer_arkit_adapter import load_prepared, _batch, validate_batch
from scripts.faceformer_arkit_model import FaceFormerARKit, FaceFormerARKitConfig
from scripts.voca_emotalk_arkit_models import VocaARKit, EmoTalkARKit, masked_mse, masked_velocity_mse
from scripts.train_formal_predictable_projection import save_json, save_checkpoint, capture_rng, restore_rng, canonical_hash
from scripts.arkit_benchmark_report import score_fullface, build_report, write_report
from scripts.baseline_core_arkit import (VocaCoreARKit, EmoTalkCoreARKit, PairedUtterances,
                                       adaptation_record, VERSION)


def make_model(method, config):
    if config.get('adapter_version') == VERSION:
        if method == 'voca': return VocaCoreARKit()
        if method == 'emotalk': return EmoTalkCoreARKit()
    if method == 'faceformer': return FaceFormerARKit(FaceFormerARKitConfig(**config))
    if method == 'voca': return VocaARKit(hidden=config['hidden'])
    if method == 'emotalk': return EmoTalkARKit(hidden=config['hidden'], layers=config['layers'])
    raise ValueError(method)


def predict(model, method, batch):
    if isinstance(model, VocaCoreARKit):
        return model(batch['audio_features'], batch['anchors'], batch['valid']), None
    if method == 'faceformer': return model.predict(batch['audio_features'], batch['anchors']), None
    if method == 'emotalk': return model(batch['audio_features'], batch['anchors'], batch['valid'])
    return model(batch['audio_features'], batch['anchors']), None


@torch.no_grad()
def evaluate_baseline(model, method, data, output, device, batch_size, *, smoke=False):
    """Score all validation clips equally; retain every native-frame prediction."""
    model.eval()
    split = data['splits']['validation']
    n = len(split['valid']); rows = []; clips = {}; mse_sum = 0.; emotion = []
    for ids in torch.arange(n).split(batch_size):
        batch = _batch(split, ids, device)
        prediction, logits = predict(model, method, batch)
        prediction = prediction.cpu()
        if logits is not None:
            emotion.extend((logits.argmax(-1).cpu() == split['emotion_id'][ids]).tolist())
        for j, i in enumerate(ids.tolist()):
            length = int(split['_lengths'][i]) if '_lengths' in split else batch['valid'].shape[1]
            target = batch['motion'][j, :length].cpu(); valid = batch['valid'][j, :length].cpu()
            channel = batch['channel_mask'][j].cpu()
            cid = split.get('clip_id', [str(k) for k in range(n)])[i]
            times = (split.batch(torch.tensor([i]), keys=('times',))['times'][0]
                     if hasattr(split, 'batch') else split['times'][i, :length])
            pred = prediction[j, :length]
            mask = valid[:, None] & channel[None]
            mse_sum += float((pred[mask] - target[mask]).square().mean())
            row = score_fullface(pred.numpy()[None], dict(clip_id=cid,
                speaker=split.get('speaker', ['unknown']*n)[i],
                sentence=split.get('sentence_id', ['unknown']*n)[i],
                emotion=int(split['emotion_id'][i]), target52=target.numpy(),
                valid=valid.numpy(), channel_mask=np.broadcast_to(channel.numpy(), target.shape),
                times=times.numpy()))
            rows.append(row)
            clips[cid] = dict(prediction=pred, valid=valid, times=times, channel_mask=channel)
        if output:
            save_json(output / 'status.json', dict(status='evaluating', clips=len(rows), total=n, test_loaded=False))
        if len(rows) % (batch_size * 25) == 0:
            print(json.dumps(dict(event='validation', clips=len(rows), total=n)), flush=True)
    report = build_report(rows, scope='development validation; not sealed test', sources=data['provenance'])
    report.update(method=method, smoke=smoke, test_loaded=False, validation_mse=mse_sum/n,
                  validation_emotion_accuracy=sum(emotion)/len(emotion) if emotion else None,
                  model_scope='MEAD-ARKit cached-audio adaptation; not official dataset reproduction')
    if isinstance(model, (VocaCoreARKit, EmoTalkCoreARKit)):
        report['baseline_adaptation'] = adaptation_record(method)
    if output:
        write_report(output / 'arkit_full.json', report)
        save_checkpoint(output / 'native_predictions.pt', dict(clips=clips, test_loaded=False,
                        scope='development validation', manifest_sha256=data['provenance'].get('manifest_sha256')))
    return report


def train_baseline(args, method):
    threads = getattr(args, 'threads', 2)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    started = time.monotonic()
    data = load_prepared(args.data, smoke=args.smoke)
    for split in data['splits'].values(): validate_batch(split)
    print(json.dumps(dict(event='data_ready', seconds=time.monotonic()-started,
                          train_clips=len(data['splits']['train']['valid']), threads=threads)), flush=True)
    if method == 'faceformer':
        config = FaceFormerARKitConfig(feature_dim=args.feature_dim, period=args.period,
                        dropout=args.dropout, max_seq_len=args.max_seq_len).__dict__
    elif getattr(args, 'adapter', 'legacy') == VERSION:
        config = dict(adapter_version=VERSION)
    else: config = dict(hidden=args.hidden, layers=args.layers)
    model = make_model(method, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 weight_decay=.01 if method == 'faceformer' else 1e-5)
    generator = torch.Generator().manual_seed(args.seed)
    train = data['splits']['train']; n = len(train['valid'])
    core = config.get('adapter_version') == VERSION
    pairs = PairedUtterances(train) if core and method == 'emotalk' else None
    bpe = (n + args.batch_size - 1) // args.batch_size
    epochs = 1 if args.smoke else args.epochs
    requested_steps = getattr(args, 'steps', None)
    steps = requested_steps if requested_steps is not None else epochs * bpe
    root = Path(__file__).resolve().parents[1]
    source_files = ['scripts/baseline_training.py', 'scripts/packed_trainval_cache.py',
                    'scripts/faceformer_arkit_model.py', 'scripts/voca_emotalk_arkit_models.py',
                    'scripts/faceformer_arkit_adapter.py']
    if core:
        source_files += ['scripts/baseline_core_arkit.py', 'docs/baseline_adaptation_20260923.md']
        source_files += (['third_party/voca_reference/utils/speech_encoder.py',
                         'third_party/voca_reference/utils/expression_layer.py',
                         'third_party/voca_reference/config_parser.py'] if method == 'voca' else
                        ['third_party/emotalk_release/model.py', 'third_party/emotalk_release/utils.py'])
    protocol = dict(schema='resumable_arkit_adapter_v2', method=method, config=config,
        epochs=epochs, steps=steps, batch_size=args.batch_size, seed=args.seed, lr=args.lr,
        data=data['provenance'], test_loaded=False, smoke=args.smoke, selection='fixed final epoch',
        sources={f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in source_files})
    if core:
        protocol['baseline_adaptation'] = adaptation_record(method)
        protocol['objective'] = ('masked coefficient MSE + 10*velocity MSE' if method == 'voca' else
            'mean(cross reconstruction,self reconstruction) + .1*mean(velocities) + .05*emotion CE + .05*level CE')
        if pairs: protocol['pairing'] = pairs.audit
    digest = canonical_hash(protocol)
    output = Path(args.output) if args.output else None
    first = 0; total_loss = 0.; elapsed = 0.
    if output:
        if getattr(args, 'resume', False):
            saved = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
            if saved['protocol_sha256'] != digest: raise ValueError('Resume protocol differs')
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            restore_rng(saved['rng'], generator)
            first=saved['step']; total_loss=saved['loss_sum']; elapsed=saved['elapsed_seconds']
        else:
            output.mkdir(parents=True, exist_ok=False)
            save_json(output/'protocol.json', protocol)
            for f in source_files:
                dest=output/'source'/f; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((root/f).read_bytes())
    def checkpoint(step):
        if output:
            save_checkpoint(output/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                step=step, rng=capture_rng(generator), protocol_sha256=digest, loss_sum=total_loss,
                elapsed_seconds=elapsed+time.monotonic()-started))
    if first == 0: checkpoint(0)
    model.train(); epoch_loss=0.; epoch_started=time.monotonic()
    try:
        for step in range(first, steps):
            epoch_step=step % bpe
            if requested_steps is not None:
                ids=torch.randint(n,(args.batch_size,),generator=generator)
            else:
                if epoch_step == 0: order=torch.randperm(n,generator=generator)
                ids=order[epoch_step*args.batch_size:(epoch_step+1)*args.batch_size]
            batch=_batch(train,ids,args.device)
            if method == 'faceformer':
                pred=model(batch['audio_features'],batch['anchors'],batch['motion'],batch['valid'],teacher_forcing=True)
                loss=masked_mse(pred,batch['motion'],batch['valid'],batch['channel_mask'])
            elif pairs:
                content_ids, emotion_ids = pairs.sample(ids, generator)
                content = _batch(train, content_ids, args.device)
                emotion = _batch(train, emotion_ids, args.device)
                pred, logits, levels = model.paired(content['audio_features'], content['valid'],
                    emotion['audio_features'], emotion['valid'], batch['anchors'], batch['valid'])
                own, own_logits, own_levels = model.paired(content['audio_features'], content['valid'],
                    content['audio_features'], content['valid'], content['anchors'], content['valid'])
                cross_loss = masked_mse(pred,batch['motion'],batch['valid'],batch['channel_mask'])
                own_loss = masked_mse(own,content['motion'],content['valid'],content['channel_mask'])
                velocity = masked_velocity_mse(pred,batch['motion'],batch['valid'],batch['channel_mask'])
                velocity = velocity + masked_velocity_mse(own,content['motion'],content['valid'],content['channel_mask'])
                ce = F.cross_entropy(logits,train['emotion_id'][emotion_ids].to(args.device))
                ce = ce + F.cross_entropy(own_logits,train['emotion_id'][content_ids].to(args.device))
                level_ce = F.cross_entropy(levels,train['intensity_id'][emotion_ids].to(args.device))
                level_ce = level_ce + F.cross_entropy(own_levels,train['intensity_id'][content_ids].to(args.device))
                loss = .5*(cross_loss+own_loss) + .05*velocity + .025*(ce+level_ce)
            else:
                pred,logits=predict(model,method,batch)
                loss=masked_mse(pred,batch['motion'],batch['valid'],batch['channel_mask'])
                loss=loss+(10. if core and method == 'voca' else .1)*masked_velocity_mse(pred,batch['motion'],batch['valid'],batch['channel_mask'])
                if logits is not None: loss=loss+.05*F.cross_entropy(logits,train['emotion_id'][ids].to(args.device))
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training objective')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            value=float(loss.detach());total_loss+=value;epoch_loss+=value
            if step % 50 == 0:
                record=dict(status='training',step=step+1,steps=steps,epoch=step//bpe+1,loss=value,
                            elapsed_seconds=elapsed+time.monotonic()-started,test_loaded=False)
                if output: save_json(output/'status.json',record)
                print(json.dumps(record),flush=True)
            if (step+1) % bpe == 0 or step+1 == steps:
                checkpoint(step+1)
                record=dict(event='epoch_complete',epoch=step//bpe+1,steps=step+1,loss=epoch_loss/(epoch_step+1),seconds=time.monotonic()-epoch_started)
                if output: save_json(output/f'epoch{step//bpe+1:03d}.json',record)
                print(json.dumps(record),flush=True);epoch_loss=0.;epoch_started=time.monotonic()
        report=dict(schema=protocol['schema'], method=method, train_clips=n,
                    validation_clips=len(data['splits']['validation']['valid']),epochs=epochs,steps=steps,
                    seed=args.seed,smoke=args.smoke,test_loaded=False,train_loss_mean=total_loss/max(1,steps))
        if core: report['baseline_adaptation'] = adaptation_record(method)
        if output:
            save_checkpoint(output/'checkpoint.pt',dict(schema=report['schema'],method=method,config=config,
                model=model.state_dict(),report=report,protocol=protocol,protocol_sha256=digest))
            save_json(output/'training_complete.json',report)
        scored=evaluate_baseline(model,method,data,output,args.device,args.batch_size,smoke=args.smoke)
        report.update(validation_mse=scored['validation_mse'],validation_emotion_accuracy=scored['validation_emotion_accuracy'])
        if method == 'faceformer':report['validation_autoregressive_mse']=report['validation_mse']
        if output:
            save_json(output/'report.json',report)
            save_json(output/'status.json',dict(status='complete',test_loaded=False,smoke=args.smoke))
        return model,report
    except BaseException as exc:
        if output: save_json(output/'failure.json',dict(error=repr(exc),recover_from='last.pt'))
        raise
