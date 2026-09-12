from __future__ import annotations

import csv
import argparse
import random
from pathlib import Path

import torch

from kinetalk_b0.data import B0ResidualDataset
from kinetalk_b0.models import Stage1Model, Stage2Model, Stage3Model, Stage4Model
from kinetalk_b0.utils import load_checkpoint, load_yaml


BS_NAMES = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
    'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
    'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
    'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight',
]

# Training motion is stored in ARKit-52 order. Blender uses the supplied
# MediaPipe-style 51-name order, so export by name rather than by position.
ARKIT_NAMES = [
    'eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft',
    'eyeLookUpLeft', 'eyeSquintLeft', 'eyeWideLeft', 'eyeBlinkRight',
    'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight',
    'eyeSquintRight', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawRight',
    'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft',
    'mouthRight', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFrownLeft',
    'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft',
    'mouthStretchRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
    'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft',
    'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
    'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
    'noseSneerLeft', 'noseSneerRight', 'tongueOut',
]
ARKIT_INDEX = {name: index for index, name in enumerate(ARKIT_NAMES)}
EXPORT_INDEX = [ARKIT_INDEX[name] for name in BS_NAMES]


def branch(dataset: B0ResidualDataset, index: int, start: int | None = None) -> dict:
    sample = dataset._read(dataset.items[index], start=start)
    neutral_index, neutral_valid = dataset._neutral_index(index)
    neutral = dataset._read(dataset.items[neutral_index], start=sample['crop_start'])
    dataset._attach_targets(sample, neutral, neutral_valid)
    return sample


def emotion_name(dataset: B0ResidualDataset, index: int) -> str:
    return dataset._emotion_name(dataset.items[index])


def pick_indices(dataset: B0ResidualDataset) -> tuple[int, int, int]:
    happy = [i for i in range(len(dataset.items)) if emotion_name(dataset, i) == 'happy']
    if not happy:
        raise RuntimeError('No happy samples found')
    for qi in happy:
        speaker, sentence = dataset._group_key(dataset.items[qi])
        angry = [i for i in dataset.by_group[(speaker, sentence)] if emotion_name(dataset, i) == 'angry']
        other = [i for sp, ids in dataset.by_speaker.items() if sp != speaker for i in ids]
        if angry and other:
            return qi, angry[0], other[0]
    raise RuntimeError('Could not find happy sample with same-group angry partner and different-speaker style donor')


def write_csv(path: Path, values: torch.Tensor) -> None:
    values = values.detach().float().cpu()
    if values.ndim != 2:
        raise ValueError(f'Expected [frames, channels], got {tuple(values.shape)}')
    # The model contract is raw ARKit-52.  Do not accept a 51-D tensor here:
    # indexing it would silently produce a plausible-looking but shifted CSV.
    if values.shape[1] != len(ARKIT_NAMES):
        raise ValueError(
            f'Expected raw ARKit-{len(ARKIT_NAMES)} tensor in ARKIT_NAMES order, '
            f'got {values.shape[1]} channels'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Frame', *BS_NAMES])
        for frame, row in enumerate(values[:, EXPORT_INDEX].tolist()):
            writer.writerow([frame, *[f'{float(value):.8f}' for value in row]])


def main() -> None:
    parser = argparse.ArgumentParser(description='Export representative Stage-4 outputs in Blender semantic order')
    parser.add_argument('--config', type=Path, default=Path('configs/train.yaml'))
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val',
                        help='Manifest split to export; validation is the default')
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/blender/_semantic_latest'))
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = B0ResidualDataset(cfg, split=args.split, random_crop=False)
    query_index, angry_index, donor_index = pick_indices(dataset)
    query = branch(dataset, query_index)
    angry = branch(dataset, angry_index, start=query['crop_start'])
    donor = branch(dataset, donor_index)

    stage1 = Stage1Model(cfg).to(device)
    load_checkpoint(cfg['paths']['stage1_ckpt'], stage1, map_location=device)
    stage2 = Stage2Model(cfg).to(device)
    load_checkpoint(cfg['paths']['stage2_ckpt'], stage2, map_location=device, strict=True)
    stage3 = Stage3Model(cfg, stage2).to(device)
    load_checkpoint(cfg['paths']['stage3_ckpt'], stage3, map_location=device, strict=True)
    stage4 = Stage4Model(cfg, stage1, stage2, stage3).to(device)
    load_checkpoint(cfg['paths']['stage4_ckpt'], stage4, map_location=device, strict=True)
    for model in (stage1, stage2, stage3, stage4):
        model.eval()

    def tensor_branch(sample: dict) -> dict:
        return {key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value for key, value in sample.items()}

    q = tensor_branch(query)
    a = tensor_branch(angry)
    d = tensor_branch(donor)
    base_batch = {'query': q, 'style_reference': q}
    donor_batch = {'query': q, 'style_reference': d}
    steps = int(cfg['model'].get('render_steps', 4))
    with torch.no_grad():
        reconstruction, _ = stage4.render(base_batch, stage4.conditions(base_batch), steps=steps, stochastic=False)
        style_swap, _ = stage4.render(donor_batch, stage4.conditions(donor_batch), steps=steps, stochastic=False)
        q_audio = stage3(q['audio_emotion'], q['mask'])
        a_audio = stage3(a['audio_emotion'], a['mask'])
        style = stage2.encode_factors(q['residual_gt'], q['residual_mask'], q['audio_emotion'])['style']
        conditions = stage4.conditions(base_batch)
        emotion_conditions = dict(conditions)
        emotion_conditions['global'] = a_audio['global']
        emotion_conditions['intensity_value'] = a_audio['intensity_value']
        emotion_swap, _ = stage4.render(base_batch, emotion_conditions, steps=steps, stochastic=False)

    output_dir = args.output_dir
    write_csv(output_dir / 'gt.csv', q['motion'][0])
    write_csv(output_dir / 'reconstruction.csv', reconstruction[0])
    write_csv(output_dir / 'emotion_swap_happy_to_angry.csv', emotion_swap[0])
    write_csv(output_dir / 'style_swap_different_person.csv', style_swap[0])
    print(f'query={query["clip_id"]} speaker={query["speaker"]} emotion={emotion_name(dataset, query_index)}')
    print(f'angry={angry["clip_id"]} speaker={angry["speaker"]} emotion={emotion_name(dataset, angry_index)}')
    print(f'donor={donor["clip_id"]} speaker={donor["speaker"]} emotion={emotion_name(dataset, donor_index)}')
    print(f'split={args.split} output_dir={output_dir.resolve()} channels={len(BS_NAMES)} frames={q["motion"].shape[1]}')


if __name__ == '__main__':
    main()
