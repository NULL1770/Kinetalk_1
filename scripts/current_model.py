"""Load the explicitly selected working model without silently selecting a pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_current_model(selection=None):
    path = Path(selection) if selection else Path(__file__).resolve().parents[1] / 'configs/current_model.json'
    record = json.loads(path.read_text(encoding='utf8'))
    if record['model_family'] != 'KineTalkV4Model':
        raise ValueError('Selected family is not the restored joint model')
    checkpoint = Path(record['checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = path.resolve().parents[1] / checkpoint
    if sha(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('Selected checkpoint hash differs')
    source_root = Path(record['source_root'])
    if not source_root.is_absolute():
        source_root = (path.parent / source_root).resolve()
    for name, digest in record['model_source_sha256'].items():
        if sha(source_root / name) != digest:
            raise ValueError(f'Selected source differs: {name}')
    return record, checkpoint, source_root


def load_current_model(selection=None, device='cpu'):
    record, checkpoint, source_root = resolve_current_model(selection)
    # Read the recorded source tree before importing the restored family.
    sys.path.insert(0, str(source_root))
    import torch
    from kinetalk_v4.unified_model_v4 import KineTalkV4Config, KineTalkV4Model
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = saved['model']
    model = KineTalkV4Model(KineTalkV4Config(**saved['config']),
        state['feature_mean'], state['feature_std'], state['scales'],
        state['decoder.geometry_jacobian'], state['decoder.geometry_offset'],
        state['decoder.geometry_scale']).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if tuple(model.decoder.output.weight.shape) != (52, 192):
        raise ValueError('Selected model has no expected joint output')
    return model, record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    model, record = load_current_model(args.selection, args.device)
    print(json.dumps(dict(selected_model=record['selected_model'],
        checkpoint_sha256=record['checkpoint_sha256'],
        model_family=type(model).__name__, strict_load=True,
        joint_output_shape=list(model.decoder.output.weight.shape),
        training_started=False)))
