import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from kinetalk_b0.data import B0ResidualDataset, collate_b0_residual
from kinetalk_b0.models import Stage2Model, Stage3Model
from kinetalk_b0.utils import load_yaml, load_checkpoint, move_to_device

cfg=load_yaml('configs/train.yaml'); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds=B0ResidualDataset(cfg,split='train',random_crop=False); dl=DataLoader(ds,batch_size=32,num_workers=0,collate_fn=collate_b0_residual)
s2=Stage2Model(cfg).to(dev); load_checkpoint(cfg['paths']['stage2_ckpt'],s2,map_location=dev,strict=True); s2.eval()
s3=Stage3Model(cfg,s2).to(dev); load_checkpoint(cfg['paths']['stage3_ckpt'],s3,map_location=dev,strict=True); s3.eval()
N=e2=i2=e3=i3=0; cos=0.
with torch.no_grad():
  for b in dl:
    b=move_to_device(b,dev); q=b['query']; f=s2.encode_factors(q['residual_gt'],q['residual_mask'],q.get('audio_emotion',q.get('audio'))); a=s3(q['audio_emotion'],q['mask']); t=s3.teacher(q['residual_gt'],q['residual_mask']); n=q['emotion_id'].numel(); N+=n; e2+=(f['emotion_logits'].argmax(-1)==q['emotion_id']).sum().item(); i2+=(f['intensity_logits'].argmax(-1)==q['intensity_id']).sum().item(); e3+=(a['emotion_logits'].argmax(-1)==q['emotion_id']).sum().item(); i3+=(a['intensity_logits'].argmax(-1)==q['intensity_id']).sum().item(); cos+=F.cosine_similarity(a['global'],t['global'],dim=-1).sum().item()
print(f'samples={N} stage2_emotion_acc={e2/N:.6f} stage2_intensity_acc={i2/N:.6f} stage3_emotion_acc={e3/N:.6f} stage3_intensity_acc={i3/N:.6f} stage3_global_cosine={cos/N:.6f}')
