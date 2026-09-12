import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from kinetalk_b0.data import B0ResidualDataset, collate_b0_residual
from kinetalk_b0.models import Stage1Model, Stage2Model, Stage3Model, Stage4Model
from kinetalk_b0.utils import load_yaml, load_checkpoint, move_to_device

cfg=load_yaml('configs/train.yaml'); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds=B0ResidualDataset(cfg,split='train',random_crop=False); dl=DataLoader(ds,batch_size=32,num_workers=0,collate_fn=collate_b0_residual)
s1=Stage1Model(cfg).to(dev); load_checkpoint(cfg['paths']['stage1_ckpt'],s1,map_location=dev); s1.eval()
s2=Stage2Model(cfg).to(dev); load_checkpoint(cfg['paths']['stage2_ckpt'],s2,map_location=dev,strict=True); s2.eval()
s3=Stage3Model(cfg,s2).to(dev); load_checkpoint(cfg['paths']['stage3_ckpt'],s3,map_location=dev,strict=True); s3.eval()
s4=Stage4Model(cfg,s1,s2,s3).to(dev); load_checkpoint(cfg['paths']['stage4_ckpt'],s4,map_location=dev,strict=True); s4.eval()
styles=[]; contents=[]; labels=[]; intens=[]; mouth_corr=[]; swap_delta=[]
with torch.no_grad():
  for b in dl:
    b=move_to_device(b,dev); q=b['query']; f=s2.encode_factors(q['residual_gt'],q['residual_mask'],q.get('audio_emotion',q.get('audio'))); styles.append(f['style']); contents.append(q['content'].mean(1)); labels.append(q['emotion_id']); intens.append(q['intensity_id'])
    cond=s4.conditions(b); final,_=s4.render(b,cond,steps=int(cfg['model'].get('render_steps',4))); alt=dict(cond); alt['style']=f['style']; alt_final,_=s4.render(b,alt,steps=int(cfg['model'].get('render_steps',4))); swap_delta.append((final-alt_final).abs().mean())
S=torch.cat(styles); C=torch.cat(contents); Y=torch.cat(labels); I=torch.cat(intens); N=S.shape[0]; split=max(1,int(N*0.8)); perm=torch.randperm(N); tr,te=perm[:split],perm[split:]
def probe(X, target, classes):
  X=X.float(); mu=X[tr].mean(0,keepdim=True); sd=X[tr].std(0,keepdim=True).clamp_min(1e-4); X=(X-mu)/sd
  W=torch.zeros(X.shape[1],classes,device=dev,requires_grad=True); b=torch.zeros(classes,device=dev,requires_grad=True); opt=torch.optim.LBFGS([W,b],max_iter=80,line_search_fn='strong_wolfe')
  def closure():
    opt.zero_grad(); loss=F.cross_entropy(X[tr]@W+b,target[tr]); loss.backward(); return loss
  opt.step(closure)
  acc=(X[te]@W+b).argmax(-1).eq(target[te]).float().mean().item(); return acc
print('N',N,'style_to_emotion_probe_acc',probe(S,Y,len(cfg['data']['emotion_classes'])),'style_to_intensity_probe_acc',probe(S,I,int(cfg['data']['num_intensity_levels'])),'chance_emotion',1/len(cfg['data']['emotion_classes']),'chance_intensity',1/int(cfg['data']['num_intensity_levels']),'mean_style_swap_delta',torch.stack(swap_delta).mean().item())
