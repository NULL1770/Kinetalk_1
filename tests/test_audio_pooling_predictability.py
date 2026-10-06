import torch
from scripts.audit_audio_pooling_predictability import masked_population_std


def test_pool_std_ignores_native_gaps_and_padding_without_filling_statistics():
    x=torch.tensor([[[1.,4.],[3.,4.],[float('nan'),float('nan')],[5.,4.]]])
    valid=torch.tensor([[True,True,False,True]]);mean=torch.tensor([[3.,4.]])
    result=masked_population_std(x,valid,mean)
    torch.testing.assert_close(result,torch.tensor([[8/3,0.]]).sqrt())
    padded=torch.cat([x,torch.full((1,3,2),float('nan'))],dim=1)
    mask=torch.cat([valid,torch.zeros(1,3,dtype=torch.bool)],dim=1)
    torch.testing.assert_close(masked_population_std(padded,mask,mean),result,rtol=0,atol=0)
