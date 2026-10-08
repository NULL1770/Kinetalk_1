import torch
from scripts.factor_swap_expression_response import distribution_statistics,swap


def test_latent_statistics_mask_dimension_and_variance():
    p={'g_mean':torch.zeros(2,2),'g_logvar':torch.zeros(2,2),
       'u_mean':torch.zeros(2,3,2),'u_logvar':torch.zeros(2,3,2),
       'u_mask':torch.tensor([[True,True,False],[True,False,False]])}
    q={k:v.clone() for k,v in p.items()};q['g_mean']+=2;q['u_mean']+=2
    q['u_mean'][~p['u_mask']]=float('nan')
    raw={'g':torch.zeros(2,4),'u':torch.zeros(2,3,4)};raw['u'][...,-2:]=3
    stats=distribution_statistics(q,p,raw)
    for part in ('g','u'):
        torch.testing.assert_close(stats[part]['mean_error_squared'],torch.full((2,2),4.))
        torch.testing.assert_close(stats[part]['kl_mean_component'],torch.full((2,2),2.))
        assert not stats[part]['kl_variance_component'].any()
    assert (stats['u']['prior_raw_high_clamped_fraction']==1).all()
    assert not stats['g']['prior_raw_high_clamped_fraction'].any()


def test_swaps_preserve_only_declared_factors():
    p={'g_mean':torch.zeros(1,2),'g_logvar':torch.zeros(1,2),'u_mean':torch.zeros(1,3,2),
       'u_logvar':torch.zeros(1,3,2),'u_mask':torch.ones(1,3,dtype=torch.bool)}
    q={k:v.clone() if v.dtype==torch.bool else v+1 for k,v in p.items()}
    for mode,prefix in [('q_g_p_u','g_'),('p_g_q_u','u_')]:
        z=swap(q,p,mode)
        for k in z:assert torch.equal(z[k],q[k] if k.startswith(prefix) else p[k])
