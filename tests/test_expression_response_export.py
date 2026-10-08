import numpy as np
import pytest
import torch
from scripts.evaluate_expression_response import export_render_input


@pytest.mark.parametrize('mask,start,stop',[
    ([False,True,True,True,False],1,4),
    ([True,True,False,True,True],0,2),
    ([True,False,True,True,True],2,5),
    ([True,True,True,True,True],0,5)])
def test_native_display_span_preserves_coefficients_and_audio_clock(tmp_path,mask,start,stop):
    valid=torch.tensor(mask);times=torch.arange(5,dtype=torch.float64)/25+2.0
    motion=torch.arange(260,dtype=torch.float32).reshape(5,52)/100
    path=tmp_path/'display.npz';original=motion.clone()
    export_render_input(path,motion=motion,base=motion+1,prior=motion+2,posterior=motion+3,
        valid=valid,times=times,channels=torch.ones(52,dtype=torch.bool),clip_id='fixed')
    assert torch.equal(motion,original)
    with np.load(path,allow_pickle=False) as z:
        assert z['source_start_frame']==start and z['source_stop_frame']==stop
        np.testing.assert_array_equal(z['times'],times[start:stop].numpy())
        np.testing.assert_array_equal(z['motions'][0],original[start:stop].numpy())
        assert z['valid'].all() and z['motions'][3].max()>1 # No implicit clipping.


def test_display_rejects_clock_gap_and_empty_mask(tmp_path):
    y=torch.zeros(5,52);args=dict(motion=y,base=y,prior=y,posterior=y,
        valid=torch.ones(5,dtype=torch.bool),times=torch.tensor([0.,.04,.08,.16,.20]),
        channels=torch.ones(52,dtype=torch.bool),clip_id='fixed')
    with pytest.raises(ValueError,match='25fps'):export_render_input(tmp_path/'gap.npz',**args)
    args['valid'].zero_()
    with pytest.raises(ValueError,match='No observed'):export_render_input(tmp_path/'empty.npz',**args)
