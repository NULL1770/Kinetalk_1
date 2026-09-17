import copy
import pytest
from scripts.audit_scaled_centered_probe import validate_metric_pair
from scripts.train_projection_scaled_centered_probe import SCHEMA, LOSS

def test_scaled_comparison_only_allows_loss_weights_to_differ():
    uniform={'schema':SCHEMA,'loss':LOSS,'args':{'epochs':8,'arm':'uniform','output':'u'},
        'teacher_probability':.5,'decode_steps':12,'trainable':['local_projection.weight'],
        'mean_anchoring_loss':False,'selected_weight_sha256':'u','channel_metric':{'training':'same'}}
    scaled=copy.deepcopy(uniform);scaled['args'].update(arm='train_rms',output='s');scaled['selected_weight_sha256']='s'
    validate_metric_pair(scaled,uniform)
    for key,value in [('teacher_probability',0),('channel_metric',{'training':'dev'}),('trainable',['renderer'])]:
        changed=copy.deepcopy(scaled);changed[key]=value
        with pytest.raises(ValueError):validate_metric_pair(changed,uniform)
