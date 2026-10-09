import torch
from scripts.audit_many_speaker_style import select_queries, summaries
from scripts.train_expression_response import development_fold


def test_selection_covers_people_emotions_but_never_uses_fit_queries_or_test():
    def make(people):
        rows=[(s,e,j) for s in people for e in range(8) for j in range(24)]
        return dict(clip_id=[f'{s}_{e}_{j}' for s,e,j in rows],
            speaker_id=torch.tensor([s for s,e,j in rows]),
            emotion_id=torch.tensor([e for s,e,j in rows]),
            sentence_id=[str(j) for s,e,j in rows])
    data=dict(fit_sids=list(range(22)),splits={'train':make(range(22)),'validation':make(range(22,25))})
    first,pool,fold=select_queries(data);second,_,_=select_queries(data)
    assert first==second and len(first)==25*8
    fit=set(fold['train'])
    assert all(split!='train' or i not in fit for split,i,_ in first)
    assert {role for _,_,role in first}=={'unseen_internal','seen_held_sentence','unseen_development'}
    assert len([v for v in first if v[2]=='unseen_internal'])==16
    assert len([v for v in first if v[2]=='unseen_development'])==24


def test_summary_does_not_average_identifiers_or_treat_missing_correlations_as_zero():
    r=[dict(clip_id='a',source=1,target=2,role='seen_held_sentence',emotion=0,method='parent',policy='raw',kind='cross_AB',mae=.1,corr=None),
       dict(clip_id='b',source=1,target=2,role='seen_held_sentence',emotion=0,method='parent',policy='raw',kind='cross_AB',mae=.3,corr=.7)]
    result=summaries(r)
    assert result=={'n':2,'metrics':{'mae':.2,'corr':.7}}
