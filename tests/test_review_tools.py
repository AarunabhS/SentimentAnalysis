"""Guard independent calibration, tied-score budgets, and cluster uncertainty."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from review_tools import partition_training, fit_sigmoid, threshold_for_budget, wilson, bootstrap_intervals, reliability_table
from workflow import make_splits


def test_four_partitions_preserve_group_isolation():
    y=pd.Series(np.tile([0,1],120))
    groups=pd.Series(np.repeat(np.arange(120),2))
    parts=partition_training(make_splits(y,groups),y,groups=groups)
    group_sets=[set(groups.iloc[rows]) for rows in parts.values()]
    assert all(not group_sets[a]&group_sets[b] for a in range(4) for b in range(a))
    assert set(np.concatenate(list(parts.values())))==set(range(len(y)))


def test_sigmoid_does_not_refit_preprocessing_on_calibration():
    X=pd.DataFrame({'x':np.linspace(-2,2,100)})
    y=pd.Series((X.x>0).astype(int))
    base=make_pipeline(StandardScaler(),LogisticRegression()).fit(X,y)
    original_mean=base[0].mean_.copy()
    calibration=pd.DataFrame({'x':np.linspace(10,30,100)})
    calibrated=fit_sigmoid(base,calibration,pd.Series(np.tile([0,1],50)))
    assert np.array_equal(base[0].mean_,original_mean)
    assert calibrated.predict_proba(X).shape==(100,2)


def test_budget_does_not_split_ties_or_exceed_validation_capacity():
    scores=np.array([.9,.9,.8,.4,.1])
    assert (scores>=threshold_for_budget(scores,.2)).sum()==0
    assert (scores>=threshold_for_budget(scores,.4)).sum()==2
    assert (scores>=threshold_for_budget(scores,1)).sum()==5


def test_wilson_reports_absent_support_and_uncertainty():
    assert wilson(0,0)['estimate'] is None
    interval=wilson(14,16)
    assert interval['low']<.875<interval['high']<1
    assert wilson(3,3)['high']==1
    assert wilson(0,3)['low']==0


def test_constant_scores_reliability_keeps_all_rows():
    frame=reliability_table([0,1,1,0],[.5,.5,.5,.5],'constant')
    assert frame.rows.sum()==4 and len(frame)==1
    assert frame.iloc[0].observed_rate==.5


def test_cluster_bootstrap_retains_perfect_ranking():
    result=bootstrap_intervals([0,1,0,1],[.1,.9,.1,.9],.5,groups=['a','a','b','b'],repeats=20)
    assert 'Cluster bootstrap' in result['method']
    assert result['intervals']['average_precision']['low']==1
    assert result['replicates']==20
