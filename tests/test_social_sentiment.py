import pandas as pd
import pytest
from phase2 import load


def sample():
    return pd.DataFrame([{'source_id':f'{split}:{i}','official_split':split,'text':f'{split} unique sentence {i}','label':i%3}
                         for split,n in [('train',36),('val',9),('test',9)] for i in range(n)])


def test_test_overlap_is_removed_from_training_without_removing_test(tmp_path):
    frame=sample()
    frame.loc[0,'text']=frame.loc[frame.official_split=='test','text'].iloc[0]
    path=tmp_path/'tweets.csv'
    frame.to_csv(path,index=False)
    X,y,cleaned,splits,info=load(path)
    assert len(splits['test'])==9
    assert 'train:0' not in set(cleaned.source_id)
    assert 1 in set(y.iloc[splits['test']])


def test_invalid_human_label_is_rejected(tmp_path):
    frame=sample()
    frame.loc[0,'label']=3
    path=tmp_path/'tweets.csv'
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='0/1/2'):
        load(path)
