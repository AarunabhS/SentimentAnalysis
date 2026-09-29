import pandas as pd
import pytest
from analysis import load_data, prepare_features, score_texts, sentiment_pipeline, vader_label


def test_neutral_is_not_positive_and_negation_is_retained():
    assert vader_label(0.0)=="neutral"
    assert vader_label(0.05)=="positive"
    assert vader_label(-0.05)=="negative"
    frame=pd.DataFrame({"text":["This is not good."]})
    assert "not" in prepare_features(frame).text.iloc[0]
    assert score_texts(frame).sentiment.iloc[0]=="negative"


def test_no_test_vocabulary_is_learned():
    frame=pd.DataFrame({"text":["excellent good film","good excellent service","bad awful film","awful bad service"]*3})
    y=[1,1,0,0]*3
    model=sentiment_pipeline().fit(frame,y)
    model.predict_proba(pd.DataFrame({"text":["heldoutuniquetoken good"]}))
    vocab=model.named_steps["columntransformer"].named_transformers_["text"].vocabulary_
    assert "heldoutuniquetoken" not in vocab


def test_duplicate_text_cannot_cross_splits(tmp_path):
    frame=pd.DataFrame({"text":[f"Sentence number {i}" for i in range(40)],"label":[0,1]*20})
    frame=pd.concat([frame,frame.iloc[:5]],ignore_index=True)
    p=tmp_path/"reviews.csv"
    frame.to_csv(p,index=False)
    X,y,_,info=load_data(p)
    assert len(X)==40 and len(y)==40 and info["duplicates_removed"]==5


def test_blank_and_missing_text_fail_clearly():
    for value in [" ",None]:
        with pytest.raises(ValueError,match="Text values"):
            prepare_features(pd.DataFrame({"text":[value]}))
