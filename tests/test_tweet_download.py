from pathlib import Path
import io
from email.message import Message
import pytest
import fetch_tweets


def setup(tmp_path,monkeypatch):
    receipt=(fetch_tweets.ROOT/'data/tweeteval_provenance.json').read_bytes()
    (tmp_path/'data').mkdir()
    (tmp_path/'data/tweeteval_provenance.json').write_bytes(receipt)
    monkeypatch.setattr(fetch_tweets,'ROOT',tmp_path)


def test_differing_local_tweets_are_preserved(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch)
    path=tmp_path/'data/tweets.csv'
    path.write_bytes(b'my original local rows')
    with pytest.raises(ValueError,match='left untouched'):
        fetch_tweets.fetch()
    assert path.read_bytes()==b'my original local rows'


def test_unverified_tweet_download_cannot_create_input(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch)
    class Response(io.BytesIO):
        headers=Message()
        headers['Content-Type']='text/plain'
        def geturl(self):
            return 'https://raw.githubusercontent.com/cardiffnlp/tweeteval/data.txt'
    monkeypatch.setattr(fetch_tweets.urllib.request,'urlopen',lambda *args,**kwargs:Response(b'tampered text'))
    with pytest.raises(ValueError,match='Unverified'):
        fetch_tweets.fetch()
    assert not (tmp_path/'data/tweets.csv').exists()
