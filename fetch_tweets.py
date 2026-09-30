"""Fetch pinned, checksum-verified plain text from the TweetEval authors."""
from pathlib import Path
import hashlib
import json
import urllib.request
from urllib.parse import urlparse
import io
import pandas as pd

ROOT=Path(__file__).resolve().parent

def fetch():
    receipt=json.loads((ROOT/'data/tweeteval_provenance.json').read_text())
    destination=ROOT/'data/tweets.csv'
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=receipt['prepared_sha256']:
            raise ValueError('Existing tweet input differs; it was left untouched')
        print('Verified data/tweets.csv')
        return
    contents={}
    for name,item in receipt['files'].items():
        with urllib.request.urlopen(item['url'],timeout=60) as response:
            if urlparse(response.geturl()).hostname!='raw.githubusercontent.com' or response.headers.get_content_type()!='text/plain':
                raise ValueError('Unexpected download host or file type')
            content=response.read(10_000_001)
        if len(content)>10_000_000 or b'\x00' in content or hashlib.sha256(content).hexdigest()!=item['sha256']:
            raise ValueError('Unverified TweetEval bytes; no input was written')
        contents[name]=content.decode('utf-8').splitlines()
    rows=[]
    for split,count in [('train',45615),('val',2000),('test',12284)]:
        texts,labels=contents[split+'_text.txt'],contents[split+'_labels.txt']
        if len(texts)!=count or len(labels)!=count or set(labels)!={'0','1','2'} or not all(t.strip() for t in texts):
            raise ValueError('Unexpected tweet schema or row count')
        rows.extend({'source_id':f'{split}:{i}','official_split':split,'text':text,'label':int(label)} for i,(text,label) in enumerate(zip(texts,labels)))
    data=pd.DataFrame(rows).to_csv(index=False).encode()
    if hashlib.sha256(data).hexdigest()!=receipt['prepared_sha256']:
        raise ValueError('Converted checksum mismatch; no input was written')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(data)
    print('Downloaded and verified data/tweets.csv')

if __name__=='__main__':
    fetch()
