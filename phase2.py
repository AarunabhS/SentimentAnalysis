"""Human-labelled three-way social sentiment with an explicit domain-shift test."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'results/phase2/local/mpl-cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parent/'results/phase2/local/cache'))
import argparse
import importlib.metadata
import platform
import hashlib
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import analysis
from workflow import input_metadata, json_write
from review_tools import fit_sigmoid, reliability_table, plot_reliability, wilson

ROOT = Path(__file__).resolve().parent
LABELS = ['negative', 'neutral', 'positive']


def pipeline():
    transform = ColumnTransformer([('words', TfidfVectorizer(ngram_range=(1,2), min_df=3,
                    max_features=18000, sublinear_tf=True, strip_accents='unicode'), 'text')])
    return make_pipeline(transform, LogisticRegression(C=2, max_iter=450, random_state=42))


def load(path):
    frame = pd.read_csv(path)
    required = {'text','label','official_split','source_id'}
    if not required.issubset(frame):
        raise ValueError('Tweet data needs text, label, official_split, source_id')
    frame['text'] = analysis.prepare_features(frame).text
    if not frame.label.isin([0,1,2]).all() or not frame.official_split.isin(['train','val','test']).all():
        raise ValueError('Tweet labels must be 0/1/2 and splits train/val/test')
    if frame.source_id.duplicated().any():
        raise ValueError('Source IDs must be unique')
    frame['key'] = frame.text.str.casefold()
    test_keys = set(frame.loc[frame.official_split=='test','key'])
    validation_keys = set(frame.loc[frame.official_split=='val','key'])
    # Preserve all official test rows. Remove overlap from earlier splits using text only.
    train = frame.loc[(frame.official_split=='train') & ~frame.key.isin(test_keys | validation_keys)].copy()
    conflicts = train.groupby('key').label.transform('nunique') > 1
    train = train.loc[~conflicts].drop_duplicates('key')
    validation = frame.loc[(frame.official_split=='val') & ~frame.key.isin(test_keys)].drop_duplicates('key')
    test = frame.loc[frame.official_split=='test']
    cleaned = pd.concat([train, validation, test], ignore_index=True)
    source = cleaned.official_split
    train_indices = np.flatnonzero(source.to_numpy()=='train')
    fit, cal = train_test_split(train_indices, test_size=0.25, stratify=cleaned.label.iloc[train_indices], random_state=42)
    splits = {'fit':fit, 'calibration':cal, 'validation':np.flatnonzero(source.to_numpy()=='val'),
              'test':np.flatnonzero(source.to_numpy()=='test')}
    key_sets = [set(cleaned.key.iloc[indices]) for indices in splits.values()]
    assert all(not key_sets[a] & key_sets[b] for a in range(4) for b in range(a))
    assert all(set(cleaned.label.iloc[indices]) == {0,1,2} for indices in splits.values())
    return analysis.prepare_features(cleaned), cleaned.label.astype(int), cleaned, splits, {
        'rows_read':len(frame), 'rows_used':len(cleaned), 'earlier_rows_removed_for_overlap_or_ambiguity':len(frame)-len(cleaned),
        'test_rows_preserved':len(test), 'test_duplicate_texts':int(test.key.duplicated().sum())}


def metrics(y, probabilities):
    predicted = np.asarray(probabilities).argmax(axis=1)
    return {'accuracy':float(accuracy_score(y,predicted)), 'macro_f1':float(f1_score(y,predicted,average='macro',zero_division=0)),
            'macro_recall':float(recall_score(y,predicted,average='macro',zero_division=0)),
            'log_loss':float(log_loss(y,np.clip(probabilities,1e-12,1-1e-12),labels=[0,1,2])),
            'multiclass_brier':float(np.mean(np.sum((np.eye(3)[np.asarray(y)]-probabilities)**2,axis=1)))}


def predict(frame, model):
    probabilities = model.predict_proba(analysis.prepare_features(frame))
    output = pd.DataFrame(probabilities, columns=[name+'_score' for name in LABELS])
    output['predicted_label'] = probabilities.argmax(axis=1)
    output['predicted_sentiment'] = output.predicted_label.map(dict(enumerate(LABELS)))
    return output


def run(data=None, output=None):
    started = time.monotonic()
    data = ROOT / 'data/tweets.csv' if data is None else Path(data)
    output = ROOT / 'results/phase2' if output is None else Path(output)
    output.mkdir(parents=True,exist_ok=True)
    X,y,frame,splits,info = load(data)
    fit,cal,val,test = (splits[name] for name in ['fit','calibration','validation','test'])
    base = pipeline().fit(X.iloc[fit],y.iloc[fit])
    models = {'prior_baseline':DummyClassifier(strategy='prior').fit(X.iloc[fit],y.iloc[fit]),
              'tweet_tfidf':base, 'tweet_tfidf_sigmoid':fit_sigmoid(base,X.iloc[cal],y.iloc[cal])}
    records = [{'model':name,'split':'validation',**metrics(y.iloc[val],model.predict_proba(X.iloc[val]))} for name,model in models.items()]
    selected = max([row for row in records if row['model']!='prior_baseline'],key=lambda row:(row['macro_f1'],-row['log_loss']))['model']
    model = models[selected]
    probability = model.predict_proba(X.iloc[test])
    predicted = probability.argmax(axis=1)
    for name,estimator in models.items():
        records.append({'model':name,'split':'test',**metrics(y.iloc[test],estimator.predict_proba(X.iloc[test]))})
    vader = analysis.score_texts(X.iloc[test]).sentiment.map({'negative':0,'neutral':1,'positive':2}).to_numpy()
    vader_metrics = {'accuracy':float(accuracy_score(y.iloc[test],vader)),
                     'macro_f1':float(f1_score(y.iloc[test],vader,average='macro',zero_division=0)),
                     'macro_recall':float(recall_score(y.iloc[test],vader,average='macro',zero_division=0))}
    records.append({'model':'vader_rules','split':'test',**vader_metrics})
    pd.DataFrame(records).to_csv(output/'metrics.csv',index=False,float_format='%.8f')
    results = predict(X.iloc[test],model)
    results.insert(0,'source_id',frame.source_id.iloc[test].to_numpy())
    results['text_sha256'] = frame.text.iloc[test].map(lambda value:hashlib.sha256(value.encode()).hexdigest()).to_numpy()
    results['actual'], results['vader_label'] = y.iloc[test].to_numpy(), vader
    results.to_csv(output/'predictions.csv',index=False,float_format='%.8f')
    results.loc[results.actual!=results.predicted_label].to_csv(output/'errors.csv',index=False,float_format='%.8f')
    # Raw text stays local; reviewers can reconstruct by pinned source_id after fetching data.
    local = output/'local'
    local.mkdir(exist_ok=True)
    review = results.copy()
    review['text'] = frame.text.iloc[test].to_numpy()
    review.loc[review.actual!=review.predicted_label].to_csv(local/'error_text.csv',index=False,float_format='%.8f')
    assignments = np.full(len(y),'',dtype=object)
    for name,indices in splits.items():
        assignments[indices] = name
    pd.DataFrame({'source_id':frame.source_id,'split':assignments,'label':y}).to_csv(output/'splits.csv',index=False)
    classes = []
    for label,name in enumerate(LABELS):
        actual_positive, predicted_positive = y.iloc[test].to_numpy()==label, predicted==label
        tp = int((actual_positive & predicted_positive).sum())
        classes.append({'label':name,'support':int(actual_positive.sum()),'predicted_rows':int(predicted_positive.sum()),
                        'recall':wilson(tp,int(actual_positive.sum())), 'precision':wilson(tp,int(predicted_positive.sum()))})
    json_write(output/'class_intervals.json',{'level':0.95,'classes':classes,'assumption':'Row independence; repeated text is clustered in bootstrap below'})
    # Cluster duplicate official-test text; do not treat repeated posts as independent evidence.
    codes,_ = pd.factorize(frame.key.iloc[test])
    clusters = [np.flatnonzero(codes==number) for number in np.unique(codes)]
    rng, draws = np.random.default_rng(42), []
    for _ in range(200):
        rows = np.concatenate([clusters[number] for number in rng.integers(0,len(clusters),len(clusters))])
        draws.append(metrics(y.iloc[test].to_numpy()[rows],probability[rows]))
    json_write(output/'uncertainty.json',{'replicates':200,'method':'Cluster bootstrap of identical normalized test text',
        'intervals':{key:{'low':float(np.quantile([draw[key] for draw in draws],.025)),
                          'high':float(np.quantile([draw[key] for draw in draws],.975))} for key in draws[0]}})
    tables=[]
    for name in ['tweet_tfidf','tweet_tfidf_sigmoid']:
        scores = models[name].predict_proba(X.iloc[test])
        tables.extend(reliability_table(scores.argmax(axis=1)==y.iloc[test].to_numpy(),scores.max(axis=1),name,scheme=scheme) for scheme in ['quantile','uniform'])
    plot_reliability(tables,output,'Social sentiment: top-class confidence reliability')
    cm = confusion_matrix(y.iloc[test],predicted,labels=[0,1,2])
    fig,ax = plt.subplots(figsize=(6,4.5),layout='constrained')
    ax.imshow(cm,cmap='Blues')
    for (a,b),value in np.ndenumerate(cm):
        ax.text(b,a,str(value),ha='center',va='center',color='white' if value>cm.max()/2 else '#172635')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted',ylabel='Human label',title='Official social-post test set')
    fig.savefig(output/'confusion.png',dpi=150)
    plt.close(fig)
    heuristics = {'negation_words':frame.text.iloc[test].str.contains(r'\b(?:not|never|no|cannot|can.t)\b',case=False,regex=True),
                  'contrast_connective':frame.text.iloc[test].str.contains(r'\b(?:but|however|although|though)\b',case=False,regex=True),
                  'sarcasm_hashtag':frame.text.iloc[test].str.contains(r'#(?:sarcasm|irony)\b',case=False,regex=True)}
    groups=[]
    for name,mask in heuristics.items():
        positions=np.flatnonzero(mask.to_numpy())
        if len(positions):
            groups.append({'heuristic':name,'rows':len(positions),**metrics(y.iloc[test].to_numpy()[positions],probability[positions])})
    pd.DataFrame(groups).to_csv(output/'diagnostic_groups.csv',index=False,float_format='%.8f')
    # Evaluate the existing review model on negative/positive tweets only; neutrality stays explicit.
    old = joblib.load(ROOT/'results/model.joblib')
    binary = y.iloc[test].to_numpy()!=1
    review_scores = old['model'].predict_proba(X.iloc[test].iloc[np.flatnonzero(binary)])[:,1]
    actual_binary = (y.iloc[test].to_numpy()[binary]==2).astype(int)
    old_metrics=json_read(ROOT/'results/metrics.json')
    json_write(output/'domain_shift.json',{'review_model_uci_accuracy':old_metrics['test']['accuracy'],
        'review_model_binary_tweet_accuracy':float(accuracy_score(actual_binary,review_scores>=old['threshold'])),
        'binary_tweet_rows':int(binary.sum()),'neutral_rows_excluded':int((~binary).sum()),
        'interpretation':'Different class prevalence and datasets; this is a transfer diagnostic, not a controlled causal estimate of domain shift.'})
    development = np.concatenate([fit,cal,val])
    stability=[]
    for seed in [43,44,45]:
        train,validation=train_test_split(development,test_size=.2,stratify=y.iloc[development],random_state=seed)
        fitting,calibrating=train_test_split(train,test_size=.25,stratify=y.iloc[train],random_state=seed)
        assert not set(train)&set(test) and not set(validation)&set(test)
        fitted=pipeline().fit(X.iloc[fitting],y.iloc[fitting])
        if selected.endswith('_sigmoid'):
            fitted=fit_sigmoid(fitted,X.iloc[calibrating],y.iloc[calibrating])
        stability.append({'seed':seed,**metrics(y.iloc[validation],fitted.predict_proba(X.iloc[validation]))})
    pd.DataFrame(stability).to_csv(output/'validation_stability.csv',index=False,float_format='%.8f')
    joblib.dump({'model':model,'labels':LABELS,'phase':2},output/'model.joblib')
    examples=ROOT/'examples/black_friday_posts.csv'
    if examples.exists():
        predict(pd.read_csv(examples),model).to_csv(output/'authored_examples.csv',index=False,float_format='%.8f')
    summary={'project':'sentiment',**input_metadata(data),**info,'selected_model':selected,
        'label_mapping':dict(enumerate(LABELS)),'selection':'validation macro F1, log loss breaks ties',
        'splits':{name:{'rows':len(indices),'class_counts':y.iloc[indices].value_counts().sort_index().to_dict()} for name,indices in splits.items()},
        'test':next(row for row in records if row['model']==selected and row['split']=='test'),
        'vader_test':vader_metrics,'runtime_seconds':round(time.monotonic()-started,2),'seed':42,'python':platform.python_version(),
        'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','matplotlib']},
        'limits':'Human labels support general social sentiment, including neutral; no Black Friday population claim. '
                 'Heuristic contrast/negation/sarcasm flags are not human annotations of those properties. '
                 'Mixed sentiment is not a separate gold class. Raw text remains local; licence is unspecified upstream.'}
    json_write(output/'metrics.json',summary)
    t=summary['test']
    (output/'REPORT.md').write_text(f'''# Three-way social sentiment

Human-labelled TweetEval / SemEval-2017 social posts: negative, neutral, and positive.
All {len(test):,} official test rows are retained. Earlier-split overlaps are removed using text only;
{info['earlier_rows_removed_for_overlap_or_ambiguity']:,} earlier rows are excluded for overlap, duplication, or training-label ambiguity.
Training-only TF-IDF; an independent training reserve fits sigmoid calibration. Official validation selects the model.

**{selected}**: test accuracy {t['accuracy']:.1%}, macro F1 {t['macro_f1']:.4f}, macro recall {t['macro_recall']:.4f}, log loss {t['log_loss']:.4f}.
VADER: accuracy {vader_metrics['accuracy']:.1%}, macro F1 {vader_metrics['macro_f1']:.4f}. Neutral is a genuine human-labelled class.

![Three-way confusion matrix](confusion.png)
![Top-class confidence calibration](reliability.png)

[Class support and Wilson intervals](class_intervals.json), [cluster bootstrap intervals](uncertainty.json),
[all metrics](metrics.csv), [validation stability](validation_stability.csv), [error source IDs](errors.csv),
and [review-to-tweet transfer](domain_shift.json) make results auditable.
[Diagnostic groups](diagnostic_groups.csv) use lexical heuristics; they cannot establish performance on human-labelled sarcasm or mixed sentiment.

This is historical general social sentiment, not a Black Friday survey. Authored examples are illustrations.
TweetEval's dataset licence is unspecified. Raw posts and error text remain local; the repository publishes hashes,
source row IDs, derived results, and a verified downloader. Models, calibration, and splits never tune on test labels.
This is a benchmark, with unknown author/topic/time overlap and no newly collected external validation.
''')
    print(f'sentiment: {selected}; accuracy={t["accuracy"]:.4f}; macro F1={t["macro_f1"]:.4f}; elapsed={summary["runtime_seconds"]}s',flush=True)
    return summary


def json_read(path):
    import json
    return json.loads(path.read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    run(args.data,args.output)
