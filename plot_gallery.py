"""Render colorful, auditable charts from saved phase-two results; never retrain.

Use: python plot_gallery.py [--results results/phase2] [--output results/charts]
PNG and SVG files are derived from the verified public outputs. No network access.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).absolute().parent/'results/charts/local/mpl-cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).absolute().parent/'results/charts/local/cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, precision_recall_fscore_support

ROOT=Path(__file__).absolute().parent
INK='#19353f'; MUTED='#526b73'; GRID='#dbe5e9'; PAPER='#f7fafb'
TEAL='#087f8c'; CORAL='#db5949'; PURPLE='#7655b8'; GOLD='#bd8616'; BLUE='#2866b0'
CONFIG={
 'Credit-Card-Fraud-Detection':{'title':'Fraud detection','subtitle':'Rare events. Explicit review costs.','accent':TEAL,
   'note':'Historical Worldline/ULB benchmark · 56,746 test rows / 74 frauds · thresholds chosen on validation',
   'plots':['fraud_precision_recall','fraud_review_budget','fraud_score_distribution']},
 'SentimentAnalysis':{'title':'Social sentiment','subtitle':'Three human-labelled classes. Visible mistakes.','accent':PURPLE,
   'note':'TweetEval official test · 12,284 posts · no raw post text · model selected on validation macro F1',
   'plots':['sentiment_confusion','sentiment_classes','sentiment_models']},
 'Predictive-Employee-Churn':{'title':'Employee churn','subtitle':'Stable signals. Honest uncertainty.','accent':CORAL,
   'note':'Retrospective HR benchmark · 2,399 test rows / 398 departures · importance is associative, not causal',
   'plots':['churn_importance','churn_departments','churn_calibration']},
 'Click-Through-Rate':{'title':'Click propensity','subtitle':'Ranking value under explicit sampling limits.','accent':BLUE,
   'note':'Downsampled click benchmark · 7,984 test contexts · observed rates do not establish population CTR',
   'plots':['ctr_gains','ctr_model_loss','ctr_position_errors']}}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,'axes.titleweight':'bold',
 'axes.labelsize':11,'axes.labelcolor':INK,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,
 'axes.edgecolor':GRID,'axes.spines.top':False,'axes.spines.right':False,'axes.facecolor':'white',
 'figure.facecolor':PAPER,'savefig.facecolor':PAPER,'svg.fonttype':'none','svg.hashsalt':'portfolio-charts-v1',
 'axes.axisbelow':True,'legend.frameon':False,'lines.linewidth':2.4})

def clean(ax,axis='y'):
    ax.grid(axis=axis,color=GRID,linewidth=.8,alpha=.8)
    ax.tick_params(length=0,pad=7)
    return ax

def title(ax,headline,subtitle):
    ax.set_title(headline,loc='left',pad=29)
    ax.text(0,1.025,subtitle,transform=ax.transAxes,color=MUTED,fontsize=10,ha='left',va='bottom')

def wilson(k,n):
    p=np.asarray(k,dtype=float)/np.asarray(n,dtype=float); z=1.959963984540054
    denom=1+z*z/n; centre=(p+z*z/(2*n))/denom
    half=z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/denom
    return np.maximum(0,centre-half),np.minimum(1,centre+half)

def bar_labels(ax,bars,format_value,axis='y',small=False):
    for bar in bars:
        if axis=='y':
            ax.annotate(format_value(bar.get_height()),(bar.get_x()+bar.get_width()/2,bar.get_height()),
                        xytext=(0,5),textcoords='offset points',ha='center',fontsize=9 if small else 10)
        else:
            ax.annotate(format_value(bar.get_width()),(bar.get_width(),bar.get_y()+bar.get_height()/2),
                        xytext=(6,0),textcoords='offset points',va='center',fontsize=10)

class Results:
    def __init__(self,path):
        self.path=Path(path); self.sources={}
        self.meta=self.json('metrics.json')
        self.pred=self.csv('predictions.csv')
        self.metrics=self.csv('metrics.csv')
        self.selected=self.meta['selected_model']
        assert len(self.pred)==self.meta['test'].get('rows',self.meta.get('test_rows_preserved'))
        if 'predicted_label' in self.pred:
            actual=self.pred.actual.to_numpy(); pred=self.pred.predicted_label.to_numpy()
            assert abs(np.mean(actual==pred)-self.meta['test']['accuracy'])<1e-7
        else:
            actual=self.pred.actual.to_numpy(); pred=self.pred.predicted.to_numpy()
            for name,value in [('tp',sum((actual==1)&(pred==1))),('fp',sum((actual==0)&(pred==1))),
                               ('fn',sum((actual==1)&(pred==0))),('tn',sum((actual==0)&(pred==0)))]:
                assert value==self.meta['test'][name],name
            assert abs(average_precision_score(actual,self.pred.selected_score)-self.meta['test']['average_precision'])<1e-4
    def remember(self,name):
        path=self.path/name
        self.sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def csv(self,name):return pd.read_csv(self.remember(name))
    def json(self,name):return json.loads(self.remember(name).read_text())

def fraud_precision_recall(ax,r):
    clean(ax)
    title(ax,'Finding rare fraud without flooding review','Test curves · AP = average precision')
    for name,label,color in [(r.selected,'Selected forest + sigmoid',TEAL),('weighted_logistic_sigmoid','Logistic + sigmoid',PURPLE)]:
        p,rec,_=precision_recall_curve(r.pred.actual,r.pred[name+'_score'])
        # Plot all saved score thresholds, without selecting or tuning on test labels.
        ap=float(r.metrics.loc[(r.metrics.model==name)&(r.metrics.split=='test'),'average_precision'].iloc[0])
        ax.plot(rec*100,p*100,color=color,label=f'{label} · AP {ap:.3f}')
    prevalence=r.meta['test']['positive_rate']*100
    ax.axhline(prevalence,color=MUTED,ls='--',lw=1.5,label=f'No-skill prevalence · {prevalence:.2f}%')
    m=r.meta['test']; ax.scatter(m['recall']*100,m['precision']*100,s=100,color=CORAL,marker='D',zorder=5)
    ax.annotate(f'Fixed F2 threshold\n{m["tp"]} found / {m["fp"]} false alerts',(m['recall']*100,m['precision']*100),
                xytext=(8,-30),textcoords='offset points',fontsize=10,color=INK)
    ax.set(xlim=(0,100),ylim=(0,105),xlabel='Recall: share of actual frauds found (%)',ylabel='Precision: share of flags that are fraud (%)')
    ax.legend(loc='lower left',fontsize=10)

def fraud_review_budget(ax,r):
    clean(ax,'x'); title(ax,'More review capacity means more false alerts','Frozen validation thresholds · test workload can differ')
    policies=r.csv('operating_policies.csv')
    rows=policies[(policies.split=='test')&(policies.validation_budget_fraction<=.010001)].sort_values('validation_budget_fraction')
    labels=['Selected F2 threshold']+[f'{v:.1%} validation budget' for v in rows.validation_budget_fraction]
    tp=np.r_[r.meta['test']['tp'],rows.tp]; fp=np.r_[r.meta['test']['fp'],rows.fp]
    positives=int(r.pred.actual.sum())
    assert np.all(rows.tp+rows.fn==positives) and np.all(rows.tp+rows.fp==rows.alerts)
    y=np.arange(len(labels)); ax.barh(y,tp,color=TEAL,height=.62,label='Actual fraud flagged')
    ax.barh(y,fp,left=tp,color=CORAL,height=.62,label='Legitimate transaction flagged')
    for i,(a,b) in enumerate(zip(tp,fp)):
        ax.text(a+b+10,i,f'{int(a)}/{positives} found · {int(b)} false alerts',va='center',fontsize=10)
    ax.set_yticks(y,labels); ax.invert_yaxis()
    ax.set(xlim=(0,max(tp+fp)*1.58),ylim=(5.15,-.65),xlabel='Transactions sent for investigation (count)')
    ax.legend(loc='lower right',fontsize=10)

def fraud_score_distribution(ax,r):
    clean(ax); title(ax,'Score separation, with the rare class visible','Cumulative share within each class · classes use separate denominators')
    for value,label,color in [(0,'Legitimate',BLUE),(1,'Fraud',CORAL)]:
        values=np.sort(r.pred.loc[r.pred.actual==value,'selected_score'].to_numpy())
        assert (values>0).all()
        indexes=np.unique(np.r_[0,np.linspace(0,len(values)-1,min(1800,len(values))).astype(int),len(values)-1])
        ax.step(values[indexes],100*(indexes+1)/len(values),where='post',color=color,label=f'{label} · n={len(values):,}')
    ax.axvline(r.meta['threshold'],color=TEAL,ls='--',lw=1.5,label=f'Fixed F2 threshold · {r.meta["threshold"]:.3f}')
    ax.set_xscale('log'); ax.set(xlabel='Saved fraud score (log scale)',ylabel='Share of class at or below score (%)',ylim=(0,105))
    ax.legend(loc='lower right',fontsize=10)

def sentiment_confusion(ax,r):
    counts=confusion_matrix(r.pred.actual,r.pred.predicted_label,labels=[0,1,2]); fractions=counts/counts.sum(axis=1,keepdims=True)
    assert counts.sum()==len(r.pred) and abs(np.trace(counts)/counts.sum()-r.meta['test']['accuracy'])<1e-7
    title(ax,'How the model confuses sentiment classes','Human labels · count and within-row percentage')
    cmap=LinearSegmentedColormap.from_list('sentiment',['#f3f1fb','#c0b0df',PURPLE,'#342550'])
    ax.imshow(fractions,cmap=cmap,vmin=0,vmax=1,aspect='auto')
    labels=['Negative','Neutral','Positive']
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f'{counts[i,j]:,}\n{fractions[i,j]:.1%}',ha='center',va='center',fontsize=13,
                    color='white' if fractions[i,j]>.52 else INK,fontweight='bold')
    ax.set_xticks(range(3),labels)
    ax.set_yticks(range(3),[f'{s} · n={n:,}' for s,n in zip(labels,counts.sum(axis=1))])
    ax.set(xlabel='Predicted sentiment',ylabel='Actual sentiment')
    ax.tick_params(length=0,pad=9); ax.grid(False)
    for spine in ax.spines.values():spine.set_visible(False)

def sentiment_classes(ax,r):
    clean(ax); title(ax,'Each sentiment class behaves differently','Selected model · official test labels')
    p,rec,f1,n=precision_recall_fscore_support(r.pred.actual,r.pred.predicted_label,labels=[0,1,2],zero_division=0)
    x=np.arange(3)
    for offset,values,label,color in [(-.24,p,'Precision',TEAL),(0,rec,'Recall',CORAL),(.24,f1,'F1',PURPLE)]:
        bars=ax.bar(x+offset,values*100,width=.22,color=color,label=label)
        bar_labels(ax,bars,lambda v:f'{v:.1f}%',small=True)
    ax.set_xticks(x,[f'{s}\nn={int(v):,}' for s,v in zip(['Negative','Neutral','Positive'],n)])
    ax.set(ylim=(0,105),ylabel='Class metric (%)'); ax.legend(loc='upper left',ncol=3,fontsize=10)

def sentiment_models(ax,r):
    clean(ax); title(ax,'A stronger baseline, still a difficult task','Same official test split · the selected model was chosen on validation, not test')
    names=['prior_baseline','vader_rules','tweet_tfidf','tweet_tfidf_sigmoid']
    labels=['Prior baseline','VADER rules','TF–IDF\n(selected)','TF–IDF + sigmoid\n(candidate)']
    rows=r.metrics[(r.metrics.split=='test')&r.metrics.model.isin(names)].set_index('model').loc[names]
    x=np.arange(4)
    for offset,column,label,color in [(-.17,'accuracy','Accuracy',TEAL),(.17,'macro_f1','Macro F1',PURPLE)]:
        bars=ax.bar(x+offset,rows[column]*100,width=.31,color=color,label=label)
        bar_labels(ax,bars,lambda v:f'{v:.1f}%',small=True)
    ax.set_xticks(x,labels); ax.set(ylim=(0,90),ylabel='Metric (%)'); ax.legend(loc='upper left',ncol=2,fontsize=10)

FEATURE_LABELS={'satisfaction_level':'Satisfaction','time_spend_company':'Tenure','number_project':'Project count',
 'average_montly_hours':'Monthly hours','last_evaluation':'Last evaluation','salary':'Salary band',
 'Work_accident':'Work accident','promotion_last_5years':'Recent promotion','department':'Department'}

def churn_importance(ax,r):
    clean(ax,'x'); title(ax,'Which signals consistently help the model?','Validation AP decrease · bars: mean · dots: three folds')
    summary=r.csv('importance_stability.csv').sort_values('mean_ap_decrease',ascending=False)
    folds=r.csv('importance_by_fold.csv'); y=np.arange(len(summary))
    ax.barh(y,summary.mean_ap_decrease*100,color=CORAL,alpha=.86,height=.62,label='Mean AP decrease')
    for i,row in enumerate(summary.itertuples()):
        values=folds[folds.feature==row.feature].sort_values('fold').ap_decrease.to_numpy()*100
        ax.scatter(values,i+np.array([-.13,0,.13]),color=PURPLE,s=27,zorder=5,edgecolors='white',lw=.5)
    ax.axvline(0,color=MUTED,lw=1)
    ax.set_yticks(y,[FEATURE_LABELS[v] for v in summary.feature]); ax.invert_yaxis()
    ax.set(xlabel='Drop in validation average precision (percentage points)',xlim=(-1.5,34))
    ax.legend(loc='lower right',fontsize=10)

def churn_departments(ax,r):
    clean(ax,'x'); title(ax,'Missed-departure rates vary, with wide uncertainty','Test missed/actual departures · Wilson 95% row intervals')
    data=r.csv('subgroup_errors.csv'); data=data[data.feature=='department'].sort_values('group')
    k=data.false_negatives.to_numpy(); n=data.positives.to_numpy(); values=k/n*100; low,high=wilson(k,n)
    y=np.arange(len(data)); ax.barh(y,values,color=CORAL,alpha=.38,height=.60)
    ax.errorbar(values,y,xerr=np.vstack([np.maximum(0,values-low*100),np.maximum(0,high*100-values)]),
                fmt='o',color=CORAL,ecolor=PURPLE,capsize=3,markersize=6,lw=1.5)
    labels={'RandD':'R&D','hr':'HR','product_mng':'Product management'}
    ax.set_yticks(y,[labels.get(v,v.title() if v!='IT' else v) for v in data.group]); ax.invert_yaxis()
    for i,(a,b,v) in enumerate(zip(k,n,values)):
        ax.text(52,i,f'{v:.1f}% · {a}/{b} missed',va='center',fontsize=10)
    ax.set(xlim=(0,82),xlabel='Missed-departure rate (%)'); ax.set_xticks([0,10,20,30,40,50])

def churn_calibration(ax,r):
    clean(ax); title(ax,'Calibration improves the overall probability loss','Uniform score bins · Wilson intervals · each bin can have limited support')
    data=r.csv('reliability.csv'); data=data[data.binning_scheme=='uniform']
    ax.plot([0,100],[0,100],color=MUTED,ls='--',lw=1.5,label='Ideal calibration')
    for name,label,color in [('random_forest','Raw forest',BLUE),('random_forest_sigmoid','Forest + sigmoid',CORAL)]:
        frame=data[data.model==name].sort_values('mean_score')
        y=frame.observed_rate.to_numpy()*100
        errors=np.vstack([np.maximum(0,y-frame.rate_low.to_numpy()*100),np.maximum(0,frame.rate_high.to_numpy()*100-y)])
        loss=float(r.metrics.loc[(r.metrics.model==name)&(r.metrics.split=='test'),'log_loss'].iloc[0])
        ax.errorbar(frame.mean_score*100,y,yerr=errors,color=color,fmt='o-',lw=2,capsize=2,
                    elinewidth=.9,alpha=.9,label=f'{label} · log loss {loss:.3f}')
    ax.set(xlim=(0,100),ylim=(0,103),xlabel='Mean predicted departure score in bin (%)',ylabel='Observed departures in bin (%)')
    ax.legend(loc='lower right',fontsize=10)

def ctr_gains(ax,r):
    clean(ax); title(ax,'Ranking concentrates the observed clicks','Untuned test ranking at fixed fractions · random ranking is the diagonal')
    data=r.csv('ranking_lift.csv'); x=np.r_[0,data.selection_fraction.to_numpy()*100]; y=np.r_[0,data.positive_capture_fraction.to_numpy()*100]
    ax.fill_between(x,x,y,color=BLUE,alpha=.12)
    ax.plot(x,y,'o-',color=BLUE,label='Selected model ranking')
    ax.plot([0,100],[0,100],color=MUTED,ls='--',lw=1.5,label='Random ranking expectation')
    row=data[np.isclose(data.selection_fraction,.1)].iloc[0]
    ax.scatter(10,row.positive_capture_fraction*100,s=105,color=GOLD,zorder=5,marker='D')
    ax.annotate(f'Top 10% → {row.positive_capture_fraction:.1%} of clicks\n{row.lift:.2f}× random-ranking lift',
                (10,row.positive_capture_fraction*100),xytext=(11,-19),textcoords='offset points',fontsize=10)
    ax.set(xlim=(0,100),ylim=(0,103),xlabel='Highest-scoring test contexts selected (%)',ylabel='Observed test clicks captured (%)')
    ax.legend(loc='lower right',fontsize=10)

def ctr_model_loss(ax,r):
    clean(ax,'x'); title(ax,'Calibration is tested, not assumed to help','Lower is better · choose on validation, then inspect test')
    names=['baseline','logistic_regression','logistic_regression_sigmoid','random_forest','random_forest_sigmoid']
    labels=['Prior baseline','Logistic (selected)','Logistic + sigmoid','Forest','Forest + sigmoid']; y=np.arange(5)
    for offset,split,label,color in [(-.17,'validation','Validation',PURPLE),(.17,'test','Test',BLUE)]:
        frame=r.metrics[r.metrics.split==split].set_index('model').loc[names]
        bars=ax.barh(y+offset,frame.log_loss,height=.29,color=color,label=label)
        bar_labels(ax,bars,lambda v:f'{v:.3f}',axis='x')
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set(xlim=(0,.60),ylim=(5.2,-.6),xlabel='Log loss (dimensionless)')
    ax.legend(loc='lower right',ncol=2,fontsize=10)

def ctr_position_errors(ax,r):
    clean(ax); title(ax,'One threshold, different placement errors','Selected validation threshold · rates use different, explicitly labelled denominators')
    data=r.csv('subgroup_errors.csv'); data=data[data.feature=='position'].sort_values('group'); x=np.arange(len(data))
    for offset,k,n,label,color in [(-.18,data.false_negatives.to_numpy(),data.positives.to_numpy(),'Missed-click rate (among clicks)',CORAL),
                                  (.18,data.false_positives.to_numpy(),(data.rows-data.positives).to_numpy(),'False-alert rate (among non-clicks)',BLUE)]:
        values=k/n*100; lo,hi=wilson(k,n)
        bars=ax.bar(x+offset,values,width=.32,color=color,label=label)
        ax.errorbar(x+offset,values,yerr=np.vstack([np.maximum(0,values-lo*100),np.maximum(0,hi*100-values)]),
                    fmt='none',color=INK,capsize=3,lw=1)
        for bar,v,upper in zip(bars,values,hi):
            ax.text(bar.get_x()+bar.get_width()/2,upper*100+2,f'{v:.1f}%',ha='center',fontsize=10)
    ax.set_xticks(x,[f'Position {int(v)}\nclick n={int(p):,} · non-click n={int(n-p):,}' for v,p,n in zip(data.group,data.positives,data.rows)])
    ax.set(ylim=(0,123),ylabel='Error rate (%)'); ax.set_yticks([0,20,40,60,80,100])
    ax.legend(loc='upper left',fontsize=10,ncol=1)

NOTES={
 'fraud_precision_recall':'Precision–recall is more informative than accuracy for this rare-event benchmark. The marked threshold is fixed from validation F2.',
 'fraud_review_budget':'Budget percentages are validation targets, not guarantees of test capacity. The F2 threshold is a separate operating choice. Only the four lowest documented budgets are shown.',
 'fraud_score_distribution':'Each class has its own denominator; this does not show population prevalence. Score is on a log axis. Up to 1,800 ordered points per class show the ECDF; endpoints are retained.',
 'sentiment_confusion':'Counts use all 12,284 official test posts. Cell percentages are normalized within actual class, not over all posts.',
 'sentiment_classes':'Class support is shown under each label. F1 balances precision and recall; class imbalance makes aggregate accuracy incomplete.',
 'sentiment_models':'All candidates are shown for transparency. TF–IDF was selected on validation macro F1 even though the calibrated candidate is stronger on this test; the test is not used to change that choice.',
 'churn_importance':'Permutation importance was measured on three development validation folds. Dots are fold estimates, not confidence bounds. A predictive association does not establish a cause of departure.',
 'churn_departments':'Intervals are Wilson 95% intervals assuming independent rows; they do not cover company shift or undocumented feature timing. Small departure counts produce wide intervals. This is not an employee decision tool.',
 'churn_calibration':'Uniform-bin uncertainty is shown, so bins with few rows can look unstable. Overall log loss summarizes all rows. The data lacks a prospective prediction horizon.',
 'ctr_gains':'Fractions were set in advance; no test-based threshold selection. Rates and lift describe the downsampled benchmark, not natural campaign CTR.',
 'ctr_model_loss':'Probabilities describe the sampled benchmark. Small differences cannot restore unknown population prevalence, and no statistical superiority is inferred.',
 'ctr_position_errors':'Whiskers are Wilson 95% row intervals, not clustered campaign inference. These descriptive position associations do not establish a causal placement effect.'}

def header(fig,config,label):
    fig.text(.055,.964,config['title'].upper()+' / '+label,color=config['accent'],fontsize=11,fontweight='bold')
    fig.text(.055,.921,config['subtitle'],fontsize=23,fontweight='bold')
    fig.text(.055,.024,config['note'],fontsize=9,color=MUTED)
    fig.add_artist(plt.Line2D([.055,.945],[.891,.891],transform=fig.transFigure,color=config['accent'],lw=3))

def save(fig,path):
    # Remove locator ticks outside the view, which are not drawn by Matplotlib.
    for ax in fig.axes:
        lo,hi=sorted(ax.get_xlim()); ax.set_xticks([t for t in ax.get_xticks() if lo<=t<=hi])
        lo,hi=sorted(ax.get_ylim()); ax.set_yticks([t for t in ax.get_yticks() if lo<=t<=hi])
    # Check every axis/value label remains within the figure; overlaps reviewed visually.
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); box=fig.bbox
    for text in fig.findobj(plt.Text):
        if not text.get_visible() or not text.get_text().strip():continue
        extent=text.get_window_extent(renderer)
        assert extent.x0>=box.x0-2 and extent.y0>=box.y0-2 and extent.x1<=box.x1+2 and extent.y1<=box.y1+2, text.get_text()
    fig.savefig(path.with_suffix('.png'),dpi=220)
    fig.savefig(path.with_suffix('.svg'),metadata={'Date':None,'Creator':'Verified project results / Matplotlib'})
    svg=path.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)

def generate(project,results,output):
    config=CONFIG[project]; r=Results(results); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    charts=[]
    for plot in config['plots']:
        fig,ax=plt.subplots(figsize=(12.5,7.8)); header(fig,config,'HELD-OUT RESULTS')
        fig.subplots_adjust(left=.23 if plot in ['fraud_review_budget','churn_importance','churn_departments','ctr_model_loss'] else .15,
                            right=.955,bottom=.18,top=.77)
        globals()[plot](ax,r); save(fig,output/plot)
        charts.append({'chart':plot,'interpretation':NOTES[plot],'png':plot+'.png','svg':plot+'.svg'})
    fig=plt.figure(figsize=(17,13.5)); header(fig,config,'PROJECT SUMMARY')
    grid=fig.add_gridspec(2,2,left=.16,right=.945,bottom=.13,top=.78,wspace=.48,hspace=.66)
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]
    for ax,plot in zip(axes,config['plots']):globals()[plot](ax,r)
    save(fig,output/'dashboard')
    manifest={'project':project,'source_directory':'results/phase2','source_sha256':r.sources,'model':r.selected,
              'model_refitted':False,'network_requests':0,'deleted_files':[],
              'checks':'Counts, selected metrics and key policy points checked against recorded metrics; figure labels checked for clipping.',
              'charts':charts,'dashboard':['dashboard.png','dashboard.svg']}
    (output/'chart-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# '+config['title']+' — colorful results gallery','',config['subtitle'],'',config['note'],'',
           '![Project summary](dashboard.png)','',
           'High-resolution PNGs and editable SVGs come from saved phase-two results. No model was retrained and no data was downloaded. These are descriptive benchmark results; model and policy choices were fixed on validation.','']
    for item in charts:
        lines += ['## '+item['chart'].replace('_',' ').title(),'',item['interpretation'],'',
                  f"![{item['chart'].replace('_',' ')}]({item['png']})",'',f"[PNG]({item['png']}) · [SVG]({item['svg']})",'']
    lines += ['## Reproduce','','```bash','python plot_gallery.py','```','',
              'Use the pinned project requirements. Sources and hashes are in [chart-provenance.json](chart-provenance.json). Original phase-two outputs are unchanged.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(project+': three charts + summary, PNG/SVG',flush=True)
    return r

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',choices=list(CONFIG),default=ROOT.name if ROOT.name in CONFIG else None)
    parser.add_argument('--results',type=Path,default=ROOT/'results/phase2')
    parser.add_argument('--output',type=Path,default=ROOT/'results/charts')
    args=parser.parse_args()
    if args.project is None:parser.error('--project is required outside a project repository')
    generate(args.project,args.results,args.output)

if __name__=='__main__':main()
