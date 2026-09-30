"""Independent calibration, uncertainty, and validation-only operating policies."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from workflow import measure, json_write


def partition_training(splits, y, *, groups=None, times=None, seed=42):
    train = np.asarray(splits['train'])
    if times is not None:
        cut = np.quantile(np.asarray(times)[train], 0.75)
        fit, calibration = train[np.asarray(times)[train] < cut], train[np.asarray(times)[train] >= cut]
    elif groups is not None:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            train, y.iloc[train], np.asarray(groups)[train]))
        fit, calibration = train[a], train[b]
    else:
        fit, calibration = train_test_split(train, test_size=0.25, stratify=y.iloc[train], random_state=seed)
    result = {'fit': fit, 'calibration': calibration, 'validation': np.asarray(splits['validation']),
              'test': np.asarray(splits['test'])}
    for label, rows in result.items():
        if y.iloc[rows].nunique() != 2:
            raise ValueError(f'{label} needs both classes')
    assert len(set(np.concatenate(list(result.values())))) == len(y)
    if groups is not None:
        sets = [set(np.asarray(groups)[rows]) for rows in result.values()]
        assert all(not sets[a] & sets[b] for a in range(4) for b in range(a))
    if times is not None:
        ranges = [(np.asarray(times)[v].min(), np.asarray(times)[v].max()) for v in result.values()]
        assert all(ranges[i][1] < ranges[i+1][0] for i in range(3))
    return result


def fit_sigmoid(model, X, y):
    # FrozenEstimator ensures calibration cannot refit the base on reserve labels.
    return CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid').fit(X, y)


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return {'estimate': None, 'low': None, 'high': None, 'n': 0}
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    spread = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    # Roundoff at all-positive/all-negative bins must not put the MLE outside its interval.
    return {'estimate': float(p), 'low': float(min(p,max(0, center-spread))),
            'high': float(max(p,min(1, center+spread))), 'n': int(total)}


def bootstrap_intervals(y, scores, threshold, *, groups=None, repeats=200, seed=42):
    y, scores = np.asarray(y), np.asarray(scores)
    rng, samples = np.random.default_rng(seed), []
    if groups is None:
        classes = [np.flatnonzero(y == label) for label in np.unique(y)]
        method = 'Class-stratified row bootstrap; conditional on observed class counts'
    else:
        codes, _ = pd.factorize(np.asarray(groups))
        clusters = [np.flatnonzero(codes == code) for code in np.unique(codes)]
        method = 'Cluster bootstrap of held-out users/anonymous contexts'
    for _ in range(repeats):
        if groups is None:
            rows = np.concatenate([rng.choice(part, len(part), replace=True) for part in classes])
        else:
            rows = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        actual, p = y[rows], scores[rows]
        if len(np.unique(actual)) != 2:
            continue
        predicted = p >= threshold
        samples.append({'average_precision': average_precision_score(actual, p),
                        'f1': f1_score(actual, predicted, zero_division=0),
                        'log_loss': log_loss(actual, np.clip(p, 1e-12, 1-1e-12), labels=[0, 1])})
    intervals = {name: {'low': float(np.quantile([row[name] for row in samples], 0.025)),
                        'high': float(np.quantile([row[name] for row in samples], 0.975))}
                 for name in samples[0]}
    predicted = scores >= threshold
    true_positives = int(((y == 1) & predicted).sum())
    return {'level': 0.95, 'method': method, 'replicates': len(samples), 'intervals': intervals,
            'recall_wilson': wilson(true_positives, int((y == 1).sum())),
            'precision_wilson': wilson(true_positives, int(predicted.sum())),
            'limits': 'Wilson intervals assume independent rows; cluster bootstrap is used when IDs exist. '
                      'Neither method establishes future-domain performance or accounts for two-day temporal dependence.'}


def reliability_table(y, scores, name, bins=10, scheme='quantile'):
    y, scores = np.asarray(y), np.asarray(scores)
    edges = (np.unique(np.quantile(scores, np.linspace(0, 1, bins+1))) if scheme=='quantile'
             else np.linspace(0,1,bins+1))
    assignment = np.searchsorted(edges[1:-1], scores, side='right')
    rows = []
    for group in np.unique(assignment):
        mask = assignment == group
        interval = wilson(int(y[mask].sum()), int(mask.sum()))
        rows.append({'model': name, 'binning_scheme':scheme, 'bin': int(group), 'rows': int(mask.sum()),
                     'mean_score': float(scores[mask].mean()), 'observed_rate': float(y[mask].mean()),
                     'rate_low': interval['low'], 'rate_high': interval['high']})
    return pd.DataFrame(rows)


def plot_reliability(tables, output, title):
    frame = pd.concat(tables, ignore_index=True)
    frame.to_csv(output / 'reliability.csv', index=False, float_format='%.8f')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout='constrained')
    for name, part in frame.groupby('model', sort=False):
        uniform=part.loc[part.binning_scheme=='uniform']
        quantile=part.loc[part.binning_scheme=='quantile']
        if len(uniform)==0:
            uniform=quantile
        errors=np.vstack([uniform.observed_rate-uniform.rate_low,uniform.rate_high-uniform.observed_rate])
        axes[0].errorbar(uniform.mean_score, uniform.observed_rate, yerr=errors,fmt='o-',label=name.replace('_',' '),markersize=4,capsize=2)
        axes[1].plot(quantile.mean_score, quantile.rows, 'o-', label=name.replace('_', ' '), markersize=4)
    axes[0].plot([0, 1], [0, 1], '--', color='#8997a1', label='ideal agreement')
    axes[0].set(xlabel='Mean predicted benchmark score', ylabel='Observed positive fraction', title='Test reliability: fixed probability bins')
    axes[1].set(xlabel='Mean predicted benchmark score', ylabel='Rows in quantile bin', title='Quantile-bin support')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(output / 'reliability.png', dpi=150)
    plt.close(fig)


def threshold_for_budget(scores, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError('Budget fraction must be between zero and one')
    values, counts = np.unique(np.asarray(scores), return_counts=True)
    values, counts = values[::-1], counts[::-1]
    affordable = np.cumsum(counts) <= int(np.floor(fraction * len(scores)))
    return float(values[np.flatnonzero(affordable)[-1]]) if affordable.any() else float(np.nextafter(values[0], np.inf))


def operating_policies(validation_y, validation_scores, test_y, test_scores, output):
    rows = []
    for budget in [0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.2]:
        threshold = threshold_for_budget(validation_scores, budget)
        for name, y, scores in [('validation', validation_y, validation_scores), ('test', test_y, test_scores)]:
            rows.append({'validation_budget_fraction': budget, 'split': name,
                         **measure(pd.Series(np.asarray(y)), np.asarray(scores), threshold)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'operating_policies.csv', index=False, float_format='%.8f')
    return frame


def ranking_curve(y, scores, output):
    y, scores = np.asarray(y), np.asarray(scores)
    ordering = np.argsort(-scores, kind='stable')
    rows = []
    for fraction in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        selected = y[ordering[:max(1, int(np.ceil(fraction*len(y))))]]
        rows.append({'selection_fraction': fraction, 'rows': len(selected), 'positives_captured': int(selected.sum()),
                     'positive_capture_fraction': float(selected.sum()/y.sum()),
                     'observed_rate': float(selected.mean()), 'lift': float(selected.mean()/y.mean())})
    pd.DataFrame(rows).to_csv(output / 'ranking_lift.csv', index=False, float_format='%.8f')
    fig, ax = plt.subplots(figsize=(6.6, 4.3), layout='constrained')
    ax.plot([row['selection_fraction'] for row in rows], [row['positive_capture_fraction'] for row in rows], 'o-', color='#166c8b', label='model ranking')
    ax.plot([0, 1], [0, 1], '--', color='#a45a37', label='random ranking expectation')
    ax.set(xlabel='Fraction of benchmark contexts selected', ylabel='Fraction of observed positives captured', title='Untuned test ranking at fixed budgets', xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.savefig(output / 'ranking.png', dpi=150)
    plt.close(fig)


def subgroup_errors(X, y, scores, threshold, columns, output):
    frame = X[columns].copy().reset_index(drop=True)
    frame['actual'], frame['score'] = np.asarray(y), np.asarray(scores)
    rows = []
    for feature in columns:
        if pd.api.types.is_numeric_dtype(frame[feature]) and frame[feature].nunique() > 10:
            values = pd.cut(frame[feature], 5, duplicates='drop').astype(str)
        else:
            values = frame[feature].fillna('missing').astype(str)
        for category, indices in values.groupby(values).groups.items():
            part = frame.loc[indices]
            prediction = part.score.to_numpy() >= threshold
            rows.append({'feature': feature, 'group': category, 'rows': len(part),
                         'positives': int(part.actual.sum()), 'false_positives': int(((part.actual == 0) & prediction).sum()),
                         'false_negatives': int(((part.actual == 1) & ~prediction).sum()),
                         'brier_score': float(np.mean((part.score-part.actual)**2))})
    pd.DataFrame(rows).to_csv(output / 'subgroup_errors.csv', index=False, float_format='%.8f')


def export_splits(y, splits, output):
    assignments = np.full(len(y), '', dtype=object)
    for name, indices in splits.items():
        assignments[indices] = name
    assert not (assignments == '').any()
    pd.DataFrame({'row_id': range(len(y)), 'split': assignments, 'label': np.asarray(y)}).to_csv(output / 'splits.csv', index=False)
