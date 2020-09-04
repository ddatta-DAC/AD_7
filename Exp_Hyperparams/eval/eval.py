
import pandas as pd
import numpy as np
from sklearn.metrics import auc

def eval(anom_scores, test_scores, order='ascending'):
    anom_labels = [1 for _ in range(np.array(anom_scores).shape[0])]
    test_labels = [1 for _ in range(np.array(test_scores).shape[0])]
    combined_scores = np.concatenate([anom_scores, test_scores], axis=0)
    combined_labels = np.concatenate([anom_labels, test_labels], axis=0)
    res_data = []
    for i, j in zip(combined_scores, combined_labels):
        res_data.append((i, j))
    res_df = pd.DataFrame(res_data, columns=['score', 'label'])

    #  Normalize values
    def _normalize_(val, _min, _max):
        return (val - _min) / (_max - _min)

    _max = max(combined_scores)
    _min = min(combined_scores)

    res_df['score'] = res_df['score'].parallel_apply(
        _normalize_,
        args=(_min, _max,)
    )
    _max = max(res_df['score'])
    _min = min(res_df['score'])
    step = (_max - _min) / 100

    P = []
    R = [0]
    num_anomalies = np.array(anom_scores).shape[0]
    if order == 'ascending':
        thresh = _min + step
        while thresh <= _max:
            sel = res_df.loc[res_df['score'] <= thresh]
            if len(sel) == 0:
                thresh += step
                continue
            correct = sel.loc[sel['label'] == 1]
            prec = len(correct) / len(sel)
            rec = len(correct) / num_anomalies
            P.append(prec)
            R.append(rec)
            if rec >= 1.0:
                break
            thresh += step
            thresh = round(thresh, 3)
        P = [P[0]] + P

    else:
        thresh = _max - step
        while thresh >= _min:
            sel = res_df.loc[res_df['score'] >= thresh]
            if len(sel) == 0:
                thresh -= step
                continue
            correct = sel.loc[sel['label'] == 1]
            prec = len(correct) / len(sel)
            rec = len(correct) / num_anomalies
            P.append(prec)
            R.append(rec)
            if rec >= 1.0:
                break
            thresh -= step
            thresh = round(thresh, 3)
        P = [P[0]] + P

    pr_auc = auc(R, P)
    return pr_auc
