# https://pyimagesearch.com/2021/05/17/introduction-to-hyperparameter-tuning-with-scikit-learn-and-python/
import time
# from dat import test_ps,train_ps,val_ps
# clfr, x_train, y_train, x_test, y_test --> preds, score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score as bal_score

from joblib import dump, load
clfr_dir = "models"

def save_clfr(clfr, meta, fname):
    print(f"SAVING: {fname}")
    meta["score_fn"] = bal_score
    dump((clfr,meta),f'{clfr_dir}/{fname}.joblib')
    print(f"SAVED: {fname}")

def load_clfr(fname):
    tmp = load(f'{clfr_dir}/{fname}.joblib')
    return tmp[0], tmp[1]

def tune_clfr(clfr, grid, splits, repeats, xs, ys, verbose=0):
    cvFold = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=1)
    randomSearch = RandomizedSearchCV(estimator=clfr, param_distributions=grid, n_jobs=-1,
    	cv=cvFold, verbose=verbose, scoring='balanced_accuracy')
    searchResults = randomSearch.fit(xs, ys)
    clfr = searchResults.best_estimator_
    params = searchResults.best_params_
    score = searchResults.best_score_
    return clfr, params, score

def rel_abundance(ys):
    ls = set(ys)
    return dict([(l,ys.count(l)/len(ys)) for l in ls])

def inverse_rel_abundance(ys):
    ls = set(ys)
    return dict([(l,1/(ys.count(l)/len(ys))) for l in ls])

def apply_classifier(clfr, x_train, y_train, x_test, y_test, score = None):
    preds = clfr.predict(x_test)
    train_preds = clfr.predict(x_train)
    if score == None:
        test_score = clfr.score(x_test, y_test)
        train_score = clfr.score(x_train, y_train)
    else:
        test_score = score(y_test, preds)
        train_score = score(y_train, train_preds)

    return preds, train_score, test_score

if __name__ == '__main__':
    naive_weights = {0:1, 1:1, 2:1, 3:1, 4:1, 5:0.2}

    clfr = SVC()
    clfr.fit(train_ps.xs, train_ps.ys)
    preds, utrain_score, utest_score = apply_classifier(clfr, train_ps.xs, train_ps.ys, test_ps.xs, test_ps.ys, score=bal_score)

    kernel = ["rbf"]
    fname = f"tuned-{kernel}-{time.time()}"
    # weight_strategies = [None, 'balanced', inverse_rel_abundance(train_ps.ys), naive_weights]
    weight_strategies = [None]
    C = [x*0.5 for x in range(1,20)]
    grid = dict(kernel=kernel, C=C, class_weight=weight_strategies)
    # tclfr, params, val_score = tune_clfr(clfr, grid, 10, 3, train_ps.xs, train_ps.ys, 10)
    meta = {'params':params,'val_score':val_score}
    save_clfr(tclfr, meta, fname)
    lclfr, meta = load_clfr(fname)
    params = meta['params']
    val_score = meta['val_score']
    print(params)
    print(val_score)

    preds, train_score, test_score = apply_classifier(lclfr, train_ps.xs, train_ps.ys, test_ps.xs, test_ps.ys, score=bal_score)
    print(train_score, test_score)
    print(utrain_score, utest_score)
    print(rel_abundance(list(preds)))
