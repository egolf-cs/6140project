from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score as bal_score
from collections import OrderedDict

from dat import comb_ps as train_ps
from abstract import save_clfr, tune_clfr


def dict2odict(d,key_fn=lambda x: x):
    ts = []
    ks = list(d.keys())
    ks.sort(key = key_fn)
    for k in ks:
        ts += [(k,d[k])]
    return OrderedDict(ts)

def update_params(o_params, tuned_params):
    for k in o_params:
        if k not in tuned_params:
            tuned_params[k] = o_params[k]

def mk_dummies():
    meta = {"train_xs":train_ps.xs, "train_ys":train_ps.ys}
    # Most Frequent
    frq_clfr = DummyClassifier(strategy="most_frequent")
    frq_clfr.fit(train_ps.xs, train_ps.ys)
    save_clfr(frq_clfr,meta,"frq_clfr")
    # Random
    rnd_clfr = DummyClassifier(strategy="uniform")
    rnd_clfr.fit(train_ps.xs, train_ps.ys)
    save_clfr(rnd_clfr,meta,"rnd_clfr")

def mk_LR():
    # w/o tuning
    lr_params = {"solver":"saga",
                    "penalty":"none",
                    "tol":1e-4,
                    "C":1.0,
                    "multi_class":"multinomial",
                    "max_iter":5000,
                    "l1_ratio":0.5}
    lr_meta = {"train_xs":train_ps.xs, "train_ys":train_ps.ys, "params":lr_params}
    lr_clfr = LogisticRegression(**lr_params)
    lr_clfr.fit(train_ps.xs,train_ps.ys)
    save_clfr(lr_clfr,lr_meta,"lr_clfr")
    # w/  tuning
    Cs = [0.5*i for i in range(1,10)]
    pens = ["elasticnet"]
    rats = [0.1*i for i in range(0,11)]
    lr_grid = {"C":Cs, "penalty":pens, "l1_ratio":rats}
    u_lr_clfr = LogisticRegression(**lr_params)
    # tu_lr_clfr, tu_lr_params, lr_val_score = tune_clfr(u_lr_clfr, lr_grid, 10, 3, train_ps.xs, train_ps.ys, 10)
    # update_params(lr_params,tu_lr_params)
    # tu_lr_meta = {"train_xs":train_ps.xs,
    #                 "train_ys":train_ps.ys,
    #                 "params":tu_lr_params,
    #                 "grid":lr_grid,
    #                 "val_score":lr_val_score}
    # save_clfr(tu_lr_clfr,tu_lr_meta,"tu_lr_clfr")
    best_lr_params = {'penalty': 'elasticnet', 'l1_ratio': 0.3, 'C': 300.0, 'solver': 'saga', 'tol': 0.0001, 'multi_class': 'multinomial', 'max_iter': 5000}
    best_lr_clfr = LogisticRegression(**best_lr_params)
    best_lr_meta = {"train_xs":train_ps.xs, "train_ys":train_ps.ys, "params":best_lr_params}
    best_lr_clfr.fit(train_ps.xs,train_ps.ys)
    save_clfr(best_lr_clfr,best_lr_meta,"test_lr_clfr")
    pass

def mk_SVC():
    # linear w/, w/o tuning
    # poly   w/, w/o tuning
    # rbf    w/, w/o tuning
    pass

def mk_DNN():
    pass

# mk_dummies()
mk_LR()
