from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score as bal_score
from collections import OrderedDict

from dat import train_ps
from abstract import save_clfr, tune_clfr

REPEATS = 1
FOLDS   = 2

def dict2odict(d,key_fn=lambda x: x):
    ts = []
    ks = list(d.keys())
    ks.sort(key = key_fn)
    for k in ks:
        ts += [(k,d[k])]
    return OrderedDict(ts)

def update_params(o_params, tuned_params):
    print("UPDATING: params")
    for k in o_params:
        if k not in tuned_params:
            tuned_params[k] = o_params[k]
    print("UPDATED: params")

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
    lr_repeats, lr_folds = REPEATS, FOLDS
    ## w/o regularization or tuning
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
    # w/ regularization, w/o tuning
    reg_lr_params = dict(lr_params)
    reg_lr_params["penalty"] = "elasticnet"
    reg_lr_meta = {"train_xs":train_ps.xs, "train_ys":train_ps.ys, "params":reg_lr_params}
    reg_lr_clfr = LogisticRegression(**reg_lr_params)
    reg_lr_clfr.fit(train_ps.xs,train_ps.ys)
    save_clfr(reg_lr_clfr,reg_lr_meta,"reg_lr_clfr")
    ## w/  tuning
    Cs = [0.5*i for i in range(1,10)]
    pens = ["elasticnet"]
    rats = [0.1*i for i in range(0,11)]
    lr_grid = {"C":Cs, "penalty":pens, "l1_ratio":rats}
    u_lr_clfr = LogisticRegression(**reg_lr_params)
    tu_lr_clfr, tu_lr_params, lr_val_score = tune_clfr(u_lr_clfr, lr_grid, lr_folds, lr_repeats, train_ps.xs, train_ps.ys, 10)
    update_params(lr_params,tu_lr_params)
    tu_lr_meta = {"train_xs":train_ps.xs,
                    "train_ys":train_ps.ys,
                    "params":tu_lr_params,
                    "grid":lr_grid,
                    "val_score":lr_val_score}
    save_clfr(tu_lr_clfr,tu_lr_meta,"tu_lr_clfr")
    # best from tuning
    best_lr_params = {'penalty': 'elasticnet', 'l1_ratio': 0.3, 'C': 3.0, 'solver': 'saga', 'tol': 0.0001, 'multi_class': 'multinomial', 'max_iter': 5000}
    best_lr_clfr = LogisticRegression(**best_lr_params)
    best_lr_meta = {"train_xs":train_ps.xs, "train_ys":train_ps.ys, "params":best_lr_params}
    # best_lr_clfr.fit(train_ps.xs,train_ps.ys)
    # save_clfr(best_lr_clfr,best_lr_meta,"best_lr_clfr")
    pass

def mk_SVC():
    svc_repeats, svc_folds = REPEATS, FOLDS
    print("STARTING: linear w/o tuning")
    ### linear w/o tuning,
    lin_svc_params = {"tol":1e-3,
                        "C":1.0,
                        "kernel":"linear"}
    lin_svc_meta = {"train_xs":train_ps.xs, "train_ys":train_ps.ys, "params":lin_svc_params}
    lin_svc_clfr = SVC(**lin_svc_params)
    lin_svc_clfr.fit(train_ps.xs,train_ps.ys)
    save_clfr(lin_svc_clfr,lin_svc_meta,"lin_svc_clfr")
    print("STARTING: linear w/ tuning")
    ### linear w/ tuning
    Cs = [0.5*i for i in range(1,8)]
    lin_svc_grid = {"C":Cs}
    u_lin_svc_clfr = SVC(**lin_svc_params)
    tu_lin_svc_clfr, tu_lin_svc_params, lin_svc_val_score = tune_clfr(u_lin_svc_clfr, lin_svc_grid, svc_folds, svc_repeats, train_ps.xs, train_ps.ys, 50)
    update_params(lin_svc_params,tu_lin_svc_params)
    tu_lin_svc_meta = {"train_xs":train_ps.xs,
                        "train_ys":train_ps.ys,
                        "params":tu_lin_svc_params,
                        "grid":lin_svc_grid,
                        "val_score":lin_svc_val_score}
    save_clfr(tu_lin_svc_clfr,tu_lin_svc_meta,"tu_lin_svc_clfr")

    print("STARTING: poly w/o tuning")
    ## poly  w/o tuning
    poly_svc_params = {"tol":1e-3,
                        "C":1.0,
                        "kernel":"poly",
                        "degree":3,
                        "gamma":"scale",
                        "coef0":0.0}
    poly_svc_meta = {"train_xs":train_ps.xs, "train_ys":train_ps.ys, "params":poly_svc_params}
    poly_svc_clfr = SVC(**poly_svc_params)
    poly_svc_clfr.fit(train_ps.xs,train_ps.ys)
    save_clfr(poly_svc_clfr,poly_svc_meta,"poly_svc_clfr")
    print("STARTING: poly w/ tuning")
    ### poly w/ tuning
    Cs = [i*0.5 for i in range(1,8)]
    degs = [i for i in range(2,5)]
    coefs = [i for i in range(0,4)]
    print("COEFS", coefs)
    poly_svc_grid = {"C":Cs, "degree":degs, "coef0":coefs}
    u_poly_svc_clfr = SVC(**poly_svc_params)
    print("STARTING: tuning")
    tu_poly_svc_clfr, tu_poly_svc_params, poly_svc_val_score = tune_clfr(u_poly_svc_clfr, poly_svc_grid, svc_folds, svc_repeats, train_ps.xs, train_ps.ys, 50)
    print("DONE: tuning")
    update_params(poly_svc_params,tu_poly_svc_params)
    tu_poly_svc_meta = {"train_xs":train_ps.xs,
                        "train_ys":train_ps.ys,
                        "params":tu_poly_svc_params,
                        "grid":poly_svc_grid,
                        "val_score":poly_svc_val_score}
    save_clfr(tu_poly_svc_clfr,tu_poly_svc_meta,"tu_poly_svc_clfr")

    ### rbf    w/, w/o tuning
    print("STARTING: rbf w/o tuning")
    rbf_svc_params = {"tol":1e-3,
                        "C":1.0,
                        "kernel":"rbf",
                        "gamma":"scale"}
    rbf_svc_meta = {"train_xs":train_ps.xs, "train_ys":train_ps.ys, "params":rbf_svc_params}
    rbf_svc_clfr = SVC(**rbf_svc_params)
    rbf_svc_clfr.fit(train_ps.xs,train_ps.ys)
    save_clfr(rbf_svc_clfr,rbf_svc_meta,"rbf_svc_clfr")
    ### rbf w/ tuning
    print("STARTING: rbf w/ tuning")
    Cs = [0.5*i for i in range(1,8)]
    rbf_svc_grid = {"C":Cs}
    u_rbf_svc_clfr = SVC(**rbf_svc_params)
    tu_rbf_svc_clfr, tu_rbf_svc_params, rbf_svc_val_score = tune_clfr(u_rbf_svc_clfr, rbf_svc_grid, svc_folds, svc_repeats, train_ps.xs, train_ps.ys, 50)
    update_params(rbf_svc_params,tu_rbf_svc_params)
    tu_rbf_svc_meta = {"train_xs":train_ps.xs,
                        "train_ys":train_ps.ys,
                        "params":tu_rbf_svc_params,
                        "grid":rbf_svc_grid,
                        "val_score":rbf_svc_val_score}
    save_clfr(tu_rbf_svc_clfr,tu_rbf_svc_meta,"tu_rbf_svc_clfr")
    pass

def mk_DNN():
    pass

mk_dummies()
mk_LR()
# mk_SVC()
