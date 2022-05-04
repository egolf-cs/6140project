from dat import test_ps
from abstract import load_clfr, save_clfr, apply_classifier
from sklearn.metrics import balanced_accuracy_score as bal_score
from collections import OrderedDict
from mkclfrs import NNClfr
from mkclfrs import Oracle
import pandas as pd
pd.set_option('display.float_format', lambda x: f"{x:.3f}")

def score_clfr(fname):
    # print(fname)
    model, meta = load_clfr(fname)
    score_fn = meta["score_fn"] if "score_fn" in meta else None
    train_xs, train_ys = meta["train_xs"],meta["train_ys"]
    preds, train_score, test_score = apply_classifier(model, train_xs, train_ys, test_ps.xs, test_ps.ys, score_fn)
    meta["train_score"] = train_score
    meta["test_score"]  = test_score
    meta["test_preds"]  = preds
    save_clfr(model, meta, fname)

def thm_stats_clfr(fname):
    model, meta = load_clfr(fname)
    preds = meta["test_preds"]
    meta["thms_att"] = len(list(filter(lambda x: x != 5, preds)))
    # preds = test_ps.ys
    zs = list(zip(preds,test_ps.rawys))
    ts = [(z[1][z[0]] if z[0] != 5 else 0) for z in zs]
    ts = [100 if t == -100 else t for t in ts]
    def is_proved(z):
        pred = z[0]
        if pred == 5:
            return 0
        time = z[1][pred]
        if time == -100:
            return 0
        return 1
    ps = [is_proved(z) for z in zs]
    meta["thms_prv"] = sum(ps)
    meta["prv_time"] = sum(ts)
    save_clfr(model, meta, fname)


def rprt_clfr(fname):
    model, meta = load_clfr(fname)
    fmt = [("Model",fname),
            ("train score",meta["train_score"]),
            ("test score",meta["test_score"]),
            ("val score",meta["val_score"] if "val_score" in meta else None),
            ("theorems attempted",meta["thms_att"] if "thms_att" in meta else None),
            ("theorems proved",meta["thms_prv"]    if "thms_prv" in meta else None),
            ("total proof time",meta["prv_time"]   if "prv_time" in meta else None),
            ("params",meta["params"] if "params" in meta else None),
            ("grid",meta["grid"]   if "grid"   in meta else None)]
    fmt = OrderedDict(fmt)
    rep = [fmt[k] for k in fmt]
    return rep, fmt

def rprt_all(fnames):
    reps = [rprt_clfr(fname) for fname in fnames]
    fmt = reps[0][1]
    reps = [r[0] for r in reps]
    header = [str(k) for k in fmt]
    rows = [header] + reps
    left_headers = [r[0] for r in rows]
    data = [r[1:] for r in rows]
    df = pd.DataFrame(data, index=left_headers)
    print(df.to_string(header=False))

dummy_fnames = ["frq_clfr", "rnd_clfr", "const_clfr", "oracle"]
lr_fnames = ["lr_clfr", "reg_lr_clfr", "tu_lr_clfr", "best_lr_clfr"]
svc_fnames =["lin_svc_clfr","tu_lin_svc_clfr","poly_svc_clfr","tu_poly_svc_clfr2","tu_poly_svc_clfr3","tu_poly_svc_clfr4","rbf_svc_clfr","tu_rbf_svc_clfr"]
dnn_fnames = ["[51, 2500, 500, 50, 6]", "[51, 2500, 500, 500, 50, 6]"]

# thm_stats_clfr("frq_clfr")

fnames = dummy_fnames + lr_fnames + svc_fnames + dnn_fnames
[score_clfr(fname) for fname in fnames]
[thm_stats_clfr(fname) for fname in fnames]
rprt_all(fnames)
