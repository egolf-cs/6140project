from dat import test_ps
from abstract import load_clfr, save_clfr, apply_classifier
from sklearn.metrics import balanced_accuracy_score as bal_score
from collections import OrderedDict
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

fnames = ["frq_clfr", "rnd_clfr", "lr_clfr", "tu_lr_clfr", "best_lr_clfr", "test_lr_clfr"]
[score_clfr(fname) for fname in fnames]
rprt_all(fnames)
