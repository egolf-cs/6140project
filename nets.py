from dat import train_ps as ps
from dat import test_ps
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score as bal_score

from _pickle import dump, load

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from collections import OrderedDict
import copy

import matplotlib.pyplot as plt

FIG_DIR = "figs"
STAT_DIR = "stats"
MOD_DIR = "nns"

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 300
LR = 0.01

class ThmsDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
            return len(self.ys)

    def __getitem__(self, i):
            return self.xs[i], self.ys[i]

def write_stats(S, train_acc, val_acc, train_loss, test_loss, train_id, val_id):
    f = open(f"{STAT_DIR}/{str(S)}-{str(train_id)}-{str(val_id)}.stats", "wb")
    tmp = dump((train_acc, val_acc, train_loss, test_loss), f)
    f.close()

def read_stats(S, train_id, val_id):
    f = open(f"{STAT_DIR}/{str(S)}-{str(train_id)}-{str(val_id)}.stats", "rb")
    tmp = load(f)
    f.close()
    train_acc, val_acc, train_loss, test_loss = tmp[0], tmp[1], tmp[2], tmp[3]
    return train_acc, val_acc, train_loss, test_loss

def save_nn(clfr, meta, S, train_id, val_id):
    print(f"SAVING: {str(S)}")
    f = open(f"{MOD_DIR}/{str(S)}-{str(train_id)}-{str(val_id)}.joblib", "wb")
    dump((clfr,meta), f)
    f.close()
    print(f"SAVED: {str(S)}")

def load_nn(S, train_id, val_id):
    f = open(f"{MOD_DIR}/{str(S)}-{str(train_id)}-{str(val_id)}.joblib", "rb")
    tmp = load(f)
    f.close()
    return tmp[0], tmp[1]

def argmax(xs, f):
    curr_max = f(xs[0])
    arg_max = xs[0]
    for i, x in enumerate(xs):
        y = f(x)
        if y > curr_max:
            curr_max = y
            arg_max = x
    return arg_max

def init_model(S):
    aux = []

    for l in range(1,len(S)-1):
        layer = (f"layer_{l}", nn.Linear(S[l-1], S[l]))
        act   = (f"activation_{l}", nn.ReLU())
        aux += [layer,act]

    aux += [("output_layer", nn.Linear(S[-2], S[-1]))]
    return nn.Sequential(OrderedDict(aux))

def train(dl, clfr, loss_fn, opt):
    size = len(dl.dataset)
    clfr.train()
    for b, (xs, ys) in enumerate(dl):
        xs = xs.to(DEVICE)
        ys = ys.to(DEVICE)

        preds = clfr(xs.float())
        loss = loss_fn(preds, ys)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if b % (BATCH_SIZE + 1) == 0:
            print(f"loss: {loss.item()} Progress: {b*len(xs):>5d}/{size:>5d}")

def test(dl, clfr, loss_fn, name):
    size = len(dl.dataset)
    n_batches = len(dl)
    clfr.eval()
    test_loss, correct = 0, 0
    per_class_correct = {}
    per_class = {}
    all_preds = []
    all_ys    = []
    with torch.no_grad():
        for xs, ys in dl:
            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)

            preds = clfr(xs.float())
            test_loss += loss_fn(preds, ys).item()

            # figure out which predictions in which classes are correct
            for j in range(len(preds)):
                p = preds[j]
                y = ys[j]
                yi = y.int().item()
                if yi not in per_class:
                    per_class[yi] = 0
                per_class[yi] += 1
                pidx = argmax(list(range(len(p))), lambda x: p[x])
                all_preds += [pidx]
                all_ys    += [yi]
                if pidx == y:
                    if yi not in per_class_correct:
                        per_class_correct[yi] = 0
                    per_class_correct[yi] += 1
                    correct += 1

    test_loss /= n_batches
    correct /= size
    per_class_acc = {}
    for k in per_class:
        if k not in per_class_correct:
            per_class_correct[k] = 0
        per_class_acc[k] = per_class_correct[k]/per_class[k]
    accs = [per_class_acc[k] for k in per_class_acc]
    bal_acc = sum(accs)/len(accs)

    print(f"{name} Error")
    print(f"Bal Acc: {bal_acc}, Avg loss: {test_loss}")
    print("Per class accuracy:")
    for k in per_class_acc:
        print(f"{k}:{per_class_acc[k]}")
    print()
    return bal_acc, test_loss, per_class_acc

def train_loop(clfr, train_dl, val_dl, loss_fn):
    opt = torch.optim.SGD(clfr.parameters(), lr=LR)
    best_model   = copy.deepcopy(clfr)
    best_val_acc = 0
    best_epoch = 0
    train_losses = []
    train_accs   = []
    val_losses   = []
    val_accs     = []
    for t in range(EPS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dl, clfr, loss_fn, opt)
        train_acc, train_loss, _ = test(train_dl, clfr, loss_fn, "TRAIN")
        val_acc,   val_loss,   _ = test(val_dl,   clfr, loss_fn, "VAL")
        train_losses += [train_loss]
        train_accs   += [train_acc]
        val_losses   += [val_loss]
        val_accs     += [val_acc]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(clfr)
            best_epoch = t
    best = (best_model, best_val_acc, best_epoch)
    train_stats = (train_losses, train_accs)
    val_stats   = (val_losses,   val_accs)
    return best, train_stats, val_stats

def train_and_save(S, train_dl, val_dl, train_id, val_id, loss_fn):
    try:
        train_accs, val_accs, train_losses, val_losses = read_stats(S, train_id, val_id)
        best_model, meta = load_nn(S,train_id,val_id)
        best_epoch = meta["best_epoch"]
    except:
        clfr = init_model(S)
        best, train_stats, val_stats = train_loop(clfr, train_dl, val_dl, loss_fn)
        best_model, best_val_acc, best_epoch = best[0], best[1], best[2]
        train_losses, train_accs = train_stats[0], train_stats[1]
        val_losses,   val_accs   = val_stats[0],   val_stats[1]
        write_stats(S, train_accs, val_accs, train_losses, val_losses, train_id, val_id)
        save_nn(best_model,{"best_epoch":best_epoch},S,train_id,val_id)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Loss")
    ax1.set(xlabel="epoch", ylabel="avg loss")
    ax1.plot(train_losses, label="Training Data")
    ax1.plot(val_losses, label="Validation Data")
    ax1.legend(loc="upper right")
    ax2.set_title("Accuracy")
    ax2.set(xlabel="epoch", ylabel="accuracy")
    ax2.plot(train_accs, label="Train Data")
    ax2.plot(val_accs, label="Validation Data")
    ax2.legend(loc="upper right")
    fig.savefig(f"{FIG_DIR}/{str(S)}-{str(train_id)}-{str(val_id)}.png")

    return best_model, best_epoch

S = [51,2500,500,50,6]

def mk_nn(S, ps):
    X = list(range(len(ps.xs)))
    y = [ps.ys[i] for i in X]
    X_train, X_val, _, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    train_xs = [ps.xs[i] for i in X_train]
    train_ys = [ps.ys[i] for i in X_train]
    counts = [0 for _ in range(len(list(set(train_ys))))]
    for y in train_ys:
        counts[y] += 1
    weight = torch.tensor([max(counts)/c for c in counts])
    loss_fn = nn.CrossEntropyLoss(weight=weight)


    val_xs = [ps.xs[i] for i in X_val]
    val_ys = [ps.ys[i] for i in X_val]

    train_data = ThmsDataset(train_xs, train_ys)
    val_data   = ThmsDataset(val_xs,   val_ys)
    train_dl = DataLoader(train_data, batch_size=BATCH_SIZE)
    val_dl   = DataLoader(val_data,  batch_size=BATCH_SIZE)

    best_model, best_epoch = train_and_save(S, train_dl, val_dl, 1, 2, loss_fn)
    # print(S)
    # print(f"Best epoch:{best_epoch}")
    # print(S)
    # test(train_dl, best_model, loss_fn, "TRAIN")
    # print(S)
    best_val_acc, _, _ = test(val_dl, best_model, loss_fn, "VAL")
    return best_model, best_epoch, best_val_acc, weight
