from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from read import DatPts
from numpy import var, mean
TRAIN = "dat/train.csv"
TEST = "dat/test.csv"
VALIDATION = "dat/validation.csv"
RAW = "dat/raw.csv"

# TODO: Compile all data and randomly divide all data into train and test sets.

# train_ps = DatPts(TRAIN)
# test_ps = DatPts(TEST)
# val_ps = DatPts(VALIDATION)
# assert(train_ps.validate())
# assert(test_ps.validate())
# assert(val_ps.validate())
#
# # This combines the validation and test set in the case that we want to do
# # cross validation instead of validation
# comb_ps = DatPts(TRAIN)
# comb_ps.concat(DatPts(VALIDATION))
# assert(comb_ps.validate())

ps = DatPts(RAW)
ps.validate()
X = list(range(len(ps.xs)))
y = [ps.ys[i] for i in X]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

train_ps = DatPts()
train_ps.ps = [ps.ps[i] for i in X_train]
train_ps.xs = [p.x for p in train_ps.ps]
train_ps.ys = [p.y for p in train_ps.ps]
train_ps.rawys = [p.rawy for p in train_ps.ps]
# print(f"Train mean before norm: {mean(train_ps.xs)}")
# print(f"Train var before norm: {var(train_ps.xs)}")

test_ps = DatPts()
test_ps.ps = [ps.ps[i] for i in X_test]
test_ps.xs = [p.x for p in test_ps.ps]
test_ps.ys = [p.y for p in test_ps.ps]
test_ps.rawys = [p.rawy for p in test_ps.ps]
# print(f"Test mean before norm: {mean(test_ps.xs)}")
# print(f"Test var before norm: {var(test_ps.xs)}")

scaler = StandardScaler()
scaler = train_ps.normalize(True, scaler)
# print(f"Train mean after norm: {mean(train_ps.xs)}")
# print(f"Train var after norm: {var(train_ps.xs)}")
test_ps.normalize(False, scaler)
# print(f"Train mean after norm: {mean(test_ps.xs)}")
# print(f"Train var after norm: {var(test_ps.xs)}")
