from read import DatPts
TRAIN = "dat/train.csv"
TEST = "dat/test.csv"
VALIDATION = "dat/validation.csv"

train_ps = DatPts(TRAIN)
test_ps = DatPts(TEST)
val_ps = DatPts(VALIDATION)
assert(train_ps.validate())
assert(test_ps.validate())
assert(val_ps.validate())

# This combines the validation and test set in the case that we want to do
# cross validation instead of validation
comb_ps = DatPts(TRAIN)
comb_ps.concat(DatPts(VALIDATION))
assert(comb_ps.validate())


# print(train_ps)
# print(test_ps)
# print(val_ps)
# print(comb_ps)
