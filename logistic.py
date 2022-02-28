from dat import test_ps
from dat import comb_ps as train_ps
from abstract import apply_classifier
# clfr, x_train, y_train, x_test, y_test --> preds, score

from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
clfr = LogisticRegression(max_iter=4000)
preds, score = apply_classifier(clfr, train_ps.xs, train_ps.ys, test_ps.xs, test_ps.ys)
