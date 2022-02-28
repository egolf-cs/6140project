def apply_classifier(clfr, x_train, y_train, x_test, y_test):
    clfr.fit(x_train, y_train)
    predictions = clfr.predict(x_test)
    score = clfr.score(x_test, y_test)
    return predictions, score
