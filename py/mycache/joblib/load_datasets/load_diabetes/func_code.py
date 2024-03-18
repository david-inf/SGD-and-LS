# first line: 58
@mem.cache
def load_diabetes():  # ok
    path = "datasets/LIBSVM/diabetes_scale.txt"
    X, y = load_svmlight_file(path)

    # add constant column
    # ones = lil_matrix(np.ones((y.size, 1)))
    # X_prep = hstack([ones, X.tolil()], format="csr")
    X_prep = add_intercept(X)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y, test_size=0.2, random_state=42)

    train_sklearn_log(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test
