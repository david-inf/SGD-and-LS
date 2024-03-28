# first line: 152
@mem.cache
def load_mushrooms():  # ok
    path = "datasets/LIBSVM/mushrooms.txt"
    X, y = load_svmlight_file(path)

    # add constant column
    X_prep = add_intercept(X)

    # encode respponse variable in {-1,1}
    encoder = LabelEncoder()
    y_prep = 2 * encoder.fit_transform(y) - 1

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y_prep, test_size=0.2, random_state=42)

    train_sklearn_log(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test
