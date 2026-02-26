# Core/metode/huber.py

def _to_list_2d(X):
    if hasattr(X, "values"):
        return X.values.tolist()
    if hasattr(X, "tolist"):
        return X.tolist()
    return X


def _to_list_1d(y):
    if hasattr(y, "values"):
        return y.values.tolist()
    if hasattr(y, "tolist"):
        return y.tolist()
    return y


def _predict_one_row(row, w, b):
    s = b
    for j in range(len(w)):
        s += w[j] * float(row[j])
    return s


def fit(x_train, y_train, delta=1.0, lr=0.01, epochs=2000, intercept=True):
    """
    Huber regresija preko gradient descent.

    delta: prag za robustnost (manji delta = agresivnije ignoriše outlier-e)
    """
    X = _to_list_2d(x_train)
    y = _to_list_1d(y_train)

    n = len(X)
    p = len(X[0])

    w = [0.0] * p
    b = 0.0

    for ep in range(epochs):
        grad_w = [0.0] * p
        grad_b = 0.0

        for i in range(n):
            y_hat = _predict_one_row(X[i], w, b if intercept else 0.0)
            err = y_hat - float(y[i])

            if abs(err) <= float(delta):
                g = err
            else:
                g = float(delta) if err > 0 else -float(delta)

            for j in range(p):
                grad_w[j] += g * float(X[i][j])
            grad_b += g

        # srednji gradijent
        for j in range(p):
            grad_w[j] = (1.0 / n) * grad_w[j]
        grad_b = (1.0 / n) * grad_b

        # update
        for j in range(p):
            w[j] = w[j] - lr * grad_w[j]
        if intercept:
            b = b - lr * grad_b

    return {"w": w, "b": b, "intercept": intercept, "delta": delta}


def predict(model, x):
    X = _to_list_2d(x)
    w = model["w"]
    b = model["b"]
    intercept = model["intercept"]

    y_pred = []
    for i in range(len(X)):
        y_pred.append(_predict_one_row(X[i], w, b if intercept else 0.0))
    return y_pred


def get_coef(model):
    return model["b"], model["w"]