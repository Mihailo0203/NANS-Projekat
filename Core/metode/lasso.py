# Core/metode/lasso.py

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


def _soft_threshold(z, lam):
    z = float(z)
    lam = float(lam)
    if z > lam:
        return z - lam
    if z < -lam:
        return z + lam
    return 0.0


def fit(x_train, y_train, alfa=0.1, epochs=100, intercept=True):
    """
    alfa: jačina L1 regularizacije.
    epochs: broj prolaza kroz sve koeficijente.
    """
    X = _to_list_2d(x_train)
    y = _to_list_1d(y_train)

    n = len(X)
    p = len(X[0])

    w = [0.0] * p
    b = 0.0

    # suma x_ij^2 po koloni
    col_sq = [0.0] * p
    for j in range(p):
        s = 0.0
        for i in range(n):
            v = float(X[i][j])
            s += v * v
        if s == 0.0:
            s = 1.0
        col_sq[j] = s

    for ep in range(epochs):
        # update intercept 
        if intercept:
            # b = srednja vrednost 
            s = 0.0
            for i in range(n):
                pred = 0.0
                for j in range(p):
                    pred += w[j] * float(X[i][j])
                s += float(y[i]) - pred
            b = s / n

        # update svake koordinate w_j
        for j in range(p):
            # računamo parcijalni rezidual bez w_j
            num = 0.0
            for i in range(n):
                pred_bez_j = b
                for k in range(p):
                    if k == j:
                        continue
                    pred_bez_j += w[k] * float(X[i][k])

                r = float(y[i]) - pred_bez_j
                num += float(X[i][j]) * r


            w[j] = _soft_threshold(num, alfa) / col_sq[j]

    return {"w": w, "b": b, "intercept": intercept, "alfa": alfa}


def predict(model, x):
    X = _to_list_2d(x)
    w = model["w"]
    b = model["b"]
    intercept = model["intercept"]

    y_pred = []
    for i in range(len(X)):
        s = b if intercept else 0.0
        for j in range(len(w)):
            s += w[j] * float(X[i][j])
        y_pred.append(s)
    return y_pred


def get_coef(model):
    return model["b"], model["w"]