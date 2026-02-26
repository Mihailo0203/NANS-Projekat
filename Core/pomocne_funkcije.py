# Core/pomocne_funkcije.py

import math

def u_listu_2d(X):
    if hasattr(X, "values"):
        X = X.values
    if hasattr(X, "tolist"):
        return X.tolist()
    return X


def u_listu_1d(y):
    # Pretvori y u listu
    if hasattr(y, "values"):
        y = y.values
    if hasattr(y, "tolist"):
        return y.tolist()
    return y


def dimenzije(A):
    r = len(A)
    c = len(A[0]) if r > 0 else 0
    return r, c


def transponuj(A):
    # Transponovanje matrice A
    A = u_listu_2d(A)
    r, c = dimenzije(A)
    AT = []
    for j in range(c):
        red = []
        for i in range(r):
            red.append(float(A[i][j]))
        AT.append(red)
    return AT


def pomnozi_matrice(A, B):
    # Množenje matrica
    A = u_listu_2d(A)
    B = u_listu_2d(B)
    rA, cA = dimenzije(A)
    rB, cB = dimenzije(B)
    if cA != rB:
        raise ValueError("Neispravne dimenzije za množenje matrica.")

    C = []
    for i in range(rA):
        red = []
        for j in range(cB):
            s = 0.0
            for k in range(cA):
                s += float(A[i][k]) * float(B[k][j])
            red.append(s)
        C.append(red)
    return C


def pomnozi_matricu_vektor(A, v):
    A = u_listu_2d(A)
    v = u_listu_1d(v)
    r, c = dimenzije(A)
    if len(v) != c:
        raise ValueError("Neispravne dimenzije za A*v")

    rez = []
    for i in range(r):
        s = 0.0
        for j in range(c):
            s += float(A[i][j]) * float(v[j])
        rez.append(s)
    return rez


def identitet(n):
    I = []
    for i in range(n):
        red = []
        for j in range(n):
            red.append(1.0 if i == j else 0.0)
        I.append(red)
    return I


def dodaj_bias_kolonu(X):
    # Dodaje kolonu jedinica na početak (za intercept)
    X = u_listu_2d(X)
    r, c = dimenzije(X)
    X2 = []
    for i in range(r):
        red = [1.0]
        for j in range(c):
            red.append(float(X[i][j]))
        X2.append(red)
    return X2


def resi_sistem_gauss_jordan(A, b):
    # Rešava sistem A x = b Gauss-Jordan metodom.
    A = u_listu_2d(A)
    b = u_listu_1d(b)

    n, m = dimenzije(A)
    if n != m:
        raise ValueError("A mora biti kvadratna matrica.")
    if len(b) != n:
        raise ValueError("Dimenzija b mora biti n.")

    M = []
    for i in range(n):
        red = []
        for j in range(n):
            red.append(float(A[i][j]))
        red.append(float(b[i]))
        M.append(red)

    # eliminacija
    for col in range(n):
        pivot_red = col
        pivot_vred = abs(M[col][col])
        for r in range(col + 1, n):
            if abs(M[r][col]) > pivot_vred:
                pivot_vred = abs(M[r][col])
                pivot_red = r

        if pivot_vred == 0.0:
            raise ValueError("Sistem nema jedinstveno rešenje (pivot 0).")

        if pivot_red != col:
            tmp = M[col]
            M[col] = M[pivot_red]
            M[pivot_red] = tmp

        pivot = M[col][col]
        for j in range(col, n + 1):
            M[col][j] = M[col][j] / pivot

        for r in range(n):
            if r == col:
                continue
            faktor = M[r][col]
            for j in range(col, n + 1):
                M[r][j] = M[r][j] - faktor * M[col][j]

    x = []
    for i in range(n):
        x.append(M[i][n])
    return x

# OLS I RIDGE POMOĆNE FUNKCIJE

def ols_koeficijenti(X, y, intercept=True):
    X = u_listu_2d(X)
    y = u_listu_1d(y)

    if intercept:
        X = dodaj_bias_kolonu(X)

    XT = transponuj(X)
    XTX = pomnozi_matrice(XT, X)


    y_col = []
    for v in y:
        y_col.append([float(v)])
    XTy_col = pomnozi_matrice(XT, y_col)


    XTy = []
    for i in range(len(XTy_col)):
        XTy.append(XTy_col[i][0])

    beta = resi_sistem_gauss_jordan(XTX, XTy)
    return beta


def ridge_koeficijenti(X, y, alfa, intercept=True):

    X = u_listu_2d(X)
    y = u_listu_1d(y)

    if intercept:
        X = dodaj_bias_kolonu(X)

    XT = transponuj(X)
    XTX = pomnozi_matrice(XT, X)

    n, _ = dimenzije(XTX)
    I = identitet(n)

    for i in range(n):
        if intercept and i == 0:
            continue
        XTX[i][i] = XTX[i][i] + float(alfa) * I[i][i]


    y_col = [[float(v)] for v in y]
    XTy_col = pomnozi_matrice(XT, y_col)
    XTy = [XTy_col[i][0] for i in range(len(XTy_col))]

    beta = resi_sistem_gauss_jordan(XTX, XTy)
    return beta

# LASSO / ELASTIC-NET POMOĆNE FUNKCIJE

def soft_threshold(z, lam):
    z = float(z)
    lam = float(lam)

    if z > lam:
        return z - lam
    if z < -lam:
        return z + lam
    return 0.0

# HUBER POMOĆNE FUNKCIJE

def huber_težine(reziduali, delta):
    w = []
    for r in reziduali:
        a = abs(float(r))
        if a <= float(delta) or a == 0.0:
            w.append(1.0)
        else:
            w.append(float(delta) / a)
    return w


def napravi_ponderisane(X, y, w):
    X = u_listu_2d(X)
    y = u_listu_1d(y)

    Xw = []
    yw = []
    for i in range(len(X)):
        s = math.sqrt(float(w[i]))
        red = []
        for j in range(len(X[i])):
            red.append(float(X[i][j]) * s)
        Xw.append(red)
        yw.append(float(y[i]) * s)
    return Xw, yw

# PREDIKCIJA

def predikcija_sa_beta(X, beta, intercept=True):
    X = u_listu_2d(X)
    beta = u_listu_1d(beta)

    if intercept:
        X = dodaj_bias_kolonu(X)

    y_hat = pomnozi_matricu_vektor(X, beta)
    return y_hat