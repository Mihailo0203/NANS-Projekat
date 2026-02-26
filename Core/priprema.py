# Core/priprema.py

import pandas as pd
import numpy as np
import math

# RUČNI TRAIN/VAL SPLIT

def train_test_split_manual(x, y, train_size=0.8, shuffle=True, random_state=42):
    n = len(x)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(int(random_state))
        rng.shuffle(idx)

    n_train = int(round(float(train_size) * n))
    idx_train = idx[:n_train]
    idx_val = idx[n_train:]

    x_train = x.iloc[idx_train].copy()
    x_val = x.iloc[idx_val].copy()
    y_train = y.iloc[idx_train].copy()
    y_val = y.iloc[idx_val].copy()
    return x_train, x_val, y_train, y_val


# 1) UČITAVANJE

def ucitaj_df(putanja_csv):
    df = pd.read_csv(putanja_csv, sep=",")
    return df


# 2) INTERPOLACIJA / DROP NA
def sredi_nedostajuce_vrednosti(df, nacin="dropna"):
    df = df.copy()

    if nacin == "dropna":
        df = df.dropna()
        return df

    if nacin == "interpolate":
        # interpoliramo sve numeričke kolone
        numericke = df.select_dtypes(include=["number"]).columns
        for kol in numericke:
            if df[kol].isna().any():
                df[kol] = df[kol].interpolate(method="linear", limit_direction="both")

        df = df.dropna()
        return df

    raise ValueError("Nepoznat nacin. Koristi 'dropna' ili 'interpolate'.")

# 3) IZBOR KOLONA 

def izaberi_kolone(df, cilj, feature_kolone=None):
    df = df.copy()

    if feature_kolone is None:
        return df

    kolone = list(feature_kolone) + [cilj]
    df = df[kolone]
    return df

# 4) DUMMY KODIRANJE

def dummy_kodiranje(df, drop_first=True):
    df = df.copy()
    df = pd.get_dummies(df, drop_first=drop_first)
    return df


# 5) STANDARDIZACIJA
def izracunaj_mean_std(x_train):
    srednje = {}
    std = {}

    for kol in x_train.columns:
        n = len(x_train)

        # mean
        s = 0.0
        for i in range(n):
            s += float(x_train.iloc[i][kol])
        mean = s / n

        # std
        ss = 0.0
        for i in range(n):
            d = float(x_train.iloc[i][kol]) - mean
            ss += d * d
        st = math.sqrt(ss / n)

        if st == 0.0:
            st = 1.0

        srednje[kol] = mean
        std[kol] = st

    return srednje, std


def standardizuj(x, srednje, std):

    x2 = x.copy().astype(float)

    for kol in x2.columns:
        mean = srednje[kol]
        st = std[kol]
        for i in range(len(x2)):
            x2.iloc[i, x2.columns.get_loc(kol)] = (float(x2.iloc[i][kol]) - mean) / st

    return x2

# 6) PORAVNANJE KOLONA
def poravnaj_kolone(x_train, x_drugi):
    x_drugi = x_drugi.copy()

    # dodaj kolone koje fale
    for kol in x_train.columns:
        if kol not in x_drugi.columns:
            x_drugi[kol] = 0

    # izbaci višak
    visak = []
    for kol in x_drugi.columns:
        if kol not in x_train.columns:
            visak.append(kol)
    if len(visak) > 0:
        x_drugi = x_drugi.drop(columns=visak)

    # isti redosled
    x_drugi = x_drugi[x_train.columns]
    return x_drugi


# 7) GLAVNA FUNKCIJA

def pripremi_podatke(
    putanja_train_csv,
    cilj,
    feature_kolone=None,
    nacin_nedostajuce="dropna",
    dummy_flag=False,
    drop_first_dummy=True,
    train_size=0.8,
    shuffle=True,
    random_state=42,
    standardizacija_flag=True
):
    # 1) učitaj
    df = ucitaj_df(putanja_train_csv)
    # 2) izbor kolona
    df = izaberi_kolone(df, cilj, feature_kolone=feature_kolone)
    # 3) NaN 
    df = sredi_nedostajuce_vrednosti(df, nacin=nacin_nedostajuce)
    # 4) dummy
    if dummy_flag:
        df = dummy_kodiranje(df, drop_first=drop_first_dummy)
    # 5) x / y
    x = df.drop(columns=[cilj])
    y = df[cilj]
    # 6) split 80/20
    x_train, x_val, y_train, y_val = train_test_split_manual(
        x, y,
        train_size=train_size,
        shuffle=shuffle,
        random_state=random_state
    )

    x_train = x_train.astype(float)
    x_val = x_val.astype(float)
    # 7) standardizacija
    if standardizacija_flag:
        srednje, std = izracunaj_mean_std(x_train)
        x_train = standardizuj(x_train, srednje, std)
        x_val = standardizuj(x_val, srednje, std)
        return df, x, y, x_train, x_val, y_train, y_val, srednje, std

    return df, x, y, x_train, x_val, y_train, y_val

# 8) PRIPREMA TESTA (ako imaš test.csv)
def pripremi_test(
    putanja_test_csv,
    cilj,
    feature_kolone=None,
    nacin_nedostajuce="dropna",
    dummy_flag=False,
    drop_first_dummy=True,
    x_train_kolone=None,
    srednje=None,
    std=None
):
    df_test = ucitaj_df(putanja_test_csv)
    df_test = sredi_nedostajuce_vrednosti(df_test, nacin=nacin_nedostajuce)
    df_test = izaberi_kolone(df_test, cilj, feature_kolone=feature_kolone)

    if dummy_flag:
        df_test = dummy_kodiranje(df_test, drop_first=drop_first_dummy)

    x_test = df_test.drop(columns=[cilj])
    y_test = df_test[cilj]

    if x_train_kolone is not None:
        x_train_dummy = pd.DataFrame(columns=x_train_kolone)
        x_test = poravnaj_kolone(x_train_dummy, x_test)

    if (srednje is not None) and (std is not None):
        x_test = standardizuj(x_test, srednje, std)

    return df_test, x_test, y_test