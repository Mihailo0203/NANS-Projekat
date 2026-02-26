# run_models.py
# Jedan fajl koji:
# 1) pozove pripremi_podatke
# 2) trenira svih 5 metoda 
# 3) izracuna RMSE i adjusted R^2 
# 4) stabilnost koeficijenata + eksperiment sa kontrolisanim šumom

from dataclasses import dataclass
import numpy as np
import pandas as pd

from Core.priprema import pripremi_podatke
from Core.metode import ols, ridge, lasso, elastic_net, huber
from Core import utils_nans1

@dataclass
class WrappedModel:
    """
    Adapter da bi naši ručni modeli mogli da se koriste sa utils_nans1.
    Naši modeli već imaju intercept, pa metrike računamo direktno nad X (bez dodavanja konstante).
    """
    method_module: object
    model_obj: dict

    def predict(self, features):
        # features može biti DataFrame ili numpy niz
        if hasattr(features, "values"):
            X = features.values
        else:
            X = np.array(features)
        return self.method_module.predict(self.model_obj, X)


def _evaluate_with_utils(model_wrapper, x_val, y_val):
    """
    Racuna RMSE i adjusted R^2 koristeci utils_nans1.
    """
    rmse = utils_nans1.get_rmse(model_wrapper, x_val, y_val)
    r2_adj = utils_nans1.get_rsquared_adj(model_wrapper, x_val, y_val)
    return rmse, r2_adj


#  Glavna funkcija: treniraj svih 5 modela

def train_all_models(
    putanja_train_csv,
    cilj,
    feature_kolone=None,
    nacin_nedostajuce="dropna",
    dummy_flag=False,
    standardizacija_flag=True,
    random_state=42,
    # kontrolisani šum (0.0 = nema šuma)
    noise_sigma=0.0,
    noise_seed=123,
    # hiperparametri
    ridge_alfa=0.1,
    lasso_alfa=0.1,
    en_alfa=0.1,
    en_l1_ratio=0.5,
    huber_delta=1.0,
    # iteracije / brzine
    gd_lr=0.01,
    gd_epochs=1500,
    cd_epochs=80,
):

    out = pripremi_podatke(
        putanja_train_csv=putanja_train_csv,
        cilj=cilj,
        feature_kolone=feature_kolone,
        nacin_nedostajuce=nacin_nedostajuce,
        dummy_flag=dummy_flag,
        train_size=0.8,
        shuffle=True,
        random_state=random_state,
        standardizacija_flag=standardizacija_flag,
    )

    # raspakuj 
    if standardizacija_flag:
        df, x, y, x_train, x_val, y_train, y_val, srednje, std = out
    else:
        df, x, y, x_train, x_val, y_train, y_val = out
        srednje, std = None, None
    if noise_sigma is not None and float(noise_sigma) > 0.0:
        rng = np.random.default_rng(int(noise_seed))
        y_arr = y_train.values if hasattr(y_train, "values") else np.array(y_train, dtype=float)
        sigma = float(np.std(y_arr)) * float(noise_sigma)
        y_noisy = y_arr + rng.normal(0.0, sigma, size=y_arr.shape[0])
        if hasattr(y_train, "index"):
            y_train = pd.Series(y_noisy, index=y_train.index)
        else:
            y_train = y_noisy

    models = {}
    rows = []

    # --- OLS ---
    m_ols = ols.fit(x_train, y_train, lr=gd_lr, epochs=gd_epochs, intercept=True)
    w_ols = WrappedModel(ols, m_ols)
    rmse, r2_adj = _evaluate_with_utils(w_ols, x_val, y_val)
    rows.append({"Metoda": "OLS", "RMSE": rmse, "R2_adj": r2_adj})
    models["OLS"] = m_ols

    # --- Ridge ---
    m_ridge = ridge.fit(x_train, y_train, alfa=ridge_alfa, lr=gd_lr, epochs=gd_epochs, intercept=True)
    w_ridge = WrappedModel(ridge, m_ridge)
    rmse, r2_adj = _evaluate_with_utils(w_ridge, x_val, y_val)
    rows.append({"Metoda": f"Ridge (alfa={ridge_alfa})", "RMSE": rmse, "R2_adj": r2_adj})
    models["Ridge"] = m_ridge

    # --- Lasso ---
    m_lasso = lasso.fit(x_train, y_train, alfa=lasso_alfa, epochs=cd_epochs, intercept=True)
    w_lasso = WrappedModel(lasso, m_lasso)
    rmse, r2_adj = _evaluate_with_utils(w_lasso, x_val, y_val)
    rows.append({"Metoda": f"Lasso (alfa={lasso_alfa})", "RMSE": rmse, "R2_adj": r2_adj})
    models["Lasso"] = m_lasso

    # --- Elastic Net ---
    m_en = elastic_net.fit(x_train, y_train, alfa=en_alfa, l1_ratio=en_l1_ratio, epochs=cd_epochs, intercept=True)
    w_en = WrappedModel(elastic_net, m_en)
    rmse, r2_adj = _evaluate_with_utils(w_en, x_val, y_val)
    rows.append({"Metoda": f"ElasticNet (alfa={en_alfa}, l1_ratio={en_l1_ratio})", "RMSE": rmse, "R2_adj": r2_adj})
    models["ElasticNet"] = m_en

    # --- Huber ---
    m_huber = huber.fit(x_train, y_train, delta=huber_delta, lr=gd_lr, epochs=gd_epochs, intercept=True)
    w_huber = WrappedModel(huber, m_huber)
    rmse, r2_adj = _evaluate_with_utils(w_huber, x_val, y_val)
    rows.append({"Metoda": f"Huber (delta={huber_delta})", "RMSE": rmse, "R2_adj": r2_adj})
    models["Huber"] = m_huber

    results_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    data_out = {
        "df": df,
        "x": x,
        "y": y,
        "x_train": x_train,
        "x_val": x_val,
        "y_train": y_train,
        "y_val": y_val,
        "srednje": srednje,
        "std": std,
    }

    return results_df, models, data_out

#  Stabilnost koeficijenata
def stabilnost_svih_metoda(
    putanja_train_csv,
    cilj,
    feature_kolone,
    repeats=20,
    nacin_nedostajuce="dropna",
    dummy_flag=False,
    standardizacija_flag=True,
    # hiperparametri 
    ridge_alfa=0.1,
    lasso_alfa=0.1,
    en_alfa=0.1,
    en_l1_ratio=0.5,
    huber_delta=1.0,
    gd_lr=0.01,
    gd_epochs=800,
    cd_epochs=50,
):
    # mean (prosek koeficijenta)
    # std  (standardna devijacija koeficijenta)
    metode = ["OLS", "Ridge", "Lasso", "ElasticNet", "Huber"]
    feature_kolone = list(feature_kolone)

    # skupljamo w po metodi:
    all_w = {m: [] for m in metode}

    for rs in range(int(repeats)):
        _, models, _ = train_all_models(
            putanja_train_csv=putanja_train_csv,
            cilj=cilj,
            feature_kolone=feature_kolone,
            nacin_nedostajuce=nacin_nedostajuce,
            dummy_flag=dummy_flag,
            standardizacija_flag=standardizacija_flag,
            random_state=rs,
            noise_sigma=0.0,          #stabilnost bez dodatnog šuma
            ridge_alfa=ridge_alfa,
            lasso_alfa=lasso_alfa,
            en_alfa=en_alfa,
            en_l1_ratio=en_l1_ratio,
            huber_delta=huber_delta,
            gd_lr=gd_lr,
            gd_epochs=gd_epochs,
            cd_epochs=cd_epochs,
        )

        for m in metode:
            w = models[m]["w"]
            all_w[m].append(w)

    rows = []
    for m in metode:
        W = np.array(all_w[m]) 
        mean = W.mean(axis=0)
        std = W.std(axis=0)

        for j, feat in enumerate(feature_kolone):
            rows.append({
                "Metoda": m,
                "Feature": feat,
                "Mean_coef": float(mean[j]),
                "Std_coef": float(std[j]),
            })

    return pd.DataFrame(rows)

#  Eksperiment sa kontrolisanim šumom
def eksperiment_sa_sumom(
    putanja_train_csv,
    cilj,
    feature_kolone,
    noise_sigma_values=(0.0, 0.05, 0.1),
    random_state=42,
    nacin_nedostajuce="dropna",
    dummy_flag=False,
    standardizacija_flag=True,
    # hiperparametri
    ridge_alfa=0.1,
    lasso_alfa=0.1,
    en_alfa=0.1,
    en_l1_ratio=0.5,
    huber_delta=1.0,
    gd_lr=0.01,
    gd_epochs=1200,
    cd_epochs=70,
):
    rows = []
    for sig in noise_sigma_values:
        res_df, _, _ = train_all_models(
            putanja_train_csv=putanja_train_csv,
            cilj=cilj,
            feature_kolone=feature_kolone,
            nacin_nedostajuce=nacin_nedostajuce,
            dummy_flag=dummy_flag,
            standardizacija_flag=standardizacija_flag,
            random_state=random_state,
            noise_sigma=float(sig),
            ridge_alfa=ridge_alfa,
            lasso_alfa=lasso_alfa,
            en_alfa=en_alfa,
            en_l1_ratio=en_l1_ratio,
            huber_delta=huber_delta,
            gd_lr=gd_lr,
            gd_epochs=gd_epochs,
            cd_epochs=cd_epochs,
        )
        tmp = res_df.copy()
        tmp["noise_sigma"] = float(sig)
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)