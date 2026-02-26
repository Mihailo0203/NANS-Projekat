import pandas as pd
import matplotlib.pyplot as plt

def prikazi_rezultate(results_df: pd.DataFrame):
    """Prikaz tabele rezultata i bar-grafika RMSE (ako postoji kolona RMSE)."""
    display(results_df)

    if "RMSE" in results_df.columns and "Metoda" in results_df.columns:
        ax = results_df.set_index("Metoda")["RMSE"].plot(kind="bar")
        plt.title("RMSE po metodama (manje = bolje)")
        plt.ylabel("RMSE")
        plt.tight_layout()
        plt.show()

def koeficijenti_u_tabelu(models: dict, feature_names: list[str]) -> pd.DataFrame:
    """Pretvara recnik modela (w,b) u tabelu koeficijenata."""
    rows = []
    for name, m in models.items():
        w = m.get("w", None)
        b = m.get("b", None)
        if w is None:
            continue
        row = {"Metoda": name, "b0": b}
        for j, fn in enumerate(feature_names):
            row[fn] = w[j]
        rows.append(row)
    return pd.DataFrame(rows)

def prikazi_stabilnost(stab_df: pd.DataFrame):
    """Sažet prikaz stabilnosti: top 10 nestabilnih + graf po feature-u."""
    # Prosečna std po metodi
    if set(["Metoda","Std_coef"]).issubset(stab_df.columns):
        summary = (stab_df.groupby("Metoda")["Std_coef"]
                   .mean().sort_values().reset_index()
                   .rename(columns={"Std_coef":"Prosecni_std_koef"}))
        display(summary)

    # Top nestabilnih
    if "Std_coef" in stab_df.columns:
        display(stab_df.sort_values("Std_coef", ascending=False).head(10))

    # Graf po feature-u
    if set(["Feature","Metoda","Std_coef"]).issubset(stab_df.columns):
        pivot = stab_df.pivot_table(index="Feature", columns="Metoda", values="Std_coef", aggfunc="mean")
        pivot.plot(kind="bar")
        plt.title("Std koeficijenata po feature-u (manje = stabilnije)")
        plt.ylabel("Std_coef")
        plt.tight_layout()
        plt.show()

def prikazi_sum(noise_df: pd.DataFrame):
    """Prikaz tabele + graf RMSE kroz nivo šuma."""
    display(noise_df)

    if set(["noise_sigma","RMSE","Metoda"]).issubset(noise_df.columns):
        pivot = noise_df.pivot_table(index="noise_sigma", columns="Metoda", values="RMSE", aggfunc="mean")
        pivot.plot(marker="o")
        plt.title("RMSE u zavisnosti od nivoa šuma")
        plt.xlabel("noise_sigma")
        plt.ylabel("RMSE")
        plt.tight_layout()
        plt.show()
