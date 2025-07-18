
"""
Trainiert eine Logistische Regression auf allen Daten
und berechnet eine Gewichtungsformel für Anchor-Paper-Erkennung.

Zusätzlich:
- Bootstrap auf allen Daten zur Stabilitätsanalyse der Koeffizienten
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

# ─── CONFIG ─────────────────────────────────────────────────────────────
DATA_PATH = Path(r"F:\PaperBA\FinalData\GephiExportfinal2_mehrV2.csv")

ANCHOR_IDS = [
    "W2144020560", "W1512387364", "W2098774185", "W1492581097", "W4385245566",
    "W2155541015", "W2162670686", "W4318149317", "W1591801644", "W2507756961",
    "W2194775991", "W2963840672", "W2606780347", "W2964308564", "W2302255633",
    "W2963907629", "W2950527759", "W2193413348", "W3001279689", "W1901129140",
    "W2896457183", "W3094502228", "W2165150801", "W4298857966", "W1995341919",
    "W2257979135", "W2064675550", "W1498436455"
]

FEATURES_BASE = [
    "cited_by_count", "survey_bonus", "publication_type_score",
    "journal_publisher_score", "indegree", "outdegree", "pageranks",
    "eigencentrality", "Authority", "Hub", "Eccentricity",
    "closnesscentrality", "betweenesscentrality"
]

FEATURES_TO_INTERACT = ["betweenesscentrality", "pageranks",
                        "eigencentrality", "Authority"]

RANDOM_STATE = 42
N_BOOT = 1000
NEG_POS_RATIO = 250
# ────────────────────────────────────────────────────────────────────────

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 1. Daten laden & vorbereiten                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

df = pd.read_csv(DATA_PATH)

# Falls d1–d4 noch Rohspalten sind → umbenennen
df.rename(columns={
    "d4": "cited_by_count",
    "d3": "survey_bonus",
    "d2": "publication_type_score",
    "d1": "journal_publisher_score"
}, inplace=True)

df["is_anchor"] = df["Id"].isin(ANCHOR_IDS).astype(int)

df[FEATURES_BASE] = df[FEATURES_BASE].fillna(0)

# Interaktionsterme erzeugen
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_inter = poly.fit_transform(df[FEATURES_TO_INTERACT])

interaction_feature_names = [
    n.replace(" ", "_") for n in poly.get_feature_names_out(FEATURES_TO_INTERACT)
]

interaction_df = pd.DataFrame(X_inter, columns=interaction_feature_names,
                              index=df.index)
interaction_df.drop(columns=FEATURES_TO_INTERACT, inplace=True)

df = pd.concat([df, interaction_df], axis=1)

FEATURES_ALL = FEATURES_BASE + interaction_df.columns.tolist()

X_all = df[FEATURES_ALL].values
y_all = df["is_anchor"].values

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 2. Modell trainieren auf ALLEN Daten                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
)

pipe.fit(X_all, y_all)

scaler = pipe.named_steps["standardscaler"]
clf = pipe.named_steps["logisticregression"]

beta_std = clf.coef_.flatten()
beta_raw = beta_std / scaler.scale_
beta0_raw = clf.intercept_[0] - np.dot(beta_raw, scaler.mean_)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 3. Gewichtungsformel ausgeben                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n### Gewichtungsformel (Originalskala, Alle Daten) ###")
print(f"Score = {beta0_raw:+.6f} +")
for coef, feat in zip(beta_raw, FEATURES_ALL):
    print(f"        {coef:+.6f} × {feat}")

print("\n### Gewichtungsformel mit standardisierten Features (Alle Daten) ###")
print(f"Standardisierter Score = {clf.intercept_[0]:+.6f} +")
for coef, feat in zip(beta_std, FEATURES_ALL):
    print(f"                      {coef:+.6f} × z({feat})")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 4. Bootstrap auf ALLEN Daten                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n### Starte Bootstrap auf ALLEN Daten ###")

idx_pos_all = np.where(y_all == 1)[0]
idx_neg_all = np.where(y_all == 0)[0]

coefs_boot = []
rng = np.random.default_rng(RANDOM_STATE)

for i in range(N_BOOT):
    idx_pos_sample = rng.choice(
        idx_pos_all,
        size=len(idx_pos_all),
        replace=True
    )
    idx_neg_sample = rng.choice(
        idx_neg_all,
        size=len(idx_pos_all) * NEG_POS_RATIO,
        replace=True
    )
    idx_sample = np.concatenate([idx_pos_sample, idx_neg_sample])

    X_sample = X_all[idx_sample]
    y_sample = y_all[idx_sample]

    try:
        pipe_b = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced"
            )
        )
        pipe_b.fit(X_sample, y_sample)

        scaler_b = pipe_b.named_steps["standardscaler"]
        clf_b = pipe_b.named_steps["logisticregression"]
        beta_std_b = clf_b.coef_.flatten()
        coefs_boot.append(beta_std_b)

    except Exception:
        continue

coefs_boot = np.array(coefs_boot)

# Median und KIs für β
beta_median = np.median(coefs_boot, axis=0)
lower = np.percentile(coefs_boot, 2.5, axis=0)
upper = np.percentile(coefs_boot, 97.5, axis=0)

print("\n### 95%-Bootstrap-Konfidenzintervalle der β-Koeffizienten ###")
for feat, med, lo, hi in zip(FEATURES_ALL, beta_median, lower, upper):
    print(f"{feat:<40s} β = {med:+.4f}   ({lo:+.4f} … {hi:+.4f})")
