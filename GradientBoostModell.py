import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import PolynomialFeatures

"""
Skripit zur Erstellung eines Rankings für Anchor-Papers
Trainier Gradienten-Boosting-Modell (LightGBM) auf Basis von Features
Nutzt Cross-Validation mit StratifiedShuffleSplit und printet die durchnittlichen Ergebnisse auf dem Test Split.
Außerdem OOF-Predictions für jedes Paper.
"""


# ─── CONFIG ───────────────────────────────────────────
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

N_SPLITS = 10
TEST_SIZE = 0.4
NEG_POS_RATIO = 70
TOP_K_LIST = [10, 50, 100, 103, 206, 5148]

RANDOM_STATE = 42
# ──────────────────────────────────────────────────────

#Daten laden
df = pd.read_csv(DATA_PATH)

#Spalten umbenennen
df.rename(columns={
    "d4": "cited_by_count",
    "d3": "survey_bonus",
    "d2": "publication_type_score",
    "d1": "journal_publisher_score"
}, inplace=True)

df["is_anchor"] = df["Id"].isin(ANCHOR_IDS).astype(int)
df[FEATURES_BASE] = df[FEATURES_BASE].fillna(0)

#Interaktionsterme erzeugen
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_inter = poly.fit_transform(df[FEATURES_TO_INTERACT])

interaction_feature_names = [
    n.replace(" ", "_") for n in poly.get_feature_names_out(FEATURES_TO_INTERACT)
]

interaction_df = pd.DataFrame(X_inter, columns=interaction_feature_names,
                              index=df.index)
interaction_df.drop(columns=FEATURES_TO_INTERACT, inplace=True)

df = pd.concat([df, interaction_df], axis=1)

FEATURES_ALL = FEATURES_BASE + list(interaction_df.columns)

X_all = df[FEATURES_ALL].values
y_all = df["is_anchor"].values

#Out-of-Fold Arrays initialisieren
oof_preds = np.zeros_like(y_all, dtype=float)
oof_counts = np.zeros_like(y_all, dtype=int)

#Cross-Validation Splits
sss = StratifiedShuffleSplit(
    n_splits=N_SPLITS,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

results = []
recall_curves = []

for split_idx, (train_idx, test_idx) in enumerate(sss.split(X_all, y_all)):
    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    #Negative Sampling
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    neg_sample_idx = np.random.choice(
        neg_idx,
        size=len(pos_idx) * NEG_POS_RATIO,
        replace=False
    )

    sample_idx = np.concatenate([pos_idx, neg_sample_idx])

    X_train_sample = X_train[sample_idx]
    y_train_sample = y_train[sample_idx]

    #LightGBM Dataset
    train_data = lgb.Dataset(X_train_sample, label=y_train_sample)

    params = {
        'objective': 'binary',
        'metric': ['auc'],
        'learning_rate': 0.05,
        'scale_pos_weight': NEG_POS_RATIO,
        'verbose': -1
    }

    bst = lgb.train(
        params,
        train_data,
        num_boost_round=200
    )

    #Predict on Test
    y_test_pred = bst.predict(X_test)

    #Speichere Out-of-Fold Predictions
    oof_preds[test_idx] += y_test_pred
    oof_counts[test_idx] += 1

    #Ranking innerhalb des Test-Sets
    idx_sorted = np.argsort(y_test_pred)[::-1]
    sorted_y_true = y_test[idx_sorted]

    #Cumulative sum = wie viele Anchors in Top-K?
    cum_sum = np.cumsum(sorted_y_true)
    recall_curve = cum_sum / cum_sum[-1] if cum_sum[-1] > 0 else np.zeros_like(cum_sum)

    recall_curves.append(recall_curve)

    anchors_in_top_k = {}
    for k in TOP_K_LIST:
        found = cum_sum[k - 1] if k <= len(cum_sum) else cum_sum[-1]
        anchors_in_top_k[k] = found

    auc = roc_auc_score(y_test, y_test_pred)
    ap = average_precision_score(y_test, y_test_pred)

    results.append({
        "Split": split_idx,
        "AUC": auc,
        "AP": ap,
        **anchors_in_top_k
    })

    print(f"Split {split_idx+1}/{N_SPLITS}: AUC={auc:.3f}, AP={ap:.3f}, Anchors in Top-100 = {anchors_in_top_k[100]}")

#Mittelwert bilden für OOF-Predictions
nonzero = oof_counts > 0
oof_preds[nonzero] /= oof_counts[nonzero]
oof_preds[~nonzero] = np.nan

# Ergebnisse der Splits zusammenfassen
results_df = pd.DataFrame(results)
print("\n=== Ergebnis über alle Splits ===\n")
print(results_df.describe().round(3))

#Finale OOF-Rankings
final_ranking_df = pd.DataFrame({
    "Id": df["Id"],
    "Label": df["Label"],
    "is_anchor": y_all,
    "oof_score": oof_preds
})

final_ranking_df = final_ranking_df.sort_values("oof_score", ascending=False)

#Top 20 Papers anzeigen
print("\n=== Top 20 Papers nach OOF-Score ===\n")
pd.set_option('display.max_columns', 7)
print(final_ranking_df.head(20))
