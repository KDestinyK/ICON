"""
modulo_ml.py - Apprendimento Supervisionato Comparativo
=======================================================
Confronta tre classificatori (Random Forest, SVM, Logistic Regression)
tramite K-Fold cross-validation con ottimizzazione degli iperparametri
via GridSearchCV. Restituisce il modello vincitore per la pipeline principale.

Classi target nel dataset (colonna 'result'):
    0 = Nessuna irrigazione richiesta (terreno umido, temp bassa)
    1 = Irrigazione necessaria (terreno secco, alta temperatura)
    2 = Stress idrico / anomalia (MOI molto alto + alta temp = rischio malattia)

Nota prestazioni: la GridSearch e la learning curve girano su un sottoinsieme
stratificato per contenere i tempi (vincolo 25 ore/progetto); la valutazione
comparativa finale (RepeatedKFold) usa l'intero dataset.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedKFold, cross_validate, GridSearchCV, learning_curve, train_test_split
)
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def _ottimizza_iperparametri(X, y):
    """GridSearchCV su sottoinsieme stratificato. Ritorna {nome: estimatore_ottimo}."""
    print("   [GridSearch] Ricerca iperparametri (cv=3 su sottoinsieme stratificato)...")
    n_gs = min(4000, len(y))
    X_gs, _, y_gs, _ = train_test_split(X, y, train_size=n_gs, random_state=42, stratify=y)

    grids = {
        "Random Forest": (RandomForestClassifier(random_state=42, n_jobs=-1),
                          {"n_estimators": [100, 200], "max_depth": [None, 15],
                           "criterion": ["gini", "entropy"]}),
        "SVM": (SVC(random_state=42),
                {"C": [1, 10], "kernel": ["rbf"], "gamma": ["scale"]}),
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42),
                                {"C": [0.1, 1, 10], "solver": ["lbfgs"]}),
    }

    migliori = {}
    for nome, (modello, grid) in grids.items():
        gs = GridSearchCV(modello, grid, cv=3, scoring="f1_weighted", n_jobs=-1)
        gs.fit(X_gs, y_gs)
        migliori[nome] = gs.best_estimator_
        print(f"   -> {nome}: {gs.best_params_}")
    return migliori


def _curva_apprendimento(modello, X, y, nome_modello):
    """Stampa la curva di apprendimento (train/test accuracy) con diagnosi bias/varianza."""
    n_lc = min(5000, len(y))
    X_lc, _, y_lc, _ = train_test_split(X, y, train_size=n_lc, random_state=42, stratify=y)
    train_sizes, train_scores, test_scores = learning_curve(
        modello, X_lc, y_lc, cv=3, scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1)

    print(f"\n   [Curva Apprendimento] {nome_modello}:")
    print(f"   {'TrainSize':<12} {'Train Acc':<12} {'Test Acc':<12}")
    for ts, tr, te in zip(train_sizes, np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)):
        print(f"   {int(ts):<12} {tr:<12.4f} {te:<12.4f}")

    gap = np.mean(train_scores[-1]) - np.mean(test_scores[-1])
    if gap > 0.10:
        print(f"   [Diagnosi] Gap train-test = {gap:.3f} -> possibile OVERFITTING")
    elif np.mean(test_scores[-1]) < 0.70:
        print(f"   [Diagnosi] Test accuracy = {np.mean(test_scores[-1]):.3f} -> possibile UNDERFITTING")
    else:
        print(f"   [Diagnosi] Gap train-test = {gap:.3f} -> buon equilibrio bias/varianza")


def esegui_machine_learning(percorso_csv):
    """Pipeline ML completa. Returns (modello_vincitore, scaler, feature_columns)."""
    print("--- [ML] Apprendimento Supervisionato Comparativo ---")

    print("1. Caricamento e preprocessing...")
    try:
        df = pd.read_csv(percorso_csv).dropna()
    except FileNotFoundError:
        print(f"   ERRORE: File '{percorso_csv}' non trovato.")
        return None, None, None

    X = pd.get_dummies(df.drop("result", axis=1))
    y = df["result"]
    feature_columns = X.columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"   {df.shape[0]} campioni, {X_scaled.shape[1]} feature, {y.nunique()} classi")
    print(f"   Distribuzione classi: {dict(y.value_counts().sort_index())}")
    print(f"   0=No irrigazione | 1=Irrigazione | 2=Stress idrico\n")

    migliori = _ottimizza_iperparametri(X_scaled, y)

    print("\n2. Valutazione comparativa (RepeatedKFold 5x5)...")
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall": make_scorer(recall_score, average="weighted", zero_division=0),
        "f1": make_scorer(f1_score, average="weighted", zero_division=0),
    }

    risultati = {}
    for nome, modello in migliori.items():
        risultati[nome] = cross_validate(modello, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)

    sep = "-" * 92
    print(sep)
    print(f"{'MODELLO':<22} | {'ACCURACY':^16} | {'PRECISION':^16} | {'RECALL':^16} | {'F1':^16}")
    print(sep)
    f1_per_modello = {}
    for nome, res in risultati.items():
        a_m, a_s = np.mean(res["test_accuracy"]),  np.std(res["test_accuracy"])
        p_m, p_s = np.mean(res["test_precision"]), np.std(res["test_precision"])
        r_m, r_s = np.mean(res["test_recall"]),    np.std(res["test_recall"])
        f_m, f_s = np.mean(res["test_f1"]),        np.std(res["test_f1"])
        f1_per_modello[nome] = f_m
        print(f"{nome:<22} | {a_m:.4f}±{a_s:.4f} | {p_m:.4f}±{p_s:.4f} | "
              f"{r_m:.4f}±{r_s:.4f} | {f_m:.4f}±{f_s:.4f}")
    print(sep)

    nome_vincitore = max(f1_per_modello, key=f1_per_modello.get)
    print(f"\n-> MODELLO VINCITORE (criterio F1 weighted): {nome_vincitore.upper()}")
    modello_vincitore = migliori[nome_vincitore]
    _curva_apprendimento(modello_vincitore, X_scaled, y, nome_vincitore)

    modello_vincitore.fit(X_scaled, y)
    print("\n-> Modello vincitore addestrato sull'intero dataset.\n")
    return modello_vincitore, scaler, feature_columns


if __name__ == "__main__":
    esegui_machine_learning("cropdata_updated.csv")
