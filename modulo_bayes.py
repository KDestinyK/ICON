"""
modulo_bayes.py - Ragionamento Probabilistico con Rete Bayesiana
================================================================
Costruisce una Rete Bayesiana apprendendo la struttura dai dati reali
tramite HillClimbSearch (scoring BIC-d) e i parametri tramite MLE.

Variabili discretizzate:
    Temperatura      : 0=bassa (<=mediana), 1=alta (>mediana)
    Umidita_Aria     : 0=bassa, 1=alta
    Umidita_Suolo    : 0=bassa (MOI<=mediana), 1=alta
    Stadio_Crescita  : stringa categorica (Germination, Flowering, ...)
    Tipo_Terreno     : stringa categorica (Black Soil, Sandy Soil, ...)
    Target_Irrigazione: 0=No irrigazione | 1=Irrigazione necessaria | 2=Stress idrico

La BN viene valutata in due modi:
    1. BIC comparativo: struttura appresa vs baseline Naive Bayes
    2. K-Fold predittivo (5 split): accuracy media ± std
"""

import warnings
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

warnings.filterwarnings("ignore")


def _discretizza_dataset(df):
    """
    Discretizza le variabili continue in variabili binarie/categoriche
    usando la mediana come soglia per le variabili numeriche.
    Restituisce il DataFrame discretizzato e le soglie usate (per inferenza futura).
    """
    df_bn = pd.DataFrame()
    soglie = {}

    # Temperatura
    col_temp = [c for c in df.columns if "temp" in c.lower()][0]
    soglia_t = df[col_temp].median()
    df_bn["Temperatura"] = (df[col_temp] > soglia_t).astype(int).astype(str)
    soglie["temp"] = (col_temp, soglia_t)

    # Umidità aria
    col_umid = [c for c in df.columns if "humid" in c.lower() or "umid" in c.lower()][0]
    soglia_u = df[col_umid].median()
    df_bn["Umidita_Aria"] = (df[col_umid] > soglia_u).astype(int).astype(str)
    soglie["humidity"] = (col_umid, soglia_u)

    # Umidità suolo (MOI)
    col_moi = [c for c in df.columns if "moi" in c.lower()]
    if col_moi:
        soglia_m = df[col_moi[0]].median()
        df_bn["Umidita_Suolo"] = (df[col_moi[0]] > soglia_m).astype(int).astype(str)
        soglie["moi"] = (col_moi[0], soglia_m)

    # Stadio di crescita (categorica)
    col_stage = [c for c in df.columns if "seedling" in c.lower() or "stage" in c.lower()]
    if col_stage:
        df_bn["Stadio_Crescita"] = df[col_stage[0]].astype(str)

    # Target (rinominato per chiarezza semantica)
    df_bn["Target_Irrigazione"] = df["result"].astype(str)

    return df_bn, soglie


def esegui_rete_bayesiana(percorso_csv):
    """
    Pipeline completa della Rete Bayesiana:
      1. Caricamento e discretizzazione dei dati reali
      2. Structure Learning (HillClimbSearch + BIC-d)
      3. Parameter Learning (MLE)
      4. Valutazione comparativa BIC: modello appreso vs Naive Bayes baseline
      5. Valutazione predittiva K-Fold (5 split) con media ± std
      6. Stampa CPD del nodo target

    Returns:
        modello: DiscreteBayesianNetwork addestrato
        df: DataFrame originale (per la pipeline)
        df_bn: DataFrame discretizzato
        soglie: dict con le soglie usate per la discretizzazione
        colonne_bn: lista di variabili nella BN
    """
    print("--- [BN] Ragionamento Probabilistico con Rete Bayesiana ---")

    # 1. Caricamento e discretizzazione
    print("1. Caricamento e discretizzazione delle variabili...")
    df = pd.read_csv(percorso_csv).dropna()
    df_bn, soglie = _discretizza_dataset(df)

    print(f"   Variabili nella BN: {list(df_bn.columns)}")
    print(f"   Soglie di discretizzazione:")
    for key, (col, soglia) in soglie.items():
        print(f"      {col}: mediana = {soglia:.2f}")

    # 2. Structure Learning
    print("\n2. Structure Learning (HillClimbSearch, scoring=BIC-d)...")
    hc = HillClimbSearch(df_bn)
    struttura = hc.estimate(scoring_method="bic-d", max_iter=200)
    archi = list(struttura.edges())
    print(f"   Archi causali scoperti: {archi}")

    if not archi:
        print("   [Fallback] Nessun arco trovato: applico struttura a stella sul Target.")
        archi = [
            (col, "Target_Irrigazione")
            for col in df_bn.columns
            if col != "Target_Irrigazione"
        ]

    # 3. Parameter Learning
    print("\n3. Parameter Learning (Maximum Likelihood Estimator)...")
    modello = DiscreteBayesianNetwork(archi)
    modello.fit(df_bn)
    print("   Apprendimento parametri completato.")

    # 4. Valutazione BIC comparativa 
    print("\n4. Valutazione comparativa: BIC Appreso vs BIC Naive Bayes (baseline)...")
    try:
        from pgmpy.structure_score import BIC

        score_appreso = BIC(df_bn).score(modello)

        archi_naive = [
            (col, "Target_Irrigazione")
            for col in df_bn.columns
            if col != "Target_Irrigazione"
        ]
        modello_naive = DiscreteBayesianNetwork(archi_naive)
        modello_naive.fit(df_bn)
        score_naive = BIC(df_bn).score(modello_naive)

        print(f"   BIC Modello Appreso (HillClimb) = {score_appreso:.2f}")
        print(f"   BIC Baseline Naive Bayes        = {score_naive:.2f}")

        # BIC: più alto è meglio (penalizza la complessità)
        if score_appreso >= score_naive:
            print("   [RISULTATO] La struttura appresa è MIGLIORE o equivalente alla baseline.")
        else:
            diff = score_naive - score_appreso
            print(f"   [RISULTATO] La baseline ha BIC superiore di {diff:.2f}. La struttura "
                  f"appresa è più complessa ma potenzialmente più espressiva.")
    except Exception as e:
        print(f"   [Errore calcolo BIC: {e}]")

    # 5. Valutazione predittiva K-Fold 
    print("\n5. Valutazione predittiva K-Fold (5 split, media ± std)...")
    try:
        from sklearn.model_selection import KFold
        from sklearn.metrics import accuracy_score, f1_score

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        acc_scores = []
        f1_scores_list = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(df_bn)):
            train_bn = df_bn.iloc[train_idx]
            test_bn = df_bn.iloc[test_idx]

            mod_fold = DiscreteBayesianNetwork(modello.edges())
            # Alleniamo solo sulle variabili effettivamente presenti nella struttura
            # appresa (HillClimbSearch può escludere variabili non informative)
            nodi_modello = list(mod_fold.nodes())
            mod_fold.fit(train_bn[nodi_modello])

            # In input alla predict passiamo solo le feature del modello (escluso il target)
            feature_modello = [n for n in nodi_modello if n != "Target_Irrigazione"]
            X_test = test_bn[feature_modello]
            y_true = test_bn["Target_Irrigazione"].values

            try:
                y_pred_df = mod_fold.predict(X_test)
                y_pred = y_pred_df["Target_Irrigazione"].values
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                acc_scores.append(acc)
                f1_scores_list.append(f1)
            except Exception as e_fold:
                print(f"   [Fold {fold+1} errore predizione: {e_fold}]")

        if acc_scores:
            acc_m, acc_s = np.mean(acc_scores), np.std(acc_scores)
            f1_m, f1_s   = np.mean(f1_scores_list), np.std(f1_scores_list)
            print(f"   Accuracy  = {acc_m:.4f} ± {acc_s:.4f}")
            print(f"   F1 weighted = {f1_m:.4f} ± {f1_s:.4f}")
            if acc_m < 0.80:
                print("   [Nota] Accuracy fisiologicamente inferiore ai discriminativi: "
                      "la BN è un modello generativo, non ottimizzato per la classificazione.")
    except Exception as e:
        print(f"   [Errore valutazione predittiva: {e}]")

    # 6. CPD del nodo Target 
    print("\n6. CPD del nodo Target_Irrigazione (appresa dai dati):")
    try:
        print(modello.get_cpds("Target_Irrigazione"))
    except Exception:
        print("   (Target non nodo radice, CPD condizionale)")

    return modello, df, df_bn, soglie, list(df_bn.columns)


if __name__ == "__main__":
    esegui_rete_bayesiana("cropdata_updated.csv")
