"""
main.py - Sistema Esperto per l'Agricoltura di Precisione
=========================================================
Pipeline multi-livello che integra tre paradigmi dell'Ingegneria della Conoscenza:

    [FASE 1] Apprendimento Supervisionato (ML)
        Classifica lo stato idrico del terreno: 0=ok, 1=irrigazione, 2=stress.
        Tre modelli comparati (RF, SVM, LR) con GridSearchCV + RepeatedKFold.

    [FASE 2] Ragionamento Probabilistico (Rete Bayesiana)
        Struttura appresa dai dati (HillClimbSearch + BIC), parametri con MLE.
        Dado l'output ML come evidenza, stima la probabilità di conferma
        e la trasforma in un segnale di "affidabilita_meteo" per Prolog.

    [FASE 3] Rappresentazione e Ragionamento Logico (Prolog)
        KB con 16 colture, 7 famiglie botaniche, regole agronomiche su
        rotazione, stagionalità, tolleranza idrica, fertilizzazione, allerte.
        Riceve i due fatti dinamici prodotti da ML e BN e deduce l'azione.

Flusso dati (integrazione reale tra i livelli):
    CSV -> ML (predizione classe) -> BN (probabilità di conferma) -> Prolog (consiglio)

Semantica classi dataset (colonna 'result'):
    0 = Nessuna irrigazione (terreno umido, temperatura bassa)
    1 = Irrigazione necessaria (terreno secco, alta temperatura)
    2 = Stress idrico / anomalia (MOI molto alto + temperatura elevata)
"""

import os
import warnings
import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

import modulo_ml
import modulo_bayes
from modulo_prolog import ModuloProlog

warnings.filterwarnings("ignore")

os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin" + os.pathsep + "/usr/local/bin"

# Dizionario di mapping: nomi inglesi del CSV -> nomi italiani dell'ontologia
DIZIONARIO_COLTURE = {
    "wheat":    "grano",
    "potato":   "patata",
    "carrot":   "carota",    # non in KB, verrà gestito col fallback
    "tomato":   "pomodoro",
    "chilli":   "peperone",
}

# Mesi e colture precedenti simulati per i 3 scenari
# (il dataset non contiene queste informazioni: scelta documentata)
SCENARI_CONTESTO = [
    {"mese": "aprile",   "coltura_prec": "patata"},
    {"mese": "ottobre",  "coltura_prec": "zucchina"},
    {"mese": "novembre", "coltura_prec": "pomodoro"},
]

NOME_CSV = "cropdata_updated.csv"


def traduci_coltura(nome_inglese: str) -> str:
    """Mappa un nome coltura inglese (CSV) all'equivalente italiano nella KB Prolog."""
    return DIZIONARIO_COLTURE.get(nome_inglese.strip().lower(), "pomodoro")


def calcola_bisogno_acqua(classe_ml: int) -> str:
    """
    Converte la classe predetta dal ML in un fatto booleano per Prolog.

    Semantica:
        Classe 1 -> bisogno_acqua(si)  [alta temp, bassa umidità, MOI bassa]
        Classe 0 -> bisogno_acqua(no)  [condizioni normali, no irrigazione]
        Classe 2 -> bisogno_acqua(no)  [stress idrico: il terreno è già saturo,
                                        irrigazione aggiungerebbe danno]
    """
    return "si" if classe_ml == 1 else "no"


def calcola_affidabilita_meteo(
    modello_bn,
    df_completo: pd.DataFrame,
    df_bn: pd.DataFrame,
    soglie: dict,
    campione: pd.DataFrame,
    classe_ml: int,
) -> tuple[str, float]:
    """
    Usa la Rete Bayesiana per stimare la probabilità che il campione corrente
    appartenga alla classe predetta dal ML (dato temperatura e umidità dell'aria).

    Restituisce:
        affidabilita: 'alta' se prob > 0.5, 'bassa' altrimenti
        prob: probabilità grezza della classe ML
    """
    evidenza = {}
    nodi_bn = list(modello_bn.nodes())

    if "Temperatura" in nodi_bn:
        col_temp, soglia_t = soglie["temp"]
        val_temp = campione[col_temp].values[0]
        evidenza["Temperatura"] = "1" if val_temp > soglia_t else "0"

    if "Umidita_Aria" in nodi_bn:
        col_umid, soglia_u = soglie["humidity"]
        val_umid = campione[col_umid].values[0]
        evidenza["Umidita_Aria"] = "1" if val_umid > soglia_u else "0"

    if "Umidita_Suolo" in nodi_bn and "moi" in soglie:
        col_moi, soglia_m = soglie["moi"]
        val_moi = campione[col_moi].values[0]
        evidenza["Umidita_Suolo"] = "1" if val_moi > soglia_m else "0"

    try:
        inferenza = VariableElimination(modello_bn)
        risultato = inferenza.query(
            variables=["Target_Irrigazione"], evidence=evidenza, show_progress=False
        )
        stato_predetto = str(classe_ml)
        stati = risultato.state_names["Target_Irrigazione"]
        if stato_predetto in stati:
            prob = float(risultato.values[stati.index(stato_predetto)])
        else:
            prob = 0.0
        affidabilita = "alta" if prob > 0.50 else "bassa"
        return affidabilita, prob
    except Exception as e:
        print(f"  [BN] Errore inferenza: {e} -> fallback a 'bassa'")
        return "bassa", 0.0


def esegui_pipeline():
    print("=" * 80)
    print("  SISTEMA ESPERTO MULTI-LIVELLO: SMART AGRICULTURE")
    print("  Integrazione ML + Reti Bayesiane + Ragionamento Prolog")
    print("=" * 80)

    # FASE 1: Machine Learning Comparativo

    print("\n[FASE 1] Apprendimento Supervisionato Comparativo")
    print("-" * 80)
    modello_ml, scaler, feature_columns = modulo_ml.esegui_machine_learning(NOME_CSV)
    if modello_ml is None:
        print("ERRORE FATALE: ML non completato. Verificare il file CSV.")
        return

    # FASE 2: Rete Bayesiana (Structure + Parameter Learning)

    print("\n[FASE 2] Ragionamento Probabilistico con Rete Bayesiana")
    print("-" * 80)
    modello_bn, df_completo, df_bn, soglie, colonne_bn = modulo_bayes.esegui_rete_bayesiana(NOME_CSV)

    # FASE 3: Caricamento Ontologia Prolog
  
    print("\n[FASE 3] Caricamento Ontologia Prolog (KB Agricoltura)")
    print("-" * 80)
    prolog = ModuloProlog("kb_agricoltura.pl")

    # FASE 4: Pipeline Integrata su Dati Reali (3 Scenari)

    print("\n" + "=" * 80)
    print("  [FASE 4] SIMULAZIONE SU DATI REALI: 3 SCENARI INTEGRATI")
    print("  (ML predice -> BN conferma/smentisce -> Prolog consiglia)")
    print("=" * 80)

    # Identificazione colonna coltura nel dataset
    col_crop = next(
        (c for c in df_completo.columns if "crop" in c.lower() or "label" in c.lower()),
        None
    )

    # Selezioniamo campioni diversi per classe (uno per ciascuna classe target)
    campioni_per_classe = {}
    for classe in [0, 1, 2]:
        subset = df_completo[df_completo["result"] == classe]
        if not subset.empty:
            campioni_per_classe[classe] = subset.sample(1, random_state=42 + classe)

    scenari_usati = list(campioni_per_classe.values())[:3]

    for i, campione in enumerate(scenari_usati):
        contesto = SCENARI_CONTESTO[i]
        mese = contesto["mese"]
        coltura_prec = contesto["coltura_prec"]

        coltura_csv = campione[col_crop].values[0] if col_crop else "Tomato"
        coltura_it = traduci_coltura(str(coltura_csv))

        # Verifica che la coltura esista nella KB; altrimenti usa fallback
        colture_in_kb = ["grano", "mais", "pomodoro", "patata", "fagiolo",
                          "cavolo", "cipolla", "aglio", "pisello", "cece",
                          "zucca", "zucchina", "melanzana", "peperone", "orzo"]
        if coltura_it not in colture_in_kb:
            coltura_it = "pomodoro"  # fallback

        classe_reale = campione["result"].values[0]

        print(f"\n{'─'*80}")
        print(f"  SCENARIO {i+1}: {coltura_it.upper()} | Mese: {mese} | "
              f"Coltura prec.: {coltura_prec} | Classe reale: {classe_reale}")
        print(f"{'─'*80}")

        # ML: predizione
        X_camp = pd.get_dummies(campione.drop("result", axis=1))
        X_camp = X_camp.reindex(columns=feature_columns, fill_value=0)
        classe_ml = int(modello_ml.predict(scaler.transform(X_camp))[0])
        bisogno_acqua = calcola_bisogno_acqua(classe_ml)

        semantica = {0: "Nessuna irrigazione", 1: "Irrigazione necessaria", 2: "Stress idrico/anomalia"}
        print(f"  [ML]  Classe predetta: {classe_ml} ({semantica[classe_ml]}) | "
              f"Bisogno acqua -> '{bisogno_acqua.upper()}'")
        corretto = "✓" if classe_ml == classe_reale else "✗"
        print(f"        Predizione {'corretta' if classe_ml == classe_reale else 'errata'} {corretto} "
              f"(classe reale: {classe_reale})")

        # BN: stima affidabilità 
        affidabilita, prob = calcola_affidabilita_meteo(
            modello_bn, df_completo, df_bn, soglie, campione, classe_ml
        )
        print(f"  [BN]  P(Target={classe_ml} | Temp, Umidità, MOI) = {prob:.3f} "
              f"-> Affidabilità meteo: '{affidabilita.upper()}'")

        # Prolog: inferenza e consigli 
        prolog.aggiorna_fatti_dinamici(bisogno_acqua, affidabilita)

        # Azione operativa principale
        azione = prolog.get_azione_operativa(coltura_it, mese, coltura_prec)
        print(f"  [Prolog] Azione: {azione}")

        # Allerta meteo (solo se rilevante)
        allerta = prolog.get_allerta_meteo(coltura_it)
        print(f"  [Prolog] Allerta: {allerta}")

        # Consiglio fertilizzazione
        fertilizzazione = prolog.get_consiglio_fertilizzazione(coltura_it)
        print(f"  [Prolog] Fertilizzazione: {fertilizzazione}")

        # Verifica rotazione
        incompatibile = prolog.verifica_rotazione(coltura_it, coltura_prec)
        if incompatibile:
            print(f"  [Prolog] ⚠ ATTENZIONE: rotazione non raccomandata "
                  f"({coltura_it} e {coltura_prec} appartengono alla stessa famiglia botanica!)")
        else:
            print(f"  [Prolog] ✓ Rotazione corretta ({coltura_it} dopo {coltura_prec})")

    # FASE 5: Advisor di Rotazione Colturale (query Prolog su KB completa)

    print("\n" + "=" * 80)
    print("  [FASE 5] ADVISOR DI ROTAZIONE COLTURALE (Ragionamento Puro su KB)")
    print("=" * 80)
    print("\n  Colture seminabili ad APRILE per tipo di terreno:")
    for terreno in ["Black Soil", "Sandy Soil", "Red Soil", "Alluvial Soil"]:
        colture_raccomandate = prolog.get_colture_raccomandate(terreno)
        colture_aprile = prolog.get_compatibili_per_mese("aprile")
        intersez = [c for c in colture_raccomandate if c in colture_aprile]
        if intersez:
            print(f"  {terreno:<15}: {', '.join(intersez)}")

    print("\n  Alternativa di rotazione dopo pomodoro (no solanacee) a maggio:")
    alternative = prolog.get_compatibili_per_mese("maggio", escludere_famiglia="solanacee")
    print(f"  Colture compatibili: {', '.join(alternative) if alternative else 'nessuna'}")

    print("\n" + "=" * 80)
    print("  Pipeline completata.")
    print("=" * 80)


if __name__ == "__main__":
    esegui_pipeline()
