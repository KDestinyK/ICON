"""
modulo_prolog.py - Interfaccia con il Motore Inferenziale Prolog
================================================================
Gestisce il ciclo di vita dell'istanza Prolog (caricamento KB,
asserzione/retrazione di fatti dinamici, interrogazione del motore).

Fatti dinamici iniettati dalla pipeline:
    bisogno_acqua/1     : si | no       (output classificatore ML)
    affidabilita_meteo/1: alta | bassa  (output inferenza Bayesiana)

Queste due variabili dinamiche consentono alle regole statiche della KB
di ragionare su evidenze calcolate a runtime dai moduli ML e Bayesiano.
"""

import os
from pyswip import Prolog


class ModuloProlog:
    """
    Wrapper attorno all'istanza Prolog.
    Espone metodi ad alto livello per la pipeline principale.
    """

    def __init__(self, percorso_kb: str = "kb_agricoltura.pl"):
        self.prolog = Prolog()
        kb_abs = os.path.abspath(percorso_kb)
        if not os.path.exists(kb_abs):
            raise FileNotFoundError(f"KB non trovata: {kb_abs}")
        self.prolog.consult(kb_abs)
        print(f"   [Prolog] KB caricata: {kb_abs}")

   
    # Fatti dinamici

    def aggiorna_fatti_dinamici(self, bisogno_acqua: str, affidabilita_meteo: str):
        """
        Ritrae i fatti dinamici precedenti e asserisce quelli nuovi.

        Parametri
        ---------
        bisogno_acqua : 'si' | 'no'
            Derivato dall'output del classificatore ML (classe 1 -> 'si')
        affidabilita_meteo : 'alta' | 'bassa'
            Derivato dalla probabilità Bayesiana sulla classe predetta
        """
        list(self.prolog.query("retractall(bisogno_acqua(_))"))
        list(self.prolog.query("retractall(affidabilita_meteo(_))"))
        self.prolog.assertz(f"bisogno_acqua({bisogno_acqua})")
        self.prolog.assertz(f"affidabilita_meteo({affidabilita_meteo})")
    
    
    # Query principali

    def get_azione_operativa(self, coltura: str, mese: str, coltura_prec: str) -> str:
        """
        Interroga la KB per l'azione agriologica raccomandata dato il contesto.
        Restituisce la stringa descrittiva del consiglio operativo.
        """
        query = f"azione_operativa({coltura}, {mese}, {coltura_prec}, Output)"
        risultati = list(self.prolog.query(query))
        if risultati:
            return str(risultati[0]["Output"])
        return "(nessuna azione applicabile per il contesto fornito)"

    def get_consiglio_fertilizzazione(self, coltura: str) -> str:
        """Restituisce il consiglio nutrizionale per la famiglia botanica della coltura."""
        query = f"consiglio_fertilizzazione({coltura}, Consiglio)"
        risultati = list(self.prolog.query(query))
        if risultati:
            return str(risultati[0]["Consiglio"])
        return "(nessun dato di fertilizzazione disponibile)"

    def get_allerta_meteo(self, coltura: str) -> str:
        """
        Interroga la KB per eventuali allerte climatiche.
        Restituisce la prima allerta rilevante (escludendo il fallback generico).
        """
        query = f"allerta_meteo({coltura}, Alert)"
        risultati = list(self.prolog.query(query))
        for r in risultati:
            alert = str(r["Alert"])
            if "Nessuna allerta" not in alert:
                return alert
        return "Nessuna allerta critica severa. Controllare parametri standard."

    def get_colture_raccomandate(self, tipo_terreno: str) -> list:
        """Restituisce le colture ideali per un dato tipo di terreno."""
        query = f"coltura_raccomandata('{tipo_terreno}', Coltura)"
        return [str(r["Coltura"]) for r in self.prolog.query(query)]

    def verifica_rotazione(self, coltura_nuova: str, coltura_precedente: str) -> bool:
        """Ritorna True se la rotazione è incompatibile (stessa famiglia botanica)."""
        query = f"incompatibile({coltura_nuova}, {coltura_precedente})"
        return len(list(self.prolog.query(query))) > 0

    def get_compatibili_per_mese(self, mese: str, escludere_famiglia: str = None) -> list:
        """
        Restituisce tutte le colture seminabili nel mese dato.
        Se escludere_famiglia è specificata, filtra le colture della stessa famiglia
        (utile per suggerire la rotazione corretta).
        """
        query = f"mese_semina(Coltura, {mese})"
        colture = [str(r["Coltura"]) for r in self.prolog.query(query)]
        if escludere_famiglia:
            filtrate = []
            for c in colture:
                q_fam = f"coltura({c}, Famiglia)"
                res = list(self.prolog.query(q_fam))
                if res and str(res[0]["Famiglia"]) != escludere_famiglia:
                    filtrate.append(c)
            return filtrate
        return colture
