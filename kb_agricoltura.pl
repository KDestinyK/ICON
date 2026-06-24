
% kb_agricoltura.pl - Base di Conoscenza per l'Agricoltura


% Ontologia con 16 colture organizzate in 7 famiglie botaniche.
% Copre: seminalita' stagionale, rotazione colturale,
% tolleranza idrica, fertilizzazione, allerte climatiche,
% ottimizzazione del tipo di terreno.
%
% Fatti DINAMICI (iniettati dalla pipeline Python a runtime):
%   bisogno_acqua/1       - si | no        (da ML)
%   affidabilita_meteo/1  - alta | bassa   (da Rete Bayesiana)
%
% Questi due fatti consentono alle regole statiche di ragionare
% su evidenze calcolate a runtime dall'apprendimento automatico.


:- dynamic bisogno_acqua/1.
:- dynamic affidabilita_meteo/1.


% ONTOLOGIA: FAMIGLIE E COLTURE


famiglia(solanacee).
famiglia(graminacee).
famiglia(leguminose).
famiglia(cucurbitacee).
famiglia(liliacee).
famiglia(crucifere).
famiglia(ombrellifere).   % aggiunta per carota

coltura(pomodoro,  solanacee).
coltura(patata,    solanacee).
coltura(melanzana, solanacee).
coltura(peperone,  solanacee).

coltura(mais,      graminacee).
coltura(grano,     graminacee).
coltura(orzo,      graminacee).

coltura(fagiolo,   leguminose).
coltura(pisello,   leguminose).
coltura(cece,      leguminose).

coltura(zucchina,  cucurbitacee).
coltura(zucca,     cucurbitacee).

coltura(cipolla,   liliacee).
coltura(aglio,     liliacee).

coltura(cavolo,    crucifere).

coltura(carota,    ombrellifere).   % presente nel dataset CSV


% VINCOLI DI ROTAZIONE

% Regola: due colture della stessa famiglia non devono succedersi.
% Motivo agronomico: riduzione rischio malattie e parasiti specifici.

incompatibile(Pianta_Nuova, Pianta_Vecchia) :-
    coltura(Pianta_Nuova, Famiglia),
    coltura(Pianta_Vecchia, Famiglia),
    Pianta_Nuova \= Pianta_Vecchia.


% STAGIONALITA' (MESI DI SEMINA)


mese_semina(pomodoro,  aprile).
mese_semina(pomodoro,  maggio).
mese_semina(patata,    marzo).
mese_semina(patata,    aprile).
mese_semina(melanzana, aprile).
mese_semina(melanzana, maggio).
mese_semina(peperone,  aprile).
mese_semina(peperone,  maggio).
mese_semina(mais,      marzo).
mese_semina(mais,      aprile).
mese_semina(grano,     ottobre).
mese_semina(grano,     novembre).
mese_semina(orzo,      ottobre).
mese_semina(orzo,      novembre).
mese_semina(fagiolo,   maggio).
mese_semina(pisello,   ottobre).
mese_semina(pisello,   marzo).
mese_semina(cece,      novembre).
mese_semina(cece,      febbraio).
mese_semina(zucchina,  aprile).
mese_semina(zucchina,  maggio).
mese_semina(zucca,     aprile).
mese_semina(zucca,     maggio).
mese_semina(cipolla,   ottobre).
mese_semina(cipolla,   marzo).
mese_semina(aglio,     novembre).
mese_semina(cavolo,    settembre).
mese_semina(cavolo,    ottobre).
mese_semina(carota,    marzo).
mese_semina(carota,    aprile).
mese_semina(carota,    agosto).   % seconda semina estiva


% TOLLERANZA ALLA SICCITA' (per famiglia botanica)


tolleranza_siccita(graminacee,   alta).
tolleranza_siccita(solanacee,    bassa).
tolleranza_siccita(cucurbitacee, bassa).
tolleranza_siccita(leguminose,   media).
tolleranza_siccita(liliacee,     alta).
tolleranza_siccita(crucifere,    bassa).
tolleranza_siccita(ombrellifere, media).


% CONDIZIONE DI SEMINA


puo_seminare(Pianta, MeseAttuale, ColturaPrecedente) :-
    mese_semina(Pianta, MeseAttuale),
    \+ incompatibile(Pianta, ColturaPrecedente).


% MOTORE INFERENZIALE: AZIONE OPERATIVA
% I fatti bisogno_acqua/1 e affidabilita_meteo/1 sono iniettati
% a runtime dalla pipeline Python (ML + Rete Bayesiana).


% CASO 1: ML -> irrigazione; BN -> conferma affidabilita' alta
azione_operativa(Pianta, Mese, ColturaPrec,
    'OK Semina. Irrigazione STANDARD (ML e BN concordano).') :-
    puo_seminare(Pianta, Mese, ColturaPrec),
    bisogno_acqua(si),
    affidabilita_meteo(alta).

% CASO 2: ML -> irrigazione; BN -> bassa affidabilita'; pianta tollerante
azione_operativa(Pianta, Mese, ColturaPrec,
    'OK Semina. Irrigazione SOSPESA (anomalia meteo, ma famiglia tollerante alla siccita).') :-
    puo_seminare(Pianta, Mese, ColturaPrec),
    bisogno_acqua(si),
    affidabilita_meteo(bassa),
    coltura(Pianta, Famiglia),
    tolleranza_siccita(Famiglia, alta).

% CASO 3: ML -> irrigazione; BN -> bassa affidabilita'; pianta sensibile
azione_operativa(Pianta, Mese, ColturaPrec,
    'OK Semina. Irrigazione DI SOCCORSO (anomalia meteo su coltura sensibile, intervenire).') :-
    puo_seminare(Pianta, Mese, ColturaPrec),
    bisogno_acqua(si),
    affidabilita_meteo(bassa),
    coltura(Pianta, Famiglia),
    \+ tolleranza_siccita(Famiglia, alta).

% CASO 4: ML -> no irrigazione (terreno ok o stress idrico)
azione_operativa(Pianta, Mese, ColturaPrec,
    'OK Semina. Terreno nelle condizioni idriche corrette. Nessuna irrigazione necessaria.') :-
    puo_seminare(Pianta, Mese, ColturaPrec),
    bisogno_acqua(no).

% ERRORE BLOCCANTE: Rotazione non rispettata
azione_operativa(Pianta, _, ColturaPrec,
    'ERRORE: Rotazione non rispettata. Stessa famiglia botanica: rischio malattie.') :-
    incompatibile(Pianta, ColturaPrec).

% ERRORE BLOCCANTE: Fuori stagione
% Il guard nonvar/1 evita errori di istanziazione quando la regola
% viene raggiunta in backtracking con variabili non legate.
azione_operativa(Pianta, Mese, _,
    'ERRORE: Periodo di semina non ottimale per questa coltura in questo mese.') :-
    nonvar(Pianta), nonvar(Mese),
    \+ mese_semina(Pianta, Mese).


% FERTILIZZAZIONE (per famiglia botanica)

fabbisogno_nutrizionale(graminacee,
    'Elevato Azoto (N) nella fase di accestimento. Concimazione frazionata in 2-3 interventi.').
fabbisogno_nutrizionale(solanacee,
    'Potassio (K) e Fosforo (P) per fioritura e fruttificazione. Evitare eccessi di Azoto.').
fabbisogno_nutrizionale(leguminose,
    'Pianta azoto-fissatrice: evitare concimazioni azotate. Utile apporto di Calcio e Zolfo.').
fabbisogno_nutrizionale(cucurbitacee,
    'Letame maturo e compost ricco di microelementi per massimizzare la resa volumetrica.').
fabbisogno_nutrizionale(liliacee,
    'Compost maturo leggero. Temono i ristagni da letame fresco: evitare eccessi idrici.').
fabbisogno_nutrizionale(crucifere,
    'Elevato fabbisogno di Zolfo e Calcio per lo sviluppo cellulare e la prevenzione delle virosi.').
fabbisogno_nutrizionale(ombrellifere,
    'Terreno profondo e sciolto. Fosforo per sviluppo radicale. Evitare eccessi di Azoto.').

consiglio_fertilizzazione(Pianta, Consiglio) :-
    coltura(Pianta, Famiglia),
    fabbisogno_nutrizionale(Famiglia, Consiglio).


% ALLERTE CLIMATICHE (usano i fatti dinamici ML + BN)


% Rischio siccita': ML dice irrigazione, BN non conferma, pianta sensibile
allerta_meteo(Pianta,
    'RISCHIO SICCITA: Coltura sensibile, irrigazione di soccorso raccomandata immediatamente.') :-
    bisogno_acqua(si),
    affidabilita_meteo(bassa),
    coltura(Pianta, Famiglia),
    \+ tolleranza_siccita(Famiglia, alta).

% Rischio marciume radicale: terreno saturo su piante sensibili al ristagno
allerta_meteo(Pianta,
    'RISCHIO MARCIUME RADICALE: Terreno troppo umido. Sospendere immediatamente le irrigazioni.') :-
    bisogno_acqua(no),
    affidabilita_meteo(bassa),
    coltura(Pianta, Famiglia),
    (Famiglia = liliacee ; Famiglia = solanacee).

% Fallback silente (nessuna allerta critica)
allerta_meteo(_, 'Nessuna allerta critica severa. Controllare parametri standard.').


% OTTIMIZZAZIONE TERRENO


terreno_ideale(pomodoro,  'Black Soil').
terreno_ideale(grano,     'Alluvial Soil').
terreno_ideale(mais,      'Alluvial Soil').
terreno_ideale(patata,    'Sandy Soil').
terreno_ideale(cece,      'Red Soil').
terreno_ideale(melanzana, 'Black Soil').
terreno_ideale(peperone,  'Red Soil').
terreno_ideale(orzo,      'Alluvial Soil').
terreno_ideale(fagiolo,   'Sandy Soil').
terreno_ideale(pisello,   'Sandy Soil').
terreno_ideale(zucchina,  'Black Soil').
terreno_ideale(zucca,     'Black Soil').
terreno_ideale(cipolla,   'Alluvial Soil').
terreno_ideale(aglio,     'Alluvial Soil').
terreno_ideale(cavolo,    'Red Soil').
terreno_ideale(carota,    'Sandy Soil').

coltura_raccomandata(Terreno, Coltura) :-
    terreno_ideale(Coltura, Terreno).
