#Configurazioni.py
## Configurazioni *parametrico con nome della Nazione*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat #polygon
import networkx as nx
import datetime

## Valore da modificare in base alla nazione

#nation = "Italy"
nation = "Germany"

## Funzioni Utili ----------------------------------------------
# Disegnare un grafo data la matrice di adiacenza ma
def disegnaGrafo(ma):
    rows, cols = np.where(ma == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=300, node_color='yellow', with_labels=True)
    plt.show()
## ------------------------------------------------------------------
# Leggo i dati relativi alle configurazioni e agli opening time
configuration = pd.read_excel("ENV_Original_"+ nation +".xlsx", sheet_name="Configuration" ) #Italy
configuration.head()

idACC = []
acc_confs = []
idConf = []
conf_sectors = []

k = -1
j = -1
for i in range(0,len(configuration)):
    if str(configuration.ACC[i]) != "nan":
        aacccc = str(configuration.ACC[i])
        idACC.append(aacccc)
        acc_confs.append([])
        k = k + 1
    elif str(configuration.Configuration[i]) != "nan":
        idConf.append((aacccc,str(configuration.Configuration[i])))
        acc_confs[k].append(str(configuration.Configuration[i]))
        conf_sectors.append([])
        j = j + 1
    elif str(configuration.Sectors[i]) != "nan":
        conf_sectors[j].append(str(configuration.Sectors[i]))

acc = []
for i in range(0,len(idACC)):
    acc.append((idACC[i],acc_confs[i]))

configurazioni = []
for i in range(0,len(idConf)):
    configurazioni.append((idConf[i],conf_sectors[i]))

# *acc* e' una lista di tuple del tipo (*id_ACC*, *lista_delle_configurazioni_ACC*)
# *configurazioni* e' una lista di tuple del tipo ((*id_ACC*, *id_Conf*), *lista_delli_settori_della_cofigurazione*)

# Lettura Opening Scheme:
openScheme = pd.read_excel("ENV_Original_"+ nation +".xlsx", sheet_name="OpeningScheme" ) #Italy
openScheme.columns=["ACC", "Name", "Data", "Label", "Configuration", "Time_Start", "Time_End"]
openScheme.head()

## Date id_ACC, data e ora di interesse ritorna la corrispondente configurazione attiva
def confAccDataOra(acc,data, ora):
    right_ACC = False
    right_Data = False
    for i in range(0,len(openScheme)):
        if str(openScheme.ACC[i]) != "nan":
            if str(openScheme.ACC[i]) == acc:
                right_ACC = True
            else:
                right_ACC = False
        else:
            if right_ACC:
                if str(openScheme.Data[i]) != "NaT":
                    if openScheme.Data[i] == data:
                        right_Data = True
                    else:
                        right_Data = False
                else:
                    if right_Data:
                        if (ora > openScheme.Time_Start[i]) and (ora < openScheme.Time_End[i]):
                            return str(openScheme.Configuration[i])
    return "NON TROVATA!"

#configurazioni attive in una certa data e ora
def confDataOra(data,ora):
    esito = []
    for i in range(0,len(idACC)):
        esito.append((idACC[i], (confAccDataOra(idACC[i], data, ora))))
    return esito

confDataOra(pd.Timestamp(2016, 1, 7), datetime.time(2, 0) )

def getSettori(acc, conf):
    for i in range(0, len(configurazioni)):
        if configurazioni[i][0][0] == acc and configurazioni[i][0][1] == conf:
            return configurazioni[i][1]
    return []

def settoriAttvi(data,ora):
    cc = confDataOra(data, ora)
    esito = []
    for i in range(0,len(cc)):
        esito = esito + getSettori(cc[i][0],cc[i][1])
    return esito

res = settoriAttvi(pd.Timestamp(2016, 1, 7), datetime.time(2, 0))
res

dfST = pd.read_csv("maST_"+ nation +".csv")
dfST.index = dfST.columns
dfST.head()

cccc = dfST.columns
cccc = cccc.drop(res)
adiacenzeSettoriAttivi = dfST.drop(cccc)
adiacenzeSettoriAttivi = adiacenzeSettoriAttivi.drop(cccc, axis = 1)
adiacenzeSettoriAttivi

matrixASA = np.matrix(adiacenzeSettoriAttivi.as_matrix())

disegnaGrafo(matrixASA)