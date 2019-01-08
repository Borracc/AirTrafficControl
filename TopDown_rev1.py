#!/usr/bin/env python
# coding: utf-8
# ## Calcolo adiacenze con grafo con approccio top-down
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat #polygon
import networkx as nx
import datetime

# ## Valori "GLOBALI"
DATA = pd.Timestamp(2016, 1, 7) # 7 gennaio 2016
#DATA = pd.Timestamp(2016, 7, 21) # 21 luglio 2016
ORA = datetime.time(2, 0) # 2:00
ORA_INIZIO = datetime.time(11, 0) # 11:00
ORA_FINE = datetime.time(12, 0) # 12:00
COUNTER_SOGLIA=1
INTORNO=100

# Valore da modificare in base alla nazione
nation = "Italy"
#nation = "Germany"
#nation = "Espana"
#nation = "France"
#nation = "All"
#nation = "Iceland"

# ## Funzioni Utili ----------------------------------------------
# Disegnare un grafo data la matrice di adiacenza *ma*
def disegnaGrafo(ma):
    rows, cols = np.where(ma == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=300, node_color='yellow', with_labels=True)
    plt.show()

# Salva dataframe CSV
def salvaDFcsv(df,file_name):
    df.to_csv(file_name, encoding='utf-8', index=False)
# ## ------------------------------------------------------------------

# ## lettura dati:
# Lettura Airblocks
def coordInt(valore):
    if valore[0]=='S' or valore[0]=='W':
        return int("-"+valore[1:8])
    else:
        return int(valore[1:8])

airblock = pd.read_excel("ENV_Original_"+ nation +".xlsx", sheet_name="Airblock" ) #Italy
airblock.columns=["ID", "Latitude", "Longitude"]

idBlocks = []
coordinate = []
j = -1
for i in range(0,len(airblock)):
    if str(airblock.ID[i]) != "nan":
        idBlocks.append(str(airblock.ID[i]))
        coordinate.append([])
        j = j + 1
    else :
        coordinate[j].append((coordInt(str(airblock.Latitude[i])), coordInt(str(airblock.Longitude[i]))))

vertici_AB = [] # formato i-esimo = (id_AB, lista_vertici)
for i in range(0, len(idBlocks)):
    vertici_AB.append( (idBlocks[i], coordinate[i]) )


# Lettura Elementary Sector
sector = pd.read_excel("ENV_Original_"+ nation +".xlsx", sheet_name="Sector" )
sector.columns=["Sector_ID", "Sector_Name", "Sector_Type", "Airblock_ID", "MinAltitude", "MaxAltitude", "Positive_Negative"]

idSector = []
secBlocks = []
j = -1
for i in range(0,len(sector)):
    if str(sector.Sector_ID[i]) != "nan":
        idSector.append(str(sector.Sector_ID[i]))
        secBlocks.append([])
        j = j + 1
    else :
        secBlocks[j].append((str(sector.Airblock_ID[i]), sector.MinAltitude[i], sector.MaxAltitude[i]))

blocchi_El = [] # formato i-esimo = (id_El, lista_blocchi)
for i in range(0, len(idSector)):
    blocchi_El.append( (idSector[i], secBlocks[i]) )


# Lettura Collapse Sector
collapse = pd.read_excel("ENV_Original_"+ nation +".xlsx", sheet_name="Airspace" )#Germany
collapse.columns=["ACC_CS","Name","Sectors","Type","NB_Items"]

idCollapse = []
colSectors = []
willMemo = False
j = -1
for i in range(0,len(collapse)):
    if ((str(collapse.ACC_CS[i]) != "nan") and (str(collapse.Type[i]) == "CS")):
        idCollapse.append(str(collapse.ACC_CS[i]))
        colSectors.append([])
        j = j + 1
        willMemo = True
    elif ((str(collapse.ACC_CS[i]) != "nan") and (str(collapse.Type[i]) != "CS")):
        willMemo = False
    else:
        if(willMemo):
            colSectors[j].append(str(collapse.Sectors[i]))

elementary_Co = [] # formato i-esimo = (id_Co, lista_elementary)
for i in range(0, len(idCollapse)):
    elementary_Co.append( (idCollapse[i], colSectors[i]) )

# Lettura Configuration
configuration = pd.read_excel("ENV_Original_"+ nation +".xlsx", sheet_name="Configuration" ) #Italy

idConf = []
conf_sectors = []

j = -1
for i in range(0,len(configuration)):
    if str(configuration.ACC[i]) != "nan":
        aacccc = str(configuration.ACC[i])
    elif str(configuration.Configuration[i]) != "nan":
        idConf.append((aacccc,str(configuration.Configuration[i])))
        conf_sectors.append([])
        j = j + 1
    elif str(configuration.Sectors[i]) != "nan":
        conf_sectors[j].append(str(configuration.Sectors[i]))

collapse_Conf = [] # formato i-esimo = (id_Conf, lista_collapse)
for i in range(0,len(idConf)):
    collapse_Conf.append((idConf[i],conf_sectors[i]))   


# Lettura Opening Scheme
openScheme = pd.read_excel("ENV_Original_"+ nation +".xlsx", sheet_name="OpeningScheme" )
openScheme.columns=["ACC", "Name", "Data", "Label", "Configuration", "Time_Start", "Time_End"]

j = -1
idAcc = []
dates_acc = []
config_date_acc = []
for i in range(0,len(openScheme)):
    if str(openScheme.ACC[i]) != "nan":
        idAcc.append(str(openScheme.ACC[i]))
        dates_acc.append([])
        config_date_acc.append([])
        j = j + 1
        k = -1
    elif str(openScheme.Data[i]) != "NaT":
        dates_acc[j].append(openScheme.Data[i])
        config_date_acc[j].append([])
        k = k + 1
    else:
        config_date_acc[j][k].append((openScheme.Configuration[i], openScheme.Time_Start[i], openScheme.Time_End[i]))

configurazioni_OS = [] # formato i-esimo = (id_acc , data, lista_configurazioni in base all'orario )
for i in range(0,len(idAcc)):
    perdata=[]
    for j in range(0, len(dates_acc[i])):
        perdata.append( (dates_acc[i][j], config_date_acc[i][j]))
    configurazioni_OS.append( (idAcc[i], perdata) )


# ## strutture che memorizzano i dati:
# vertici_AB
# blocchi_El
# elementary_Co
# collapse_Conf
# configurazioni_OS
# ## -----------------------------------------------

# Funzioni per il reperimento# In[11]:
#ritorna lista dei vertici di un Airblock dato
def getVertici(ab):
    for i in range(0, len(vertici_AB)):
        if(ab == vertici_AB[i][0]):
            return vertici_AB[i][1]
    return []

#ritorna lista dei blocchi di un Elementary dato
def getAirblocks(elem):
    for i in range(0, len(blocchi_El)):
        if(elem == blocchi_El[i][0]):
            return blocchi_El[i][1]
    return []

#ritorna lista degli Elementary di un Collapse dato
def getElementary(coll):
    for i in range(0, len(elementary_Co)):
        if(coll == elementary_Co[i][0]):
            return elementary_Co[i][1]
    return []

#ritorna lista dei Collapse di una configurazione data (acc, nome)
def getCollapse(conf):
    for i in range(0, len(collapse_Conf)):
        if(conf[0] == collapse_Conf[i][0][0] and conf[1] == collapse_Conf[i][0][1]):
            return collapse_Conf[i][1]
    return []

#ritorna l'Opening Scheme delle configuarazioni acc dato
def getConfigurazioni(os):
    for i in range(0, len(configurazioni_OS)):
        if(os == configurazioni_OS[i][0]):
            return configurazioni_OS[i][1]
    return []
# ## -----------------------------------------------

# Reperire *entita'* attive
def configurazioniAttive():
    esito = []
    for i in range(0,len(configurazioni_OS)):
        if(configurazioni_OS[i][1] != []):
            if(configurazioni_OS[i][1][0][0] == DATA):
                for j in range(0, len(configurazioni_OS[i][1][0][1])):
                    start = configurazioni_OS[i][1][0][1][j][1]
                    end = configurazioni_OS[i][1][0][1][j][2]
                    if((ORA_FINE <= end and ORA_FINE >= start ) or (ORA_INIZIO <= end and ORA_INIZIO >= start )):
                        esito.append( (configurazioni_OS[i][0] , configurazioni_OS[i][1][0][1][j][0]) )
    return esito
#configurazioniAttive()

def collapseAttivi():
    res = configurazioniAttive()
    esito = []
    for i in range(0, len(res)):
        esito = esito + getCollapse(res[i])
    return esito
#collapseAttivi()

def elementaryAttivi():
    res = collapseAttivi()
    esito = []
    for i in range(0, len(res)):
        esito = esito + getElementary(res[i])
    return esito
#elementaryAttivi()

def sectorAttivi():
    res_Co = collapseAttivi()
    res_El = elementaryAttivi()
    res_Sec = res_Co + res_El
    return res_Sec
#sectorAttivi()

def airblocksAttivi():
    res = sectorAttivi()
    esito = []
    for i in range(0, len(res)):
        elementi = getElementary(res[i])
        if (len(elementi)<1):
            esito = esito + getAirblocks(res[i])
        else:
            for j in range(0, len(elementi)):
                esito = esito + getAirblocks(elementi[j])
    return esito
#airblocksAttivi()

# ## Calcolo Adiacenze
# adiacenze airblocks

def adiacenza2D(vert_i, vert_j):
    for k in range(0, (len(vert_i))):
        poly = pat.Polygon(xy = np.array(vert_j))  
        counter = 0
        if(poly.contains_point(point=vert_i[k])):
            counter = counter + 1
            if(counter > COUNTER_SOGLIA):
                return True
        if(poly.contains_point(point=(vert_i[k][0]+INTORNO,vert_i[k][1]))):
            counter = counter + 1
            if(counter > COUNTER_SOGLIA):
                return True
        if(poly.contains_point(point=(vert_i[k][0]-INTORNO,vert_i[k][1]))):
            counter = counter + 1
            if(counter > COUNTER_SOGLIA):
                return True
        if(poly.contains_point(point=(vert_i[k][0],vert_i[k][1]+INTORNO))):
            counter = counter + 1
            if(counter > COUNTER_SOGLIA):
                return True
        if(poly.contains_point(point=(vert_i[k][0],vert_i[k][1]-INTORNO))):
            counter = counter + 1
            if(counter > COUNTER_SOGLIA):
                return True
        if(poly.contains_point(point=(vert_i[k][0]+INTORNO,vert_i[k][1]+INTORNO))):
            counter = counter + 1
            if(counter > COUNTER_SOGLIA):
                return True
        if(poly.contains_point(point=(vert_i[k][0]-INTORNO,vert_i[k][1]+INTORNO))):
            counter = counter + 1
            if(counter > COUNTER_SOGLIA):
                return True
        if(poly.contains_point(point=(vert_i[k][0]-INTORNO,vert_i[k][1]-INTORNO))):
            counter = counter + 1
            if(counter > COUNTER_SOGLIA):
                return True
        if(poly.contains_point(point=(vert_i[k][0]+INTORNO,vert_i[k][1]-INTORNO))):
            counter = counter + 1
            if(counter > COUNTER_SOGLIA):
                return True
    return False

def unici(lista):
    nuovaLista = []
    for i in range(0,len(lista)):
        if(lista[i] not in nuovaLista):
            nuovaLista.append(lista[i])
    return nuovaLista

def unici_AB(lista):
    nuovaLista = []
    nomi = []
    for i in range(0,len(lista)):
        if(lista[i][0] not in nomi):
            nomi.append(lista[i][0])
            nuovaLista.append(lista[i])
    return nuovaLista

def getNomi_AB(lista):
    nomi = []
    for i in range(0,len(lista)):
        nomi.append(lista[i][0])
    return nomi

def adiacenza_AB():
    abAtt = unici_AB(airblocksAttivi())
    n = len(abAtt)
    nomi_AB = getNomi_AB(abAtt)
    ma = [ [0] * n for _ in range(n)]
    for i in range(0, n):
        for j in range(i, n):
            if(i == j):
                ma[i][j]=1
            elif(adiacenza2D(getVertici(abAtt[i][0]),getVertici(abAtt[j][0]))):
                ma[i][j]=1
                ma[j][i]=1
    matrixAd = np.asmatrix(ma)
    dfma = pd.DataFrame(matrixAd, index = nomi_AB, columns = nomi_AB)
    return dfma

dfma_AB = adiacenza_AB()
#dfma_AB.head()
matrix_AB = np.asmatrix(dfma_AB)

salvaDFcsv(dfma_AB,"matAdBlocks_"+nation+"_21072016-11-12.csv")

dfma_AB = pd.read_csv("matAdBlocks_"+ nation +"_21072016-11-12.csv")
dfma_AB.index = dfma_AB.columns
#dfma_AB.head()

# adiacenze settori
def adiacenza3D(i_ab,j_ab):
    for k in range(0, len(i_ab)):
        for l in range(0, len(j_ab)):
            if( (dfma_AB.loc[i_ab[k][0]][j_ab[l][0]] == 1) and (i_ab[k][2] >= j_ab[l][1]) and (i_ab[k][1] <= j_ab[l][2]) ):
                return True
    return False

def adiacenza_Lista(elem, lista):
    for m in range(0,len(lista)):
        if(adiacenza3D(getAirblocks(elem),getAirblocks(lista[m]))):
                return True
    return False

def adiacenza_Sec():
    secAtt = unici(sectorAttivi())
    n = len(secAtt)
    ma = [ [0] * n for _ in range(n)]
    
    for i in range(0, n):
        for j in range(i, n):
            if(i == j):
                ma[i][j]=1
            else:
                sec_Att_i = getElementary(secAtt[i])
                sec_Att_j = getElementary(secAtt[j])
                if( (len(sec_Att_i)>0) and (len(sec_Att_j)>0) ):# confronto adiacenza fra collapse
                    for k in range(0, len(sec_Att_i)):
                        if(adiacenza_Lista(sec_Att_i[k],sec_Att_j)):
                            ma[i][j]=1
                            ma[j][i]=1
                            break           
                elif((len(sec_Att_i)>0)):
                    if(adiacenza_Lista(secAtt[j],sec_Att_i)):
                        ma[i][j]=1
                        ma[j][i]=1
                elif((len(sec_Att_j)>0)):
                    if(adiacenza_Lista(secAtt[i],sec_Att_j)):
                        ma[i][j]=1
                        ma[j][i]=1
                else:
                    if(adiacenza3D(getAirblocks(secAtt[i]),getAirblocks(secAtt[j]))):
                        ma[i][j]=1
                        ma[j][i]=1
    matrixAd = np.asmatrix(ma)
    dfma = pd.DataFrame(matrixAd, index = secAtt, columns = secAtt)
    return dfma

dfma_Sec = adiacenza_Sec()
#dfma_Sec.head()

matrix_Sec = np.asmatrix(dfma_Sec)

salvaDFcsv(dfma_Sec,"matAdSector_"+nation+"_21072016-11-12.csv")

dfma_Sec = pd.read_csv("matAdSector_"+ nation +"_21072016-11-12.csv")
dfma_Sec.index = dfma_Sec.columns
dfma_Sec

disegnaGrafo(matrix_Sec)

# Grafo AirBlocks
disegnaGrafo(matrix_AB)

# Adiacenza configurazioni
def adiacenza_ListeSec(co_i, co_j):
    for k in range(0, (len(co_i))):
        for l in range(0, (len(co_j))):
            if(dfma_Sec.loc[co_i[k]][co_j[l]] == 1):
                return True
    return False

def adiacenza_Conf():
    confAtt = configurazioniAttive()
    n = len(confAtt)
    ma = [ [0] * n for _ in range(n)]
    
    for i in range(0, n):
        for j in range(i, n):
            if(i == j):
                ma[i][j]=1
            elif(adiacenza_ListeSec(getCollapse(confAtt[i]),getCollapse(confAtt[j]))):
                ma[i][j]=1
                ma[j][i]=1

    matrixAd = np.asmatrix(ma)
    dfma = pd.DataFrame(matrixAd, index = confAtt, columns = confAtt)
    return dfma

dfma_Conf = adiacenza_Conf()
dfma_Conf.head()

matrix_Conf = np.asmatrix(dfma_Conf)

salvaDFcsv(dfma_Conf,"matAdConfig_"+nation+"_21072016-11-12.csv")

dfma_Conf = pd.read_csv("matAdConfig_"+ nation +"_21072016-11-12.csv")
dfma_Conf.index = dfma_Conf.columns
dfma_Conf.head()

matrix_Conf = np.asmatrix(dfma_Conf)
# Grafo configurazioni
disegnaGrafo(matrix_Conf)