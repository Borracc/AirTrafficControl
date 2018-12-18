#Adiacenze.py
## Creazione di grafo dei settori/blocchi *parametrico con nome della Nazione*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat #polygon
import networkx as nx

## Valore da modificare o commentare in base alla nazione

#nation = "Italy"
nation = "Germany"

## Funzioni Utili ----------------------------------------------
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
## -------------------------------------------

# Lettura del file excel/csv esportato da NEST
airblock = pd.read_excel("ENV_Original_"+ nation +".xlsx", sheet_name="Airblock" ) #Italy
airblock.columns=["ID", "Latitude", "Longitude"]
airblock.head()

# I blocchi sono memorizzati attraversole coordinate geografiche (lat, lon) dei loro vertici
#--> il dataset ha un formato scomodo per essere elaborato
#--> vorrei ottenere una lista di vertici per ogni blocco
# ## N.B. :
# Latitude NORD --> Positive, SUD --> Negative, Longitude EST --> Positive, OVEST --> Negative
def coordInt(valore):
    if valore[0]=='S' or valore[0]=='W':
        return int("-"+valore[1:8])
    else:
        return int(valore[1:8])

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

def cmpCoord(cA, cB):
    return (cA[0] == cB[0]) and (cA[1] == cB[1])


# ## L'adiacenza: un blocco è adiacente ad un altro se ha almeno un vertice che è contenuto nell'altro,
# dove per vertice intenderemo un suo intorno "largo" formato da 8 punti ad una certa distanza
# ## Attenzione:
# i punti del 'bordo inferiore' del poligono non vengono ritenuti interni al poligono a causa della funzione contains_point..
#in alcuni casi può sbgliare, ma considerando gli 8 punti dovrebbe succedere molto raramente

# ## Calcolo delle adiacenze:
# adiacenza calcola se il blocco i è adiacente al blocco j
# rispetto la precedente funzione dovrò verificare se almeno un vertice del blocco i-esimo può essere contenuto nel blocco j,
#perchè questo sia vero almeno 2 punti su 8 del suo intorno dovranno appartenervi

sogliaCounter=1
intorno=100

def adiacenza(i,j):
    for k in range(0, (len(coordinate[i]))):
        poly = pat.Polygon(xy = np.array(coordinate[j]))  
        counter = 0
        if(poly.contains_point(point=coordinate[i][k])):
            counter = counter + 1
            if(counter > sogliaCounter):
                return True
        if(poly.contains_point(point=(coordinate[i][k][0]+intorno,coordinate[i][k][1]))):
            counter = counter + 1
            if(counter > sogliaCounter):
                return True
        if(poly.contains_point(point=(coordinate[i][k][0]-intorno,coordinate[i][k][1]))):
            counter = counter + 1
            if(counter > sogliaCounter):
                return True
        if(poly.contains_point(point=(coordinate[i][k][0],coordinate[i][k][1]+intorno))):
            counter = counter + 1
            if(counter > sogliaCounter):
                return True
        if(poly.contains_point(point=(coordinate[i][k][0],coordinate[i][k][1]-intorno))):
            counter = counter + 1
            if(counter > sogliaCounter):
                return True
        if(poly.contains_point(point=(coordinate[i][k][0]+intorno,coordinate[i][k][1]+intorno))):
            counter = counter + 1
            if(counter > sogliaCounter):
                return True
        if(poly.contains_point(point=(coordinate[i][k][0]-intorno,coordinate[i][k][1]+intorno))):
            counter = counter + 1
            if(counter > sogliaCounter):
                return True
        if(poly.contains_point(point=(coordinate[i][k][0]-intorno,coordinate[i][k][1]-intorno))):
            counter = counter + 1
            if(counter > sogliaCounter):
                return True
        if(poly.contains_point(point=(coordinate[i][k][0]+intorno,coordinate[i][k][1]-intorno))):
            counter = counter + 1
            if(counter > sogliaCounter):
                return True
    return False

n = len(idBlocks)
mAd = [ [0] * n for _ in range(n)]

for i in range(0, len(coordinate)):#iterazioni su blocco
    for j in range(0, len(coordinate)):
        if i != j:
            if adiacenza(i,j):
                mAd[i][j] = 1
                mAd[j][i] = 1

mAdd = np.asmatrix(mAd)
dfma = pd.DataFrame(mAdd, index = idBlocks, columns = idBlocks)

## Grafo Airblocks
disegnaGrafo(mAdd)

salvaDFcsv(dfma,"matAdBlocks_"+ nation +".csv")

dfMAdBlocks = pd.read_csv("matAdBlocks_"+ nation +".csv")
dfMAdBlocks.index = dfMAdBlocks.columns
dfMAdBlocks.head()

## Leggo Settori:
# Per ogni settore devo crere una lista di blocchi che lo compongono quindi stabilire una matrice delle adiacenze
# dei settori in base alla matrice delle adiacenze dei blocchi

sector = pd.read_excel("ENV_Original_"+ nation +".xlsx", sheet_name="Sector" )
sector.columns=["Sector_ID", "Sector_Name", "Sector_Type", "Airblock_ID", "MinAltitude", "MaxAltitude", "Positive_Negative"]
sector.head()

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

def adiacenzaSectors(i,j):
    for k in range(0, (len(secBlocks[i]))):
        for l in range(0, (len(secBlocks[j]))):
            if((dfMAdBlocks.loc[secBlocks[i][k][0]][secBlocks[j][l][0]] == 1)
               and ((secBlocks[i][k][2] >= secBlocks[j][l][1]) and (secBlocks[i][k][1] <= secBlocks[j][l][2]))):
                return True       
    return False

n = len(idSector)
mAdS = [ [0] * n for _ in range(n)]

for i in range(0, len(mAdS)):#iterazioni sui blocchi
    for j in range((i+1), len(mAdS)):
        if adiacenzaSectors(i,j):
            mAdS[i][j] = mAdS[i][j] + 1
            mAdS[j][i] = mAdS[j][i] + 1

matrixSector = np.asmatrix(mAdS)
dfmaSector = pd.DataFrame(matrixSector, index = idSector, columns = idSector)
dfmaSector.head()

salvaDFcsv(dfmaSector,"matAdSectors_"+ nation +".csv")

dfMASector = pd.read_csv("matAdSectors_"+ nation +".csv")
dfMASector.index = dfMASector.columns
dfMASector.head()

## Grafo Settori
disegnaGrafo(matrixSector)

n = len(secBlocks)
sovra = [ [0] * n for _ in range(n) ]
sovraPerc = [ [0.0] * n for _ in range(n) ] 

for i in range(0,len(secBlocks)):
    for j in range(0, len(secBlocks)):
        for k in range(0, len(secBlocks[i])):
            for l in range(0, len(secBlocks[j])):
                if ((secBlocks[i][k][0] == secBlocks[j][l][0])
                    and ((secBlocks[i][k][2] > secBlocks[j][l][1]) and (secBlocks[i][k][1] < secBlocks[j][l][2]))):
                    sovra[i][j] = sovra[i][j] + 1
                    sovraPerc[i][j] = float(sovra[i][j])*100/(len(secBlocks[i]))


mSovra = np.asmatrix(sovra)
dfSovra = pd.DataFrame(mSovra, index = idSector, columns = idSector)
dfSovra.head()

mSovraPerc = np.asmatrix(sovraPerc)
dfSovraPerc = pd.DataFrame(mSovraPerc, index = idSector, columns = idSector)
dfSovraPerc.head()

for i in range(0,len(sovraPerc)):
    for j in range(0, len(sovraPerc)):
        if(i!=j) and (sovraPerc[i][j] >= 1.0):
            print( " --> " + idSector[i] + " - " + idSector[j] + " % = " + str (sovraPerc[i][j]) )

# ## Lettura collapse
# ... che sono memorizzati nel file excel nel foglio airspace con tipo CS

collapse = pd.read_excel("ENV_Original_"+ nation +".xlsx", sheet_name="Airspace" )#Germany
collapse.columns=["ACC_CS","Name","Sectors","Type","NB_Items"]
collapse.head()

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

# ## Calcolo adiacenza dei collapse
# leggo matrice di adiacenza dei settori;
# Un collapse e' adiacente ad un'altro se ha almeno un settore adiacente ad un settore dell'altro

# Lettura adiacenze settori
dfMASector = pd.read_csv("matAdSectors_"+ nation +".csv")
dfMASector.index = dfMASector.columns
dfMASector.head()

# Calcolo adiacenze collapse
def adiacenzaCollapse(i,j):
    for k in range(0, (len(colSectors[i]))):
        for l in range(0, (len(colSectors[j]))):
            if(dfMASector.loc[colSectors[i][k]][colSectors[j][l]] == 1):
                return True
    return False

n = len(idCollapse)
mAdC = [ [0] * n for _ in range(n)]

for i in range(0, len(mAdC)):#iterazioni sui collapse
    for j in range((i+1), len(mAdC)):
        if adiacenzaCollapse(i,j):
            mAdC[i][j] = mAdC[i][j] + 1
            mAdC[j][i] = mAdC[j][i] + 1

matrixCollapse = np.asmatrix(mAdC)
dfmaCollapse = pd.DataFrame(matrixCollapse, index = idCollapse, columns = idCollapse)
dfmaCollapse.head()

salvaDFcsv(dfmaCollapse,"matAdCollapse_"+ nation +".csv")

dfmaCollapse = pd.read_csv("matAdCollapse_"+ nation +".csv")
dfmaCollapse.index = dfmaCollapse.columns
dfmaCollapse.head()

# ## Grafo Collapse
disegnaGrafo(matrixCollapse)

## Unione dei settori: Unione degli ELEMENTARY Sector con i COLLAPSED Sector:
# Unifico gli elementy con i collapsed formattandoli tutti allo stesso modo, come se fossero tutti dei collapsed:
# una lista di id e una lista di liste di elementary che che formano il collapsed,
# in quest'ultimo caso gli elementary sector saranno composti da una lista di un'unico elementary che corrisponde esattamente a se' stesso.

idST = idCollapse + idSector

compST = colSectors
for i in range(0, len(idSector)):
    compST.append([idSector[i]])

# Calcolo la frequenza dei 'settori totali'

def adiacenzaST(i,j):
    for k in range(0, (len(compST[i]))):
        for l in range(0, (len(compST[j]))):
            if(dfMASector.loc[compST[i][k]][compST[j][l]] == 1):
                return True
    return False

n = len(idST)
maST = [ [0] * n for _ in range(n)]

for i in range(0, len(maST)):#iterazioni sui collapse
    for j in range((i+1), len(maST)):
        if adiacenzaST(i,j):
            maST[i][j] = maST[i][j] + 1
            maST[j][i] = maST[j][i] + 1

matrixST = np.asmatrix(maST)
dfST = pd.DataFrame(matrixST, index = idST, columns = idST)
dfST.head()

salvaDFcsv(dfST,"maST_"+ nation +".csv")

dfST = pd.read_csv("maST_"+ nation +".csv")
dfST.index = dfST.columns
dfST.head()

## Grafo ST ( Sector Totali )
disegnaGrafo(matrixST)

# ## Calcolo delle sovraposizioni
# remind: siccome gli elementary sector non si sovrappongono diremmo che due settori (elem + coll)
#si sovrappongono in base agli elementary che gli compongono

n = len(compST)
sovra = [ [0] * n for _ in range(n) ]
sovraPerc = [ [0.0] * n for _ in range(n) ] 

for i in range(0,len(compST)):
    for j in range(0, len(compST)):
        for k in range(0, len(compST[i])):
            for l in range(0, len(compST[j])):
                if (compST[i][k] == compST[j][l]):
                    sovra[i][j] = sovra[i][j] + 1
                    sovraPerc[i][j] = float(sovra[i][j])*100/(len(compST[i]))

mSovra = np.asmatrix(sovra)
dfSovra = pd.DataFrame(mSovra, index = idST, columns = idST)
dfSovra.head()

mSovraPerc = np.asmatrix(sovraPerc)
dfSovraPerc = pd.DataFrame(mSovraPerc, index = idST, columns = idST)
dfSovraPerc.head()

for i in range(0,len(sovraPerc)):
    for j in range(0, len(sovraPerc)):
        if(i!=j) and (sovraPerc[i][j] >= 100.0):
            print( " --> " + idST[i] + " - " + idST[j] + " % = " + str (sovraPerc[i][j]) )

## Matrice e grafo delle sovrapposizioni
maSovra = sovra
enne = len(sovra)
for i in range(0,enne):
    for j in range(0,enne):
        if(maSovra[i][j] >= 1):
            maSovra[i][j] = 1
        else:
            maSovra[i][j] = 0

matrixAdiacenzaSovra = np.asmatrix(maSovra)
dfSovra = pd.DataFrame(matrixAdiacenzaSovra, index = idST, columns = idST)
dfSovra.head()

disegnaGrafo(matrixAdiacenzaSovra)

