#Creazione di grafo dei settori/blocchi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat #polygon
import networkx as nx

#Lettura del file excel/csv esportato da NEST
#airblock = pd.read_excel("ENV_1601_408Europe.xlsx", sheet_name="Airblock" )
airblock = pd.read_excel("LI_ENV_1601_408.xlsx", sheet_name="Airblock" )
airblock.columns=["ID", "Latitude", "Longitude"]
airblock.head()

#I blocchi sono memorizzati attraversole coordinate geografiche (lat, lon) dei loro vertici
# --> il dataset ha un formato scomodo per essere elaborato 
# --> vorrei ottenere una lista di vertici per ogni blocco
# N.B. :
#Latitude NORD --> Positive, SUD --> Negative,
#Longitude EST --> Positive, OVEST --> Negative

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

'''
L'adiacenza va modificata! .. non e' detto che solo se 2 blocchi hanno 2 vertici consecutivi in comune,
allora sono adiacenti Diremo che un blocco e' adiacente ad un altro se ha almeno un vertice che e' contenuto
nell'altro Per vertice intenderemo un suo intorno "largo" formato da 8 punti ad una certa distanza
'''
#Calcolo delle adiacenze:
#adiacenza calcola se il blocco i e' adiacente al blocco j rispetto la precedente funzione dovro' verificare
#se almeno un vertice del blocco i-esimo puo' essere contenuto nel blocco j, perche' questo sia vero almeno
#2 punti su 8 del suo intorno dovranno appartenervi

sogliaCounter=2
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
    for j in range((i+1), len(coordinate)):
        if adiacenza(i,j):
            mAd[i][j] = mAd[i][j] + 1
            mAd[j][i] = mAd[j][i] + 1

mAdd = np.asmatrix(mAd)
dfma = pd.DataFrame(mAdd, index = idBlocks, columns = idBlocks)

rows, cols = np.where(mAdd == 1)
edges = zip(rows.tolist(), cols.tolist())
gr = nx.Graph()
gr.add_edges_from(edges)
nx.draw(gr, node_size=300, node_color='yellow', with_labels=True)
plt.show()

file_name = 'matrAddItaly.csv'
dfma.to_csv(file_name, encoding='utf-8', index=False)