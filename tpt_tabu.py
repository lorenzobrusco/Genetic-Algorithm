import numpy as np
import math
import time
import random
import itertools
import queue
import pandas as pd
from IPython.display import display, Markdown
import networkx as nx
import matplotlib.pyplot as plt

# HP or LP
dataset = "HP"
# 51, 76 or 101
N = 101

filename = "data/dataset-" + dataset + ".xlsx"
df = pd.read_excel(filename, sheetname="eil" + str(N), header=None, index_col=0)

df.columns = ['x', 'y', 'prof']

display(df[0:10])

distances = [-1]
prof = [-1]

for lab, row in df.iterrows():
    tempDist = [-1]
    prof.append(row['prof'])
    for lab2, row2 in df.iterrows():
        dist = math.sqrt(math.pow(row['x'] - row2['x'], 2) + math.pow(row['y'] - row2['y'], 2))
        tempDist.append(dist)
    distances.append(tempDist)

# dff holds the main data as given from the xls
# Started the indices from 1
dff = [[0, 0, 0]]
for lab, row in df.iterrows():
    dff.append([row['x'], row['y'], row['prof']])


def calculateObj(route):
    if len(route) == 0:
        return -99999999

    objVal = 0

    for i in range(1, len(route)):
        objVal = objVal + dff[route[i]][2] - distances[route[i - 1]][route[i]]

    return objVal


def calculateTour(route):
    objVal = 0

    for i in range(1, len(route)):
        objVal = objVal + distances[route[i - 1]][route[i]]

    return objVal


def updateGraph(G, old_route, route, se, visualize):
    G.remove_edge(old_route[se[0] - 1], old_route[se[0]])
    G.remove_edge(old_route[se[1]], old_route[se[1] + 1])
    G.add_edge(route[se[0] - 1], route[se[0]])
    G.add_edge(route[se[1]], route[se[1] + 1])
    if visualize:
        nx.draw(G, pos, with_labels=True)
        plt.show()
        print(str(old_route[se[0] - 1]) + ',' + str(old_route[se[0]]) + ' - ' + str(old_route[se[1]]) + ',' + str(
            old_route[se[1] + 1]))
    return G


def twoOpt(route, G=None, visualize=False):
    if G != None:
        pos = nx.get_node_attributes(G, 'pos')
    if visualize and G != None:
        nx.draw(G, pos, with_labels=True)
        plt.show()
    se = (0, 0)
    xx = 0
    while (True):
        xx = xx + 1
        temp_route = list(route)
        old_route = list(route)
        route_distance = -999999999
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                diff_distance = distances[route[i - 1]][route[i]] + distances[route[j]][route[j + 1]]
                diff_distance = diff_distance - distances[new_route[i - 1]][new_route[i]] - distances[new_route[j]][
                    new_route[j + 1]]
                if diff_distance > route_distance:
                    temp_route = list(new_route)
                    route_distance = diff_distance
                    se = (i, j)
        if route_distance > 0.01:
            route = list(temp_route)
            if G != None:
                G = updateGraph(G, old_route, route, se, visualize)
        else:
            break
    return route, G


def threeOptSwap(route, i, j, k):
    bestRoute = list(route)
    best_diff = 0

    a = i
    b = j + 1
    c = k + 2

    nRoute = route[:a] + list(reversed(route[a:b])) + list(reversed(route[b:c])) + route[c:]
    diff = distances[route[a - 1]][route[a]] + distances[route[b - 1]][route[b]] + distances[route[c - 1]][route[c]]
    diff = diff - distances[route[a - 1]][route[b - 1]] - distances[route[a]][route[c - 1]] - distances[route[b]][
        route[c]]
    if diff > best_diff:
        best_diff = diff
        bestRoute = list(nRoute)

    nRoute = route[:a] + route[b:c] + route[a:b] + route[c:]
    diff = distances[route[a - 1]][route[a]] + distances[route[b - 1]][route[b]] + distances[route[c - 1]][route[c]]
    diff = diff - distances[route[a - 1]][route[b]] - distances[route[c - 1]][route[a]] - distances[route[b - 1]][
        route[c]]
    if diff > best_diff:
        best_diff = diff
        bestRoute = list(nRoute)

    nRoute = route[:a] + route[b:c] + list(reversed(route[a:b])) + route[c:]
    diff = distances[route[a - 1]][route[a]] + distances[route[b - 1]][route[b]] + distances[route[c - 1]][route[c]]
    diff = diff - distances[route[a - 1]][route[b]] - distances[route[c - 1]][route[b - 1]] - distances[route[a]][
        route[c]]
    if diff > best_diff:
        best_diff = diff
        bestRoute = list(nRoute)

    nRoute = route[:a] + list(reversed(route[b:c])) + route[a:b] + route[c:]
    diff = distances[route[a - 1]][route[a]] + distances[route[b - 1]][route[b]] + distances[route[c - 1]][route[c]]
    diff = diff - distances[route[a - 1]][route[c - 1]] - distances[route[b]][route[a]] - distances[route[b - 1]][
        route[c]]
    if diff > best_diff:
        best_diff = diff
        bestRoute = list(nRoute)

    return bestRoute, best_diff


def threeOpt(route):
    xx = 0
    while (True):
        xx += 1
        temp_route = list(route)
        old_route = list(route)
        best_diff = 0.01
        brk = False
        li = list(range(1, len(route) - 2))
        random.shuffle(li)
        for i in li:
            lj = list(range(i, len(route) - 2))
            random.shuffle(lj)
            for j in lj:
                lk = list(range(j, len(route) - 2))
                random.shuffle(lk)
                for k in lk:
                    new_route, new_diff = threeOptSwap(route, i, j, k)
                    if new_diff > best_diff:
                        temp_route = list(new_route)
                        best_diff = new_diff
                        brk = True
                        break
                if brk:
                    break
            if brk:
                break
        if not brk:
            break
        if best_diff > 0.01:
            route = list(temp_route)
        else:
            break
    return route


def initialization():
    ''' Construction Heuristic '''
    best_objs = []
    best_routes = []
    for i in [int(N / 2)]:
        local_obj = -99999999
        local_route = []
        for t in range(5):
            route = [1, 1]
            for j in range(i):
                min_obj = 99999999
                k = random.randint(0, len(route) - 2)
                temp_route = list(route)
                for lab in range(1, N + 1):
                    if lab not in route:
                        new_route = route[:k + 1] + [lab] + route[k + 1:]
                        diff_obj = (distances[route[k]][lab] + distances[lab][route[k + 1]] - distances[route[k]][
                            route[k + 1]]) / prof[lab]
                        if diff_obj < min_obj:
                            temp_route = list(new_route)
                            min_obj = diff_obj
                route = list(temp_route)
            temp_route = twoOpt(route)[0]
            temp_obj = calculateObj(temp_route)
            if temp_obj > local_obj:
                local_obj = temp_obj
                local_route = list(temp_route)

        best_routes.append(local_route)
        best_objs.append(local_obj)

    route = list(best_routes[0])
    rat = 0
    for i in range(len(best_routes)):
        if best_objs[i] / len(best_routes[i]) > rat:
            rat = best_objs[i] / len(best_routes[i])
            route = list(best_routes[i])

    return route


def dispersionIndex(cluster):
    if len(cluster) == 1:
        return 0
    else:
        sm = 0
        for c1 in cluster:
            for c2 in cluster:
                sm = sm + distances[c1][c2]
        return sm / (len(cluster) * (len(cluster) - 1))


def proximityMeasure(cluster1, cluster2):
    sm = 0
    for c1 in cluster1:
        for c2 in cluster2:
            sm = sm + distances[c1][c2]

    return (2 / (len(cluster1) * len(cluster2))) * sm - dispersionIndex(cluster1) - dispersionIndex(cluster2)


def insertionCandidates():
    candidates = []
    rList = [1, int(N / 2), int(2 * N / 3), int(3 * N / 4), int(4 * N / 5), int(5 * N / 6), int(6 * N / 7),
             int(7 * N / 8), int(8 * N / 9), int(9 * N / 10)]

    Pr = []
    Pr = [[x] for x in range(2, N + 1)]
    candidates.append(list(Pr))

    for r in range(2, N):
        minProx = 99999999
        minProxInd = []
        for i in range(len(Pr)):
            for j in range(i + 1, len(Pr)):
                pM = proximityMeasure(Pr[i], Pr[j])
                if pM < minProx:
                    minProx = pM
                    minProxInd = [i, j]
        Pr.append(Pr[minProxInd[0]] + Pr[minProxInd[1]])
        del (Pr[minProxInd[1]])
        del (Pr[minProxInd[0]])

        if r in rList:
            candidates.append(list(Pr))

    return candidates


def deletionCandidates(route):
    candidates = []
    edges = []

    K = random.randint(2, int(max(4, len(route)) / 2))

    for i in range(len(route) - 1):
        edges.append([distances[route[i]][route[i + 1]], i, i + 1])

    edges = list(reversed(sorted(edges)))[:K]
    edges.sort(key=lambda x: x[1])

    for i in range(K - 1):
        tempList = []
        for j in range(edges[i][2], edges[i + 1][1] + 1):
            tempList.append(route[j])

        candidates.append(tempList)

    return candidates


def findBestInsertionCandidate(route, tabuList, insCandidates):
    bestInsCandidate = []
    bestInsObj = -99999999

    for iC in insCandidates:
        profitSum = 0
        gCenter = [0, 0]
        for c in iC:
            if c not in route and c not in tabuList:
                gCenter[0] = gCenter[0] + dff[c][0] / len(iC)
                gCenter[1] = gCenter[1] + dff[c][1] / len(iC)
                profitSum = profitSum + dff[c][2]

        minDist = 99999999
        for j in range(len(route) - 1):
            distAdd1 = calculateDist(dff[route[j]][0], dff[route[j]][1], gCenter[0], gCenter[1])
            distAdd2 = calculateDist(gCenter[0], gCenter[1], dff[route[j + 1]][0], dff[route[j + 1]][1])
            distRem = calculateDist(dff[route[j]][0], dff[route[j]][1], dff[route[j + 1]][0], dff[route[j + 1]][1])

            dist = distAdd1 + distAdd2 - distRem
            if dist < minDist:
                minDist = dist

        if profitSum / minDist > bestInsObj:
            bestInsObj = profitSum / minDist
            bestInsCandidate = list(iC)

    return bestInsCandidate


def calculateDist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


# Iteration Count
ITER = 1000

# Start the timer
t1 = time.clock()

# Create the initial route
route = initialization()

# Determine all possible insertion partitions
insCandidatesAll = insertionCandidates()
tabuList = {}
solutionIndex = [0]

bestRoute = list(route)
bestObj = calculateObj(bestRoute)

# Start tabu search
for i in range(ITER):
    # Choose one insertion partition ramdompy
    insCandidates = list(insCandidatesAll[random.randint(0, len(insCandidatesAll) - 1)])

    # Determine deletion candidates
    if len(route) < 3:
        delCandidates = []
    else:
        delCandidates = deletionCandidates(route)

    candidateRoute = []
    tabuAddition = []

    # Find best insertion candidate from the selected partition
    bestInsCandidate = findBestInsertionCandidate(route, tabuList, insCandidates)

    # Calculate the gain of inserting the insertion candidate to the route
    insertedRoute = list(route)
    profitSum = 0
    distSum = 0
    random.shuffle(bestInsCandidate)
    for c in bestInsCandidate:
        if c not in insertedRoute and c not in tabuList:
            profitSum = profitSum + dff[c][2]
            minDist = 99999999
            temp_route = list(insertedRoute)
            for j in range(len(insertedRoute) - 1):
                new_route = insertedRoute[:j + 1] + [c] + insertedRoute[j + 1:]
                diffDist = distances[insertedRoute[j]][c] + distances[c][insertedRoute[j + 1]] - \
                           distances[insertedRoute[j]][insertedRoute[j + 1]]
                if diffDist < minDist:
                    temp_route = list(new_route)
                    minDist = diffDist
            insertedRoute = list(temp_route)
            distSum = distSum + minDist
    if distSum == 0:
        distSum = 99999999
    insertedObj = profitSum / distSum

    # Choose the best deletion candidate from the selected ones, then calculate its gain
    deletedRoute = list(route)
    maxDeletedObj = -99999999
    for dC in delCandidates:
        tempRoute = list(route)
        profitSum = 0
        distSum = 0
        for c in dC:
            if c in tempRoute:
                cPrev = tempRoute[tempRoute.index(c) - 1]
                cNext = tempRoute[tempRoute.index(c) + 1]

                profitSum = profitSum + dff[c][2]
                distSum = distances[cPrev][c] + distances[c][cNext] - distances[cPrev][cNext]
                tempRoute.remove(c)
        if profitSum != 0 and distSum / profitSum > maxDeletedObj:
            maxDeletedObj = distSum / profitSum
            deletedRoute = list(tempRoute)
            tabuAddition = list(dC)
    deletedObj = maxDeletedObj

    # Compare the insertion and deletion gains, and apply the better one
    if insertedObj > deletedObj:
        candidateRoute = list(insertedRoute)
        chosen = ['I', len(insertedRoute) - len(route)]
    else:
        candidateRoute = list(deletedRoute)
        chosen = ['D', len(route) - len(deletedRoute)]

    # Update the tabu list
    for key, value in list(tabuList.items()):
        tabuList[key] = tabuList[key] - 1
        if tabuList[key] == 0:
            del (tabuList[key])

    # If deletion action is performed then add the chosen deletion candidates to the tabu list.
    if chosen[0] == 'D':
        for tA in tabuAddition:
            if tA in route:
                tabuList[tA] = random.randint(5, 25)

    route = list(candidateRoute)

    # Improve the route
    if i % 5 == 0:
        route = twoOpt(route)[0]

    # Best solution update
    if calculateObj(route) > bestObj:
        solutionIndex.append(i)
        route = threeOpt(route)
        bestRoute = list(route)
        bestObj = calculateObj(route)

    # Shuffle to Reset
    if i - solutionIndex[-1] >= 1000:
        tabuList.clear()
        tempRoute = bestRoute[1:-1]
        random.shuffle(tempRoute)
        tempRoute = [1] + tempRoute + [1]
        route = list(tempRoute)
        solutionIndex.append(i)

# Stop  the timer
t2 = time.clock()

print("Instance: ")
print("eil" + str(N) + "-" + str(dataset))
print()

print("Best Objective Value:")
print("%.2f" % calculateObj(bestRoute))
print()

print("Number of Customers Visited (Depot Excluded):")
print(len(bestRoute) - 2)
print()

print("Sequence of Customers Visited:")
print(bestRoute)
print()

print("CPU Time (s):")
timePassed = (t2 - t1)
print("%.2f" % timePassed)

plt.figure(figsize=(9, 9))

G = nx.Graph()

for lab, row in df.iterrows():
    G.add_node(lab, pos=(row['x'], row['y']))

for i in range(1, len(bestRoute)):
    G.add_edge(bestRoute[i - 1], bestRoute[i])

pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True)

plt.show()