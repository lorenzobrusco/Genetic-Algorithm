import math
import numpy as np

def CalculateDistance (x1,y1,x2,y2):

    x_distance = x1 - x2
    y_distance = y1 - y2

    return int(math.sqrt((x_distance * x_distance) + (y_distance *\
                                                      y_distance)))

def GenerateProfits(minimum,maximum,n):
    #random_integers [low,high]  we need (low,high] low excluded

        return np.random.random_integers((minimum+ 1 ),(n/2)*maximum)

def setRandomSeed(seed):
    np.random.seed(seed)
