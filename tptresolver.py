#!/bin/python3.6

import os
import getopt
import sys
import random
import argparse
from pathlib import Path
import bin.utility as utility
import numpy as np
import tempfile
import bin.tpt_opt as GA
import bin.tpt_tabu as TABU

separator = "/"
_path = "./data"
timeout = 15
seed_number = 4
dire = './data'
file_list = None
seeds = []

def check_data(directory):

    global separator

    data_directory = 'data'
    if directory != None:
        data_directory = directory

    data_filename = 'data'

    accepted_weights = ['EUC_2D','CEIL_2D', 'GEO']

    to_remove = []
    print ("CHECKING ON FOLDER " + directory+ "\n")

    is_valid = False
    for root, dirs, files in os.walk(directory):
        is_valid = True
        for directoryName in dirs:
            if directoryName.endswith('.tsp'):
                with open(root+separator+directoryName+separator+data_filename, 'r') as fin:
                    lines = 0
                    for line in fin:
                        is_valid = False
                        message= "\033[1;31;40m NOT ACCEPTED \033[0m"
                        if "EDGE_WEIGHT_TYPE" in line:
                            for weights in accepted_weights:
                                if weights in line:
                                     is_valid = True
                                     message= "\033[1;32;40m ACCEPTED \033[0m"
                                     break
                            if not is_valid:
                                to_remove.append(directoryName)
                            print ("DATA IN "  + directoryName + " IS "+ \
                                       message+" ," + line.strip())
                            break
    if len(to_remove) > 0:
        print ("Those elements should be removed :" )
        print(to_remove)

    return is_valid



def read_data():
    file_list = []
    for root, dirs, files in os.walk(dire):
        for directoryName in sorted(dirs):
            if directoryName.endswith('.tsp'):
               file_list.append(root+separator+directoryName)

    return file_list



def GenerateProfits(n_cities):
    #random_integers [low,high]  we need (low,high] low excluded

        return np.random.random_integers((_minimum_dinstance_from_depot + 1 ),(n_cities/2)*_maximum_dinstance_from_depot)


def AssignProfits(_nodes,seed):
    utility.setRandomSeed(seed)
    size = len(_nodes)
    for i in range(size):
        _nodes[i][2]=\
        utility.GenerateProfits(_minimum_dinstance_from_depot,_maximum_dinstance_from_depot,size)
    return _nodes

def LoadData (_file):

    file = Path(_file+'/data')
    if not file.exists():
        print("No file found")
        exit(-1)

    global _minimum_dinstance_from_depot
    global _maximum_dinstance_from_depot

    _minimum_dinstance_from_depot = 0
    _maximum_dinstance_from_depot = 0
    _nodes= []
    size =0
    on_data = False
    x0 = y0 = 0
    depot= 0
    with open(file) as f:
        for line in f:
            line = line.strip('\n')

            if "NODE_COORD_SECTION" in line:
                on_data = True
                continue


            if on_data:
                line = line.split()
                if len(line) >=3:
                    x =float (line[1])
                    y =float (line[2])
                    _nodes.append([x,y,0])
                    if size == depot:
                        x0 = x
                        y0 = y
                    size += 1
                    if size > 1 :
                        dinstance = utility.CalculateDistance(x0,y0,x,y)

                        if size == 2:
                            _minimum_dinstance_from_depot =\
                            _maximum_dinstance_from_depot = dinstance
                        elif (dinstance < _minimum_dinstance_from_depot):
                             _minimum_dinstance_from_depot = dinstance
                        elif(dinstance > _maximum_dinstance_from_depot):
                             _maximum_dinstance_from_depot = dinstance
    return _nodes

def parse():
  parser = argparse.ArgumentParser(description='Load TSP instances and uses GA\
                                   and Tabu to resolve them.')
  parser.add_argument('-s','--seed',help="Specify an integer value used as\
                      random's seed",\
                      type = int)
  parser.add_argument('-t','--timeout',help='Specify the timeout (seconds)',type\
                      = int )
  parser.add_argument('--directory','-d',help='The data directory')
  parser.add_argument('-file','-f',nargs='+',help='Specify instances')
  parser.add_argument('-c','--check-data',help='Check data in specific\
                      directory')
  parser.add_argument('-p','--profit',nargs='+', help='Adds profits to data')
  args = parser.parse_args()
  return args


def init(args):

    if args.check_data:
        if not check_data(args.check_data):
            print("No valid data")
            sys.exit()
    if args.timeout:
        global timeout
        timeout = args.timeout
    if args.directory:
        global dire
        dire = args.directory
    if args.file:
        global file_list
        file_list = args.file
    return

if __name__ == "__main__":

    init(parse())

    if file_list == None:
        file_list = read_data() #if no file is specified load everything

    for current_file in file_list:
        print (current_file)
        nodes = LoadData(current_file)

        for i in range(seed_number):

            if i < len(seeds):
                current_seed = seeds[seed_index]
            else:
                current_seed = random.randint(1,2**32)

            nodes = AssignProfits(nodes,current_seed)
            GA.main(current_file,nodes)
            TABU.main(current_file,nodes)

    sys.exit()
