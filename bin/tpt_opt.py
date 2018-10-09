import random
import sys
import copy
import time
import os
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.compat import range
from threading import Thread, Event
import time
import gc
import argparse
import numpy as np
import pandas as pd
#from utility import CalculateDistance
#import utility
import math

seed = 4
_timeout = 60
_file= ""
iteration = 0
_depo = 0
_travel_cost = 1
_unitary_cost = 1
_unused_node = -9
_penality = False
_maximum_dinstance_from_depot = 0
_minimum_dinstance_from_depot = 0
random.seed(seed)
location = None
population = None
ga = None
file_list = []
logfile = None

def parse():
    parser = argparse.ArgumentParser(description='Implementation of GA.')
    parser.add_argument('-s','--seed',help="Specify an integer value used as\
                        random's seed",\
                        type = int)
    parser.add_argument('-t','--timeout',help='Specify the timeout (seconds)',type\
                        = int )
    parser.add_argument('file',nargs='+',help='Specify the input file')
    args = parser.parse_args()
    return args


def init(args):

    if args.seed:
        global seed
        seed = args.seed
    if args.timeout :
        global _timeout
        _timeout = args.timeout
    if args.file:
        global file_list
        file_list = args.file

    np.random.seed(seed)

def CalculateDistance (x1,y1,x2=None,y2= None):

    if (x2 == None and y2 == None):
        node1 = x1;
        node2 = y1;
        x_distance = _nodes[node1][0] - _nodes[node2][0]
        y_distance = _nodes[node1][1] - _nodes[node2][1]

    else :
        x_distance = x1 - x2
        y_distance = y1 - y2

    return int(math.sqrt((x_distance * x_distance) + (y_distance *\
                                                      y_distance)))


def calculate_distance(location):
    global _maximum_dinstance_from_depot
    global _minimum_dinstance_from_depot

    assign = True
    size = len(location)
    for j in range(_depo +1,size):
        distance = CalculateDistance(_depo,j)

        if distance > _maximum_dinstance_from_depot:

            _maximum_dinstance_from_depot = distance

        elif distance < _minimum_dinstance_from_depot:

            _minimum_dinstance_from_depot = distance

    return

'''
            This function is the implementation of :
"Heuristics for the traveling repairman problem with profits"
T. Dewilde, D. Cattrysse , S. Coene , F.C.R. Spieksma, P. Vansteenwegen
                            -Section 4

'''

def calculate_profits(location,n_cities):
    #random_integers [low,high]  we need (low,high] low excluded

    for i in range(n_cities):
        profit =\
        np.random.random_integers((_minimum_dinstance_from_depot + 1 ),(n_cities/2)*_maximum_dinstance_from_depot)
        location[i][2] = profit

def load_data(_file):
    file = Path(_file+'/data')
    if not file.exists():
        print("No file found")
        exit(-1)

    global _nodes
    global _minimum_dinstance_from_depot
    global _maximum_dinstance_from_depot

    _nodes = []
    size =0
    on_data = False
    with open(file) as f:
        for line in f:
            if "NODE_COORD_SECTION" in line:
                on_data = True
                continue
            if on_data:
                line = line.split()
                if len(line) ==3:
                    x =float (line[1])
                    y =float (line[2])
                    _nodes.append([x,y,0])
                    if size == _depo:
                        x0 = x
                        y0 = y

                    size = len(_nodes)
                    if size > 1 :

                        dinstance = CalculateDistance(x0,y0,x,y)
                        if size == 2:
                            _minimum_dinstance_from_depot =\
                            _maximum_dinstance_from_depot = dinstance
                        if (dinstance < _minimum_dinstance_from_depot):
                             _minimum_dinstance_from_depot = dinstance
                        elif(dinstance > _maximum_dinstance_from_depot):
                             _maximum_dinstance_from_depot = dinstance

    return size


class Individual:

    def __init__(self, create=False, genes=None):
        self._fitness = 0
        self._genes = []
        self.useless = []
        if create is True:
            self._genes = self.generate_individual()
        if genes is not None:
            self._genes = genes
        self.calculate_fitness()

    def __repr__(self):
        s = "["
        for gene in self._genes:
            s += " %2d" % gene
        s += "]"
        return s

    def generate_individual(self):
        """
            Generate the default individual simple get
            a sorted integers list
        """
        genes = []
        for i in range(_n_cities):
            genes.append((i + 1) % _n_cities)
        return genes

    def calculate_total_distance(self):

        size = len (self.get_genes)
        for i in range(size):
            x1 = self.get_genes[i][0]
            y1 = self.get_genes[i][1]

            x2 = self.get_genes[(i+1) % size][0]
            y2 = self.get_genes[(i+1) % size][1]
            distance += CalculateDistance(x1,y1,x2,y2)
        return distance

    def calculate_fitness(self):
        """
            Calculate the fitness function
        """
        self._fitness = Fitness.calculate(self._genes)

    def remove(self, i):
        useless = self._genes[i]
        self._genes.remove(useless)
        self.useless.append(useless)

    def get_fitness(self):
        return self._fitness

    def get_genes(self):
        return self._genes

    def set_genes(self, genes):
        self._genes = genes

    def get_gene(self, index):
        return self._genes[index]

    def get_useless(self):
        return self.useless

    def size(self):
        return len(self._genes)

    def append(self, value):
        self._genes.append(value)

    def set_gene(self, index, value):
        self._genes[index] = value

    def contains(self, value):
        return value in self._genes

    def sort(self):
        return self._genes.sort()

    def insert(self, i, value):
        self._genes.insert(i, value)


class Population:
    def __init__(self, population_size, initialise=False):
        self._individuals = []
        if initialise is True:
            print ("initializing population , size " + str(population_size))
            for i in range(population_size):
                individual = Individual(create=True)
                self._individuals.append(individual)
        pass

    def get_individual(self, index):
        if len(self._individuals) <= 0:
           return Individual(create=True)
        return self._individuals[index]

    def save_individual(self, individual):
        self._individuals.append(individual)

    def size(self):
        return len(self._individuals)

    def beast_fitness(self):
        """
            It finds the beast individual's fitness
            :return:
        """

        fittest = self.get_individual(0)
        for i in range(self.size()):
            individual = self.get_individual(i)
            if fittest.get_fitness() <= individual.get_fitness():
                fittest = individual
        return fittest


class GA:

    def __init__(self, uniform_rate=0.5, mutation_rate=0.015, tournament_size=10, elitism=True, two_opt=False):
        self.uniform_rate = uniform_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.two_opt = two_opt
        self.best = Individual(True)

    def evolve_population(self, population):
        new_population = Population(population.size(), initialise=False)
        if self.elitism:
            if self.best.get_fitness() <= population.beast_fitness().get_fitness():
                if self.two_opt is True:
                    self.best.set_genes(self.twoOpt(population.beast_fitness().get_genes()))
                else:
                    self.best = population.beast_fitness()

        for i in range(population.size()):
           # if check_state():
           #     return population
            if self.elitism:
                indiv1 = copy.deepcopy(self.best)
            else:
                indiv1 = self.tournament_selection(population)
            indiv2 = self.tournament_selection(population)
            new_indiv = self.crossover(indiv1, indiv2)
            new_population.save_individual(new_indiv)

        for i in range(new_population.size()):
            self.mutate(new_population.get_individual(i))
        return new_population

    def crossover(self, indiv1, indiv2):
        labeled = []
        new_indiv = Individual()
        new_indiv.insert(0, _depo)
        if random.uniform(0, 1) <= self.uniform_rate:
            pather1 = indiv1
            pather2 = indiv2
        else:
            pather2 = indiv1
            pather1 = indiv2
        first_split = random.randint(0, pather1.size() - 2)
        second_split = random.randint(first_split + 1, pather1.size() - 1)
        for i in range(first_split, second_split):
            #if check_state ():
            #    return new_indiv
            labeled.append(pather1.get_gene(i))
        for i in range(pather2.size()):
            #if check_state():
            #    return new_indiv
            if pather2.get_gene(i) not in labeled:
                if i < first_split:
                    new_indiv.append(pather2.get_gene(i))
                else:
                    labeled.append(pather2.get_gene(i))
        for i in range(len(labeled)):
            new_indiv.append(labeled[i])
            first_split += 1
        return new_indiv

    def mutate(self, indiv):
        for i in range(1, indiv.size()):
            if check_state():
                return
            if i >= indiv.size():
                break
            if random.uniform(0, 1) <= self.mutation_rate:
                if random.uniform(0, 1) <= self.uniform_rate:
                    indiv.remove(i)
                else:
                    if len(indiv.get_useless()) > 0:
                        useless = indiv.get_useless().pop()
                        indiv.insert(i, useless)
                    else:
                        indiv.remove(i)
        uniq = []
        [uniq.append(x) for x in indiv.get_genes() if x not in uniq]
        indiv.set_genes(uniq)
        indiv.append(_depo)
        indiv.calculate_fitness()

    def tournament_selection(self, population):
        tournament = Population(self.tournament_size, False)
        fittest = tournament.beast_fitness()
        for k in range(self.tournament_size):
            if check_state():
                return fittest
            random_id = random.randint(0, population.size() - 1)
            tournament.save_individual(population.get_individual(random_id))
        fittest = tournament.beast_fitness()
        return fittest

    def twoOpt(self, route):
        xx = 0
        while (True):
            xx += 1
            temp_route = list(route)
            route_distance = -999999999
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]



                    diff_distance = CalculateDistance(_nodes[route[i-1]][0],\
                                                      _nodes[route[i-1]][1],\
                                                      _nodes[route[i]][0],\
                                                      _nodes[route[i]][1]) +\
                            CalculateDistance(_nodes[route[j]][0],\
                                              _nodes[route[j]][1],\
                                              _nodes[route[j+1]][0],\
                                              _nodes[route[j+1]][1])
                    diff_distance = diff_distance -\
                    CalculateDistance(_nodes[new_route[i -1]][0],\
                                      _nodes[new_route[i -1]][1],\
                                      _nodes[new_route[i]][0],\
                                      _nodes[new_route[i]][1])\
                            - CalculateDistance(_nodes[new_route[j]][0],\
                                                _nodes[new_route[j]][1],\
                                                _nodes[new_route[j+1]][0],\
                                                _nodes[new_route[j+1]][1])

                    if diff_distance > route_distance:
                        temp_route = list(new_route)
                        route_distance = diff_distance
                    #if check_state():
                    #    return route
            if route_distance > 0.01:
                route = list(temp_route)
            else:
                break
        return route


class Fitness:

    @staticmethod
    def calculate(solution):
        if solution is None or len(solution) == 0:
            return -999999
        prof = 0
        solution_size = len(solution)
        for i in range(solution_size):
            #if check_state():
            #    return 0


           # x1= _nodes[solution[i]][0]
           # y1= _nodes[solution[i]][1]

           # x2 = _nodes[solution[(i + 1) % solution_size]][0]
           # y2 = _nodes[solution[(i + 1) % solution_size]][1]

            cost =CalculateDistance(_nodes[solution[i]][0],\
                                    _nodes[solution[i]][1],\
                                    _nodes[solution[(i + 1) % solution_size]][0],\
                                    _nodes[solution[(i + 1) % solution_size]][1])\
                                    * _travel_cost
            prof += (_nodes[solution[(i + 1) % solution_size]][2] - cost)

        if _penality is True:
            nodes = []
            [nodes.append(x) for x in range(_n_cities) if x not in solution]
            for i in nodes:
                prof -= _nodes[i][2]
        return prof


def generate_graph(graph, location, show=True):
    G = nx.DiGraph()
    plt.figure(figsize=(9, 9))
    plt.axis('off')
    for i in range(len(location)):
        x = location[i][0]
        y = location[i][1]
        G.add_node(i, pos=(x, y))
    for i in range(graph.size()):
        G.add_edge(graph.get_gene(i), graph.get_gene((i + 1) % graph.size()))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'))
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, arrows=True)

    if show is True:
        plt.show()

def action(printed=False):
    global population
    global ga

    print ("Algorithm started\n")
    i = 0
    while(not check_state()):
        if printed is True:
            print("Iteration %d: generate new population,  fitness %d\n" % (i, population.beast_fitness().get_fitness()))
        population = ga.evolve_population(population)
        i += 1
    print ("Timeout reached")

def check_state():
    global stop_event
    return stop_event.is_set()

def solve_ga(graphic=False, printed=True):
    global location
    global population
    global ga
    global _timeout

    population = Population(_n_cities, True)
    ga = GA(two_opt=False)
    start = time.clock()
    action_thread = Thread (target=action, args=(printed,))
    action_thread.start()
    action_thread.join(timeout=_timeout)
    stop_event.set()
    end = time.clock()
    ga_time = (end - start)
    writeToFile("GA\n")
    writeToFile("Best Objective Value: %.2f\n" % \
                population.beast_fitness().get_fitness(),True)
    writeToFile("Number of Customers Visited (Depot Excluded): %d \n" \
            % (population.beast_fitness().size() - 2),True)
    writeToFile("Sequence of Customers Visited:\n %s\n" % \
                population.beast_fitness(),True)
    writeToFile("CPU Time (s): %.2f\n" %ga_time,True)

    if action_thread.isAlive():
        action_thread.join()

    del(action_thread)

    if graphic is True:
        generate_graph(population.beast_fitness(), _nodes)


def writeToFile(value,stdout = False):
    global logfile
    global iteration

    if stdout:
        print (value)

    with open('./results.txt','a') as f:
        f.write(value+'\n')
        f.close()


def main(_file,nodes):

    global stop_event
    stop_event = Event()
    global _nodes
    _nodes = nodes
    global _n_cities
    _n_cities = len(_nodes)
    cities = [(index + 1) % _n_cities for index in range(0, _n_cities)]
    fitness = Fitness.calculate(cities)
    solve_ga(graphic=False, printed=False)
    stop_event.clear()
    if logfile !=None:
        logfile.close()

    return


if __name__ == "__main__":
    init(parse())
    global stop_event
    width = 35
    output_size = 55
    stop_event = Event()
    for _file in file_list:
        final_string = '-' * output_size + '\n'
        string = "Instance : %s" %(_file,)
        str_len = len(string)
        string += ' ' * (width - str_len)
        string += "Seed : [ %10d ]\n" %(seed,)
        final_string += string
        string =""
        start = time.clock()
        _n_cities = load_data(_file)
        end = time.clock()
        final_string += '-' * output_size + '\n'
        string +="Data loaded in : %.2f "  % (end - start,)
        string += ' ' * (width - len(string))
        start = time.clock()
        calculate_profits(_nodes,_n_cities)
        end = time.clock()
        string += "Profits calculated : %.2f\n" % (end -start,)
        final_string += string
        string =""
        start = time.clock()
        cities = [(index + 1) % _n_cities for index in range(0, _n_cities)]
        end = time.clock()
        string ="Instantiation : %.2f" % (end-start)
        string += ' ' * (width - len(string))
        final_string += string
        fitness = Fitness.calculate(cities)
        string = "Initial fitness: %d\n" % (fitness,)
        final_string += string
        print (final_string)
        del(string)
        del(start)
        del(end)
        solve_ga(graphic=False, printed=False)
        stop_event.clear()
    if logfile !=None:
        logfile.close()
    sys.exit()
