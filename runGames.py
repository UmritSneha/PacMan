import matplotlib.pyplot as plt
from datetime import datetime
from pacman import *
import ghostAgents
import layout
import textDisplay
import graphicsDisplay
import copy
import numpy as np
from pprint import pprint
import sys

## set up the parameters to newGame
gridx = 25
gridy = 20
numtraining = 0
timeout = 30

# set layout of game
layout=layout.getLayout("mediumClassic")
# set game Agent
pacmanType = loadAgent("pacmanAgent", True)
# set number of agents
numGhosts = 2
ghosts = [ghostAgents.RandomGhost(i+1) for i in range(numGhosts)]
catchExceptions=True
# set possible directions
options = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]

def run(code,noOfRuns, beQuiet):
    rules = ClassicGameRules(timeout)
    games = []
    if beQuiet:
        gameDisplay = textDisplay.NullGraphics()
        rules.quiet = True
    else:
        timeInterval = 0.001
        textDisplay.SLEEP_TIME = timeInterval
        gameDisplay = graphicsDisplay.PacmanGraphics(1.0, timeInterval)
        rules.quiet = False
    for gg in range(noOfRuns):
        thePacman = pacmanType()
        thePacman.setCode(code)
        
        game = rules.newGame( layout, thePacman, ghosts, gameDisplay, \
                          beQuiet, catchExceptions )
        game.run()
        games.append(game)
    scores = [game.state.getScore() for game in games]
    return sum(scores) / float(len(scores))

## Genetic Algorithm

# mutation
def mutate(parentp,numberOfMutations=10):
    parent = copy.deepcopy(parentp)
    for _ in range(numberOfMutations):
        xx = random.randrange(gridx)
        yy = random.randrange(gridy)
        parent[xx][yy] = random.choice(options)
    return parent

# crossover
def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    for xx in range(gridx):
        for yy in range(gridy):
            child[xx][yy] = random.choice([parent1[xx][yy], parent2[xx][yy]])
    return child

# run search algorithm
def run_genetic(popSiz=20, timescale=20, numberOfRuns=40, tournamentSize=7):
    # create random initial population
    population = []
    for _ in range(popSiz):
        program = np.empty((gridx,gridy),dtype=object)
        for xx in range(gridx):
            for yy in range(gridy):
                program[xx][yy] = random.choice(options)
        population.append(program)
            
    print("Beginning Evolution")
    averages = []
    bests = []
    for tt in range(timescale):
        # evaluate population
        fitness = []
        for pp in population:
            print(".",end="",flush=True)
            if tt < timescale - 1:
                fitness.append(run(pp,numberOfRuns, True))
            else: 
                fitness.append(run(pp,numberOfRuns, False))
        
        print("\n******")
        print(fitness)
        averages.append(1000+sum(fitness)/popSiz)
        print("av ",1000+sum(fitness)/popSiz)
        bests.append(1000+max(fitness))
        print("max ",1000+max(fitness))

        popFitPairs = list(zip(population,fitness))
        # print('population fitness pairs: ', popFitPairs)
        newPopulation = []
        for _ in range(popSiz-1):
            ## crossover the parents
            # selecting first parent
            tournament1 = random.sample(popFitPairs, tournamentSize)
            parent1 = max(tournament1,key=lambda x:x[1])[0]

            # selecting second parent
            tournament2 = random.sample(popFitPairs, tournamentSize)
            parent2 = max(tournament2,key=lambda x:x[1])[0]

            # apply crossover to obtain child
            child = crossover(parent1, parent2)

            # mutate child
            mutant_child = mutate(child)

            # add to new population
            newPopulation.append(mutant_child)

        ## Keeping best population member
        best_member = [list(pop) for pop,fit in popFitPairs if fit == max(fitness)]
        # print('Best Member: ', best_member[0])
        newPopulation[-1] = best_member[0]
        population = copy.deepcopy(newPopulation)

    print(averages)
    print(bests)
    
    ## Plotting averages and bests
    plt.plot(averages, label='average')
    plt.plot(bests, label='best')
    plt.xlabel("time")
    plt.ylabel("score")
    plt.legend()
    plt.show()

def run_memetic(popSize=20, timescale=20, numberOfRuns=15, tournamentSize=7):
   # create random initial population
    population = []
    for _ in range(popSize):
        program = np.empty((gridx,gridy),dtype=object)
        for xx in range(gridx):
            for yy in range(gridy):
                program[xx][yy] = random.choice(options)
        population.append(program)
            
    print("Beginning Evolution")
    averages = []
    bests = []
    for tt in range(timescale):
        # evaluate population
        fitness = []
        for pp in population:
            print(".",end="",flush=True)
            fitness.append(run(pp,numberOfRuns, True))
        
        print("\n******")
        print(fitness)
        
        averages.append(1000+sum(fitness)/popSize)
        print("av ",1000+sum(fitness)/popSize)
        
        bests.append(1000+max(fitness))
        print("max ",1000+max(fitness))
        max_fitness = max(fitness)

        popFitPairs = list(zip(population,fitness))
        # print('population fitness pairs: ', popFitPairs)
        newPopulation = []
        for _ in range(popSize-1):
            tournament1 = random.sample(popFitPairs, tournamentSize)
            parent1 = max(tournament1,key=lambda x:x[1])[0]

            tournament2 = random.sample(popFitPairs, tournamentSize)
            parent2 = max(tournament2,key=lambda x:x[1])[0]

            ## apply local search, hill climbing algortihm
            loop_index = 0
            stop = False
            # intial state
            child = crossover(parent1, parent2) 
            # evaluating intial state
            child_score = run(child,numberOfRuns, True)

            # check if initial state is greater than goal state
            if child_score >= max_fitness:
                newPopulation.append(child)
            else:
                # Loop until solution state is found or termination criteria is met
                while stop == False:
                    mutant_list = []
                    score_list = []
                    
                    # apply mutation to produce a variant 
                    mutant_child = mutate(child) 
                    mutant_list.append(mutant_child)
                    # evaluate variant
                    mutant_score = run(mutant_child, numberOfRuns, True) # evaluate possible solution
                    score_list.append(mutant_score)
                   
                    # check if new state is greater than goal state
                    if mutant_score >= max_fitness:
                        # add to new population
                        newPopulation.append(mutant_child)
                        stop=True
                    elif loop_index == 10:
                        # termination criteria
                        max_index = score_list.index(max(score_list))
                        chosen_child = mutant_list[max_index]
                        newPopulation.append(chosen_child)
                        stop=True

                    # increment loop index
                    loop_index += 1
                        
        # Keeping best population member
        best_member = [list(pop) for pop,fit in popFitPairs if fit == max_fitness]
        # print('Best Member: ', best_member[0])
        rnd = random.randrange(19)
        newPopulation[rnd] = best_member[0]
        population = copy.deepcopy(newPopulation)

    print('Averages: ', averages)
    print('Bests: ', bests)
    
    ## Plotting averages and bests
    plt.plot(averages, label='average')
    plt.plot(bests, label='best')
    plt.xlabel("time")
    plt.ylabel("score")
    plt.legend()
    plt.show()

# test experiment
def runTest():
    program = np.empty((25,20),dtype=object)
    for xx in range(25):
        for yy in range(20):
            # goes to the right hand side of the screen
            # eventually gets stuck
            program[xx][yy] = Directions.EAST
    # run chooses a direction based on agrid
    # print(program)
    run(program,1,beQuiet=False)


if __name__ == '__main__':
    # runTest()   
    # run_genetic()
    start = datetime.now()
    run_memetic()
    print('Runtime: ', datetime.now() - start)
