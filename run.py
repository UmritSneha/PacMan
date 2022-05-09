import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from pacman import *
import ghostAgents
import layout
import textDisplay
import graphicsDisplay
import copy
import numpy as np
from pprint import pprint


## set up the parameters to newGame
timeout = 50

# set layout of game
chosen_layout = "smallClassic"

# minimaxClassic: 8 4
# contestClassic: 19 8
# capsuleClassic: 18 6
# mediumClassic: 19 10
# originalClassic: 27 26
# smallClassic: 19 6

# grid dimension
if chosen_layout == "minimaxClassic":
    gridx = 8
    gridy = 4
if chosen_layout == "contestClassic":
    gridx = 19
    gridy = 8
if chosen_layout == "mediumClassic":
    gridx = 19
    gridy = 10
if chosen_layout == "originalClassic":
    gridx = 27
    gridy = 26
if chosen_layout == "smallClassic":
    gridx = 19
    gridy = 6
if chosen_layout == "capsuleClassic":
    gridx = 18
    gridy = 6

layout=layout.getLayout(chosen_layout)
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
        timeInterval = 0.0001
        textDisplay.SLEEP_TIME = timeInterval
        gameDisplay = graphicsDisplay.PacmanGraphics(1.0, timeInterval)
        rules.quiet = False
    for _ in range(noOfRuns):
        thePacman = pacmanType()
        thePacman.setCode(code, gridx, gridy)
        
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
def run_genetic(popSiz=20, timescale=40, numberOfRuns=3, tournamentSize=5):
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
                fitness.append(run(pp,numberOfRuns, True))
        
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

    # save results in csv file
    genetic_data = {'layout':chosen_layout, 'averages':averages, 'bests':bests}
    genetic_df = pd.DataFrame(genetic_data)
    genetic_df.to_csv('genetic_results.csv', mode='a', header=False, index=False)

def run_memetic(popSize=20, timescale=40, numberOfRuns=3, tournamentSize=5):
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
            if tt < timescale - 1:
                fitness.append(run(pp,numberOfRuns, True))
            else: 
                fitness.append(run(pp,numberOfRuns, False))
        
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
                    # evaluate initial state
                    mutant_score = run(mutant_child, numberOfRuns, True) # evaluate possible solution
                    score_list.append(mutant_score)
                   
                    # check if new state is greater than goal state
                    if mutant_score >= max_fitness:
                        # add to new population
                        newPopulation.append(mutant_child)
                        stop=True
                    elif loop_index == 10:
                        # termination criteria
                        # get child producing better performance
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

    # save results in csv file
    memetic_data = {'layout':chosen_layout, 'averages':averages, 'bests':bests}
    memetic_df = pd.DataFrame(memetic_data)
    memetic_df.to_csv('memetic_results.csv', mode='a', header=False, index=False)

if __name__ == '__main__':
    start = datetime.now()
    # run_genetic()
    run_memetic()
    print('Runtime: ', datetime.now() - start)
