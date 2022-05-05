# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from socket import timeout
from pacman import *
from game import Agent
import textDisplay, graphicsDisplay
import random
import game, ghostAgents
import matplotlib.pyplot as plt
import util, copy
import numpy as np



class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

class GreedyAgent(Agent):
    "An agent that tries to maximise score at every opportunity"
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)

class GeneticAgent(Agent):
    def setCode(self, codep):
        self.code = codep

    # mutation
    def mutate(self, parentp,numberOfMutations=10):
        parent = copy.deepcopy(parentp)
        for _ in range(numberOfMutations):
            xx = random.randrange(gridx)
            yy = random.randrange(gridy)
            parent[xx][yy] = random.choice(options)
        return parent

    def crossover(self, parent1, parent2):
        child = copy.deepcopy(parent1)
        for xx in range(gridx):
            for yy in range(gridy):
                child[xx][yy] = random.choice([parent1[xx][yy], parent2[xx][yy]])
        return child

    def run_ga(self):
        pop_size = 20
        timescale = 20
        num_of_runs = 2
        tournament_size = 7
        games = []
        # creating random intial population
        population = []
        for _ in range(pop_size):
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
                    fitness.append(run(pp,num_of_runs, True))
                else: 
                    fitness.append(run(pp,num_of_runs, False))

            print("\n******")
            print(fitness)
            averages.append(1000+sum(fitness)/pop_size)
            print("av ",1000+sum(fitness)/pop_size)
            bests.append(1000+max(fitness))
            print("max ",1000+max(fitness))

            pop_fit_pairs = list(zip(population,fitness))
            newPopulation = []
            for _ in range(pop_size-1):
                # crossover the parents
                # selecting first parent
                tournament1 = random.sample(pop_fit_pairs, tournament_size)
                parent1 = max(tournament1,key=lambda x:x[1])[0]

                # selecting second parent
                tournament2 = random.sample(pop_fit_pairs, tournament_size)
                parent2 = max(tournament2,key=lambda x:x[1])[0]

                # apply crossover to generate child
                child = self.crossover(parent1, parent2)

                # mutate child
                mutant_child = self.mutate(child)

                # add to new population
                newPopulation.append(mutant_child)

            # Keeping the best population member
            best_member = [list(pop) for pop,fit in pop_fit_pairs if fit == max(fitness)]
            rnd = random.randrange(gridx)
            newPopulation[rnd] = best_member[0]
            population = copy.deepcopy(newPopulation)

        # print average and best score
        print(averages)
        print(bests)
    
        ## Plotting averages and bests
        plt.plot(averages, label='average')
        plt.plot(bests, label='best')
        plt.xlabel("time")
        plt.ylabel("score")
        plt.legend()
        plt.show()
            
    
    def getAction(self, state):
        px,py = state.getPacmanPosition()
        
        direction = self.code[px][py]
        legal = state.getLegalPacmanActions()

        if direction not in legal:
            direction = random.choice(legal)

        return direction

def scoreEvaluation(state):
    return state.getScore()

# setting game parameters
gridx = 25
gridy = 20
numtraining = 0
timeout = 30
layout=layout.getLayout("mediumClassic")
pacmanType = loadAgent("GeneticAgent", True)
numGhosts = 1
ghosts = [ghostAgents.RandomGhost(i+1) for i in range(numGhosts)]
thresholdDist = 4
catchExceptions=True
options = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]

def run(code, no_of_runs, beQuiet):
    rules = ClassicGameRules(timeout)
    games = []
    if beQuiet:
        game_display = textDisplay.NullGraphics()
        rules.quiet = True
    else:
        timeInterval = 0.001
        textDisplay.SLEEP_TIME = timeInterval
        game_display = graphicsDisplay.PacmanGraphics(1.0, timeInterval)
        rules.quiet = False
    for _ in range(no_of_runs):
        thePacman = pacmanType()
        thePacman.setCode(code)
        
        game = rules.newGame( layout, thePacman, ghosts, game_display, \
                          beQuiet, catchExceptions )
        game.run()
        games.append(game)
    scores = [game.state.getScore() for game in games]
    return sum(scores) / float(len(scores))


