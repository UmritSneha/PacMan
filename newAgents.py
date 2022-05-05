from os import stat
from game import Agent
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from pacmanAgents import run

class pacmanAgent(Agent):
    '''def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None'''
        
    def setCode(self,codep):
        self.code = codep
        # print('self code: ', self.code)

    def getGhostDetails(self, state, ghost_num, px, py):
        # get x, y position of ghost
        gx,gy = state.getGhostPosition(ghost_num)

        # get ghost angle
        ghostAngle = np.arctan2(gy-py,gx-px)
        
        if ghostAngle < 0.0:
            ghostAngle += 2.0*math.pi

        # calculate distance between ghost and pacman
        ghostDist = math.floor(np.sqrt( (gx-px)**2 + (gy-py)**2 ))

        # set ghost position using angle
        if math.pi/4.0 < ghostAngle <= 3.0*math.pi/3.0:
            ghostPos = "up"
        if 3.0*math.pi/4.0 < ghostAngle <= 5.0*math.pi/3.0:
            ghostPos = "left"
        if 5.0*math.pi/4.0 < ghostAngle <= 7.0*math.pi/3.0:
            ghostPos = "down"
        if 7.0*math.pi/4.0 < ghostAngle <= 2.0*math.pi:
            ghostPos = "right"
        if 0.0 <= ghostAngle <= math.pi/4.0:
            ghostPos = "right"

        return ghostDist, ghostPos
    
    def getAction(self,state):
        # get pacman position
        px,py = state.getPacmanPosition()
        pacman_direction = self.code[px][py]

        # get all legal actions of pacman
        legal = state.getLegalPacmanActions()
        
        # get first ghost position and distance
        ghost1_dist, ghost1_pos = self.getGhostDetails(state, 1, px, py)

        # get second ghost position and distance
        ghost2_dist, ghost2_pos = self.getGhostDetails(state, 2, px, py)

        # check when pacman is close to ghost
        # adopt a greedy search approach
        if ghost1_dist < 4 or ghost2_dist < 4:
            # get successor state for all legal actions
            successors = [(state.generateSuccessor(0, action), action) for action in legal]

            # evaluate successor states
            scores = [(scoreEvaluation(state), action) for state, action in successors]
            
            # get maximum score
            max_score = max(scores)[0]

            # get actions that produce maximum score
            best_actions = [pair[1] for pair in scores if pair[0] == max_score]
            # print('Best Actions: ', best_actions)

            # return random action from the list of best actions
            pacman_direction = random.choice(best_actions)

        # list of positions (x, y) of remaining capsules
        # print('Get capsules: ', state.getCapsules())
        # print('Get Food: ', state.getFood())
        # print(state.hasFood(px, py))

        # check food status of pacman
        # food_status = state.hasFood(px, py)
        # print('food_status: ', food_status)
 
        # generate random action if not legal
        if pacman_direction not in legal:
            pacman_direction = random.choice(legal)

        return pacman_direction

def scoreEvaluation(state):
    return state.getScore()

'''if direction == 'North':
    north+=1
if direction == 'East':
    east+=1
if direction == 'South':
    south+=1
if direction == 'West':
    west+=1
print(north, east, south, west)'''

# set action when close to ghost
'''if ghost1Dist < 4:
    if ghost1Pos == 'up':
        direction = Directions.NORTH
    if ghost1Pos == 'left':
        direction = Directions.EAST
    if ghost1Pos == 'down':
        direction = Directions.SOUTH
    if ghost1Pos == 'right':
        direction = Directions.WEST'''