from os import stat
from game import Agent
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from pacmanAgents import run
from searchAgents import euclideanHeuristic

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

    def euclideanDistance(self, x1, y1, x2, y2):
        return ((x1 - x2)**2 + (y1-y2)**2)**0.5
        
    def getFoodAction(self, state, px, py):
        # print(state.getFood())
        # print(state.hasFood(px, py))
        
        # check food status of pacman
        food_matrix = state.getFood()
        # print(food_matrix)
        # print('\n')
        for xx in range(19):
            for yy in range(8):
                # print(food_matrix[xx][yy])
                if food_matrix[xx][yy] == True:
                    # calcuate distance
                    food_dist = self.euclideanDistance(xx,yy,px,py)
                    return food_dist, xx, yy

    def getCapsuleDistance(self, state, px, py):
        # list of positions (x, y) of remaining capsules
        
        capsule_list =  state.getCapsules()
        if capsule_list:
            for each in capsule_list:
                distance = ((each[0] - px)**2 + (each[1]-py)**2)**0.5
                return distance, each[0], each[1]
    
    def getAction(self,state):
        # get pacman position
        px,py = state.getPacmanPosition()
        pacman_direction = self.code[px][py]

        # get all legal actions of pacman
        legal = state.getLegalPacmanActions()

        # food actions
        food_dist, fx, fy = self.getFoodAction(state, px, py)
        if food_dist < 2.0:
            px = fx
            py = fy
        # print('x: ', x)
        # print('y: ', y)


        # capsule actions
        capsule_dist, cx, cy = self.getCapsuleDistance(state, px, py)
        if capsule_dist < 5.0:
            px = cx
            py = cy

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
        direction = Directions.WEST
        
'''