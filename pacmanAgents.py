from game import Agent
import random
import numpy as np
import math


class pacmanAgent(Agent):
        
    def setCode(self,codep, gridx, gridy):
        self.code = codep
        self.gridx = gridx
        self.gridy = gridy
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
        # check food status of pacman
        food_matrix = state.getFood()
        for xx in range(self.gridx):
            for yy in range(self.gridy):
                if food_matrix[xx][yy] == True:
                    # calcuate euclidean distance
                    food_dist = self.euclideanDistance(xx,yy,px,py)
                    return food_dist, xx, yy

    def getCapsuleDistance(self, state, px, py, capsule_list):
        # list of positions (x, y) of remaining capsule
        for each in capsule_list:
            distance = self.euclideanDistance(each[0], each[1], px, py)
            # print(distance)
            # distance = ((each[0] - px)**2 + (each[1]-py)**2)**0.5
            return distance, each[0], each[1]

    def getAction(self,state):
        # get pacman position
        px,py = state.getPacmanPosition()
        pacman_direction = self.code[px][py]

        # get all legal actions of pacman
        legal = state.getLegalPacmanActions()

        # food actions
        food_dist, fx, fy = self.getFoodAction(state, px, py)
        food_count = state.getNumFood()
        if food_count > 0 and food_dist < 2.0:
            px = fx
            py = fy
        # print('x: ', x)
        # print('y: ', y)


        # capsule actions
        '''capsule_list =  state.getCapsules()
        if capsule_list:
            capsule_dist, cx, cy = self.getCapsuleDistance(state, px, py, capsule_list)
            if capsule_dist < 5.0:
                px = cx
                py = cy'''

        # get first ghost position and distance
        ghost1_dist, ghost1_pos = self.getGhostDetails(state, 1, px, py)

        # get second ghost position and distance
        ghost2_dist, ghost2_pos = self.getGhostDetails(state, 2, px, py)

        # check when pacman is close to ghost
        # adopt a greedy search approach
        if ghost1_dist < 4 or ghost2_dist < 4:

            # check capsule status
            # capsule actions
            capsule_list =  state.getCapsules()
            if capsule_list:
                capsule_dist, cx, cy = self.getCapsuleDistance(state, px, py, capsule_list)
                if capsule_dist < 2.0:
                    px = cx
                    py = cy
            else:
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

