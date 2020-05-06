import cv2
import numpy as np


class Player:
    def __init__(self):
        self.cX = 0 #initialisation of creature's X and Y coordinate
        self.cY = 0 #initialisation of creature's X and Y coordinate
        self.done = False #to decide when to stop the game
        self.action = None #an action 0 1 2 3 as up right down left
        self.state = None #combination of coordinate of food and creature and action
        self.nextState = None # state after taking an action
        self.env = np.zeros((480, 640), np.uint8) #the game environment in opencv
        self.reward = 0 #a reward to agent if it played good or bad
        self.im = self.env.copy()
        self.foodX = 0
        self.foodY = 0
        self.spawn_food() #spawns a food at random location
        self.prev_distance = None #distance of previous from creature to food.
        self.score = 0 #score = +1 is the food is eaten

    def spawn(self):#spawns the creature at random location
        self.cX = np.random.randint(low=0, high=640, dtype=np.uint32)
        self.cY = np.random.randint(low=0, high=480, dtype=np.uint32)

    def spawn_food(self):
        print("Spawning food")#spawns a food at random location
        self.foodX = np.random.randint(low=1, high=639, dtype=np.uint32)
        self.foodY = np.random.randint(low=1, high=479, dtype=np.uint32)

    def creature(self, command):#span the creature at 1 step up left down or right as per command
        if command == 0:  # up
            final_point = 0
            if self.cY != final_point:
                self.cY = self.cY - 10
        elif command == 1:  # right
            final_point = 640
            if self.cX != final_point:
                self.cX = self.cX + 10
        elif command == 2:  # down
            final_point = 480
            if self.cY != final_point:
                self.cY = self.cY + 10
        elif command == 3:  # left
            final_point = 0
            if self.cX != final_point:
                self.cX = self.cX - 10

        return self.cX, self.cY

    def run(self): #responsibe to operate the creature 
        self.im = self.env.copy()
        self.state = self.cX, self.cY, self.foodX, self.foodY #create the state of the creature
        #take an action based on condition
        if self.action == 3:
            self.cX, self.cY = self.creature(3)
        if self.action == 1:
            self.cX, self.cY = self.creature(1)
        if self.action == 0:
            self.cX, self.cY = self.creature(0)
        if self.action == 2:
            self.cX, self.cY = self.creature(2)
        self.nextState = self.cX, self.cY, self.foodX, self.foodY #next state is the state after taking an action
        cv2.circle(self.im, (self.cX, self.cY), 5, (255, 255, 255), 1) #draw the creature as a circle
        cv2.circle(self.im, (self.foodX, self.foodY), 5, (255, 255, 255), 5)#draw the food
        cc = np.array([self.cX, self.cY])
        dd = np.array([self.foodX, self.foodY])
        distance = np.linalg.norm(np.subtract(dd, cc)) #distance between food and the creature
        if distance < 12: #if distance is less than 12 it actually hits the food 
            self.spawn_food() #respawn the food 
            self.score = self.score + 1 #give a score
        cv2.putText(self.im, "score " + str(self.score), (20, 30), 1, cv2.FONT_HERSHEY_DUPLEX, (255, 255, 255), 1)
        #if the creature is going far from the food, we give it a negative reward
        if self.prev_distance:
            if distance < self.prev_distance:
                self.reward = 10
            else:
                self.reward = -10
            self.prev_distance = distance
        else:
            self.prev_distance = distance
        if self.cX <= 0 or self.cY <= 0 or self.cX >= 640 or self.cY >= 480: #if the creature hits the wall, we break the game and restart
            self.done = True
            self.reward = -10
        else:
            self.done = False
        cv2.imshow("env", self.im)
        cv2.waitKey(1)
