import cv2
import numpy as np


class Player:
    def __init__(self):
        self.cX = 0
        self.cY = 0
        self.done = False
        self.action = None
        self.state = None
        self.nextState = None
        self.env = np.zeros((480, 640), np.uint8)
        self.reward = 0
        self.im = self.env.copy()
        self.foodX = 0
        self.foodY = 0
        self.spawn_food()
        self.prev_distance = None
        self.score = 0

    def spawn(self):
        self.cX = np.random.randint(low=0, high=640, dtype=np.uint32)
        self.cY = np.random.randint(low=0, high=480, dtype=np.uint32)

    def spawn_food(self):
        print("Spawning food")
        self.foodX = np.random.randint(low=1, high=639, dtype=np.uint32)
        self.foodY = np.random.randint(low=1, high=479, dtype=np.uint32)

    def creature(self, command):
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

    def run(self):
        self.im = self.env.copy()
        self.state = self.cX, self.cY, self.foodX, self.foodY
        if self.action == 3:
            self.cX, self.cY = self.creature(3)
        if self.action == 1:
            self.cX, self.cY = self.creature(1)
        if self.action == 0:
            self.cX, self.cY = self.creature(0)
        if self.action == 2:
            self.cX, self.cY = self.creature(2)
        self.nextState = self.cX, self.cY, self.foodX, self.foodY
        cv2.circle(self.im, (self.cX, self.cY), 5, (255, 255, 255), 1)
        cv2.circle(self.im, (self.foodX, self.foodY), 5, (255, 255, 255), 5)
        cc = np.array([self.cX, self.cY])
        dd = np.array([self.foodX, self.foodY])
        distance = np.linalg.norm(np.subtract(dd, cc))
        if distance < 12:
            self.spawn_food()
            self.score = self.score + 1
        cv2.putText(self.im, "score " + str(self.score), (20, 30), 1, cv2.FONT_HERSHEY_DUPLEX, (255, 255, 255), 1)
        if self.prev_distance:
            if distance < self.prev_distance:
                self.reward = 10
            else:
                self.reward = -10
            self.prev_distance = distance
        else:
            self.prev_distance = distance
        if self.cX <= 0 or self.cY <= 0 or self.cX >= 640 or self.cY >= 480:
            self.done = True
            self.reward = -10
        else:
            self.done = False
        cv2.imshow("env", self.im)
        cv2.waitKey(1)
