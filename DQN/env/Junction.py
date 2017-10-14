from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import traci
import gym
from gym import spaces
import Utils


class Junction(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.__visualize = False
        self.__firstRun = True
        self.__numOfTLightConfig = 2
        self.__roadLength = 510
        self.__discretizationRes = 84
        self.__observationShape = (self.__discretizationRes, self.__discretizationRes)
        self.__discretizationStep = self.__roadLength * 2 / self.__discretizationRes
        centerCoord = (int(self.__discretizationRes / 2), int(self.__discretizationRes / 2))
        self.__downTLightPos = tuple(np.subtract(centerCoord, (0, 1)))
        self.__leftTLightPos = centerCoord
        self.__upTLightPos = tuple(np.subtract(centerCoord, (1, 0)))
        self.__rightTLightPos = tuple(np.subtract(centerCoord, (1, 1)))

        self.action_space = spaces.Discrete(self.__numOfTLightConfig)
        self.observation_space = spaces.Box(0, 1.0, self.__observationShape)

    def _seed(self, seed=None):
        if seed is None:
            self.__seed = 123
        else:
            self.__seed = seed
        return [self.__seed]

    def _step(self, action):
        self.__tLightSwitched = (traci.trafficlights.getRedYellowGreenState("0") == "GrGr" and action == 1) or (traci.trafficlights.getRedYellowGreenState("0") == "rGrG" and action == 0)

        traci.trafficlights.setPhase("0", action)
        traci.simulationStep()

        observation = self.__getObservationMatrix()

        reward = self.__getReward()

        done = traci.simulation.getMinExpectedNumber() == 0
        done = bool(done)

        info = {}

        return observation, reward, done, info

    def _reset(self):
        if not self.__firstRun:
            traci.close()
        else:
            self.__firstRun = False

        Utils.initSimulation(self.__visualize, self.__seed)

        self.__switches = 0
        observation = np.zeros(self.__observationShape)

        return observation

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        traci.close()

    def setVisualization(self, visualize):
        self.__visualize = visualize

    def __getObservationMatrix(self):
        observation = np.zeros(self.__observationShape)

        # Add vehicles location
        vehiclesNames = traci.vehicle.getIDList()
        for vehicle in vehiclesNames:
            pos = traci.vehicle.getPosition(vehicle)
            x = np.math.floor(pos[0] / self.__discretizationStep)
            y = np.math.floor(pos[1] / self.__discretizationStep)
            observation[x][y] = 1.

        # Add traffic light state
        tLightState = traci.trafficlights.getRedYellowGreenState("0")
        observation[self.__downTLightPos[0], self.__downTLightPos[1]] = 0.333 + int(tLightState[0] == 'G') * 0.333
        observation[self.__leftTLightPos[0], self.__leftTLightPos[1]] = 0.333 + int(tLightState[1] == 'G') * 0.333
        observation[self.__upTLightPos[0], self.__upTLightPos[1]] = 0.333 + int(tLightState[2] == 'G') * 0.333
        observation[self.__rightTLightPos[0], self.__rightTLightPos[1]] = 0.333 + int(tLightState[3] == 'G') * 0.333

        return observation

    def __getReward(self):
        teleports = self.__getTeleports()
        if self.__tLightSwitched:
            self.__switches += 1
        delay = self._getDelay()
        waitTime = self.__getWaitTime()

        return -1 * (0.1 * teleports + 0.1 * self.__switches + 0.4 * delay + 0.4 * waitTime)

    def __getTeleports(self):
        return traci.simulation.getStartingTeleportNumber()

    def __getWaitTime(self):
        waitTime = 0.
        vehiclesNames = traci.vehicle.getIDList()
        for vehicle in vehiclesNames:
            waitTime += traci.vehicle.getWaitingTime(vehicle)
        return waitTime

    def _getDelay(self):
        delay = 0
        vehiclesNames = traci.vehicle.getIDList()
        for vehicle in vehiclesNames:
            vMaxSpeed = traci.vehicle.getMaxSpeed(vehicle)
            delay += (vMaxSpeed - traci.vehicle.getSpeed(vehicle)) / vMaxSpeed
        return delay

