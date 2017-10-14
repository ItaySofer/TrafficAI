from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import numpy as np
import random

try:
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME"), "tools"))
    from sumolib import checkBinary  # noqa
except ImportError:
   sys.exit(
       "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci
import gym
from gym import spaces
from gym.utils import seeding


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
        self.__maxSpeed = 16.67

        self.action_space = spaces.Discrete(self.__numOfTLightConfig)
        self.observation_space = spaces.Box(0, 1.0, self.__observationShape)

    def _seed(self, seed=None):
        self.npRandom, seed = seeding.np_random(seed)
        return [seed]

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
        self.__initSimulation()

        self.__switches = 0
        observation = np.zeros(self.__observationShape)

        return observation

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        traci.close()

    def __initSimulation(self):
        self.__generateRoutefile()
        if self.__visualize:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')
        traci.start([sumoBinary, "-c", os.path.join(os.path.dirname(__file__), "data/cross.sumocfg"), "--start", "--quit-on-end"])
        traci.trafficlights.setPhase("0", self.__numOfTLightConfig - 1)

    def __generateRoutefile(self):
        N = 3600  # number of time steps
        # demand per second from different directions
        pWE = 1. / 10
        pEW = 1. / 10
        pNS = 1. / 10
        pSN = 1. / 10
        with open(os.path.join(os.path.dirname(__file__), "data/cross.rou.xml"), "w") as routes:
            print("""<routes>
            <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed=\"""" + str(self.__maxSpeed) + """\" guiShape="passenger"/>

            <route id="right" edges="51o 1i 2o 52i" />
            <route id="left" edges="52o 2i 1o 51i" />
            <route id="up" edges="53o 3i 4o 54i" />
            <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
            vehNr = 0
            for i in range(N):
                if self.npRandom.uniform(low=-0., high=1.) < pWE:
                    print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if self.npRandom.uniform(low=-0., high=1.) < pEW:
                    print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if self.npRandom.uniform(low=-0., high=1.) < pNS:
                    print('    <vehicle id="down_%i" type="typeWE" route="down" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if self.npRandom.uniform(low=-0., high=1.) < pSN:
                    print('    <vehicle id="up_%i" type="typeWE" route="up" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
            print("</routes>", file=routes)

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
            delay += (self.__maxSpeed - traci.vehicle.getSpeed(vehicle)) / self.__maxSpeed
        return delay

