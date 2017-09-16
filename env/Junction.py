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


class Junction(gym.Env):

    def __init__(self):
        self.initSimulation()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.array([0]), np.array([3600]))

    def _seed(self, seed=None):
        pass

    def _step(self, action):
        traci.trafficlights.setPhase("0", action)
        traci.simulationStep()

        observation = np.array([traci.vehicle.getIDCount()])

        reward = -1 * traci.vehicle.getIDCount()

        done = traci.simulation.getMinExpectedNumber() == 0
        done = bool(done)

        info = {}

        return observation, reward, done, info

    def _reset(self):
        traci.close()
        self.initSimulation()

        observation = np.array([traci.vehicle.getIDCount()])
        return observation

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        traci.close()

    def initSimulation(self):
        sumoBinary = checkBinary('sumo')
        self.generate_routefile()
        traci.start([sumoBinary, "-c", os.path.join(os.path.dirname(__file__), "data/cross.sumocfg")])
        traci.trafficlights.setPhase("0", 2)

    def generate_routefile(self):
        random.seed(42)  # make tests reproducible
        N = 3600  # number of time steps
        # demand per second from different directions
        pWE = 1. / 10
        pEW = 1. / 11
        pNS = 1. / 30
        with open(os.path.join(os.path.dirname(__file__), "data/cross.rou.xml"), "w") as routes:
            print("""<routes>
            <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
            <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>

            <route id="right" edges="51o 1i 2o 52i" />
            <route id="left" edges="52o 2i 1o 51i" />
            <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
            lastVeh = 0
            vehNr = 0
            for i in range(N):
                if random.uniform(0, 1) < pWE:
                    print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pEW:
                    print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pNS:
                    print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
            print("</routes>", file=routes)