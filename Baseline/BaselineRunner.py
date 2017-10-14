from __future__ import absolute_import
from __future__ import print_function

import traci
import numpy as np
import Utils


def runNaive():
    step = 0
    phase = 1
    traci.trafficlights.setPhase("0", phase)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        if step < 20:
            step += 1
        else:
            step = 0
            phase = (phase + 1) % 2
            traci.trafficlights.setPhase("0", phase)
    traci.close()


# main
seed = 123
visualize = False
np.random.seed(seed)
Utils.initSimulation(visualize, seed)
runNaive()
