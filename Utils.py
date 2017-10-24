import os
import sys

import traci
from gym.utils import seeding

try:
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME"), "tools"))
    from sumolib import checkBinary  # noqa
except ImportError:
   sys.exit(
       "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")


def initSimulation(visualize=False, seed=123):
    __generateRoutefile(seed)
    if visualize:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    traci.start(
        [sumoBinary, "-c", os.path.join(os.path.dirname(__file__), "data/cross.sumocfg"), "--start", "--quit-on-end",
         "--duration-log.statistics"])
    traci.trafficlights.setPhase("0", 0)


def __generateRoutefile(seed):
    npRandom, seed = seeding.np_random(seed)
    N = 3600  # number of time steps
    # demand per second from different directions
    pWE = 1. / 10
    pEW = 1. / 10
    pNS = .3
    pSN = .3
    with open(os.path.join(os.path.dirname(__file__), "data/cross.rou.xml"), "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>

        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="up" edges="53o 3i 4o 54i" />
        <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
        vehNr = 0
        for i in range(N):
            if npRandom.uniform(low=0., high=1.) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if npRandom.uniform(low=0., high=1.) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if npRandom.uniform(low=0., high=1.) < pNS:
                print('    <vehicle id="down_%i" type="typeWE" route="down" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if npRandom.uniform(low=0., high=1.) < pSN:
                print('    <vehicle id="up_%i" type="typeWE" route="up" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)

