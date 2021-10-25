import os
import imcn
from os.path import join
import numpy as np
import jpype as jp


def get_jar_location():

    jar_file_name = "infodynamics.jar"
    jar_location = imcn.__file__
    jar_location = jar_location.replace('__init__.py', '')
    jar_location = join(jar_location, jar_file_name)

    return jar_location


def init_jvm():

    jar_location = get_jar_location()

    if jp.isJVMStarted():
        return
    else:
        jp.startJVM(jp.getDefaultJVMPath(), "-ea",
                    "-Djava.class.path=" + jar_location)


def calc_TE(source, target, num_threads=1):

    init_jvm()
    # jar_location = get_jar_location()
    # jp.startJVM(jp.getDefaultJVMPath(), "-ea",
    #             "-Djava.class.path=" + jar_location)
    calcTEClass = jp.JPackage(
        "infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    calcTE = calcTEClass()
    calcTE.setProperty("NUM_THREADS", str(num_threads))

    # source = coordinates[links[jj][0]][:-1]
    # target = np.diff(coordinates[links[jj][1]])
    calcTE.initialise()
    calcTE.setObservations(source, target)
    te = calcTE.computeAverageLocalOfObservations()

    return te


def calc_MI(source, target, num_threads=1, jar_location=None):

    pass
