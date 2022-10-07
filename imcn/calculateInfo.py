import imcn
import jpype as jp
from os.path import join
# import numpy as np


def get_jar_location():
    '''
    Returns the location of the infodynamics.jar file

    Return: str
        location of the infodynamics.jar file

    '''
    jar_file_name = "infodynamics.jar"
    jar_location = imcn.__file__
    jar_location = jar_location.replace('__init__.py', '')
    jar_location = join(jar_location, jar_file_name)

    return jar_location


def init_jvm():
    '''
    initialize the JVM

    '''
    jar_location = get_jar_location()

    if jp.isJVMStarted():
        return
    else:
        jp.startJVM(jp.getDefaultJVMPath(), "-ea",
                    "-Djava.class.path=" + jar_location)


def calc_TE(source, target, num_threads=1, num_surrogates=0):
    '''
    Calculate transfer entropy

    Parameters
    ----------
    source : numpy.ndarray
        source time series
    target : numpy.ndarray
        target time series
    num_threads : int
        number of threads

    Return: float
        transfer entropy [bit]

    '''

    init_jvm()
    calcTEClass = jp.JPackage(
        "infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    calcTE = calcTEClass()
    calcTE.setProperty("NUM_THREADS", str(num_threads))
    calcTE.initialise()
    calcTE.setObservations(source, target)
    te = calcTE.computeAverageLocalOfObservations()
    if num_surrogates > 0:
        NullDist = calcTE.computeSignificance(num_surrogates)
        NullMean = NullDist.getMeanOfDistribution()
        NullStd = NullDist.getStdOfDistribution()

        return te, NullMean, NullStd
    else:
        return te


def calc_MI(source, target, NUM_THREADS=1, k=4, TIME_DIFF=1, num_surrogates=0):
    '''
    calculate mutual information

    Parameters
    ----------
    source : numpy.ndarray
        source time series
    target : numpy.ndarray
        target time series
    NUM_THREADS : int
        number of threads
    k : int
        number of nearest neighbours
    TIME_DIFF : int
        time difference

    '''

    assert((len(source) > 0) and (len(target) > 0))
    init_jvm()

    calcClass = jp.JPackage(
        "infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
    calc = calcClass()
    calc.setProperty("k", str(int(k)))
    calc.setProperty("NUM_THREADS", str(int(NUM_THREADS)))
    calc.setProperty("TIME_DIFF", str(int(TIME_DIFF)))
    calc.initialise()
    calc.setObservations(source.tolist(), target.tolist())
    me = calc.computeAverageLocalOfObservations()

    if num_surrogates > 0:
        NullDist = calc.computeSignificance(num_surrogates)
        NullMean = NullDist.getMeanOfDistribution()
        NullStd = NullDist.getStdOfDistribution()

        return me * 1.4426950408889634, NullMean, NullStd
    else:
        return me * 1.4426950408889634  # np.log2(np.exp(1)) in bits
