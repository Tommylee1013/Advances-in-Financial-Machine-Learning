import numba as nb
import numpy as np
import FinancialMachineLearning as fml
import pandas as pd
import copyreg, types, multiprocessing as mp

@nb.njit
def func(arr,i):
    col = arr[i]
    mask = np.where(col>0)
    return np.mean(col[mask])

@nb.njit
def njit_getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c = indM.sum(axis=1).reshape(-1,1) # concurrency
    u = np.divide(indM,c) # uniqueness
    avgU = np.zeros(len(u.T)) # avg. uniqueness
    i = 0
    for i in range(len(u.T)):
        avgU[i] = func(u.T,i)
        i += 1
    return avgU

@nb.jit
def jit_seqBootstrap(indM,sLength=None):
    # Generate a sample via sequential bootstrap
    if sLength is None:sLength=indM.shape[1]
    phi=[]
    while len(phi)<sLength:
        avgU=pd.Series()
        for i in indM:
            indM_=indM[phi+[i]] # reduce indM
            avgU.loc[i]=njit_getAvgUniqueness(indM_.values)[-1]
        prob=avgU/avgU.sum() # draw prob
        phi+=[np.random.choice(indM.columns,p=prob)]
    return phi
#------------------------

def split_t1(t1, partitions):
    return np.array_split(t1, partitions)

def mp_func(indM):
    # jit funcs about 2x as fast
    phi = jit_seqBootstrap(indM)
    seqU = njit_getAvgUniqueness(indM[phi].values).mean()
    #phi = seqBootstrap(indM)
    #seqU= getAvgUniqueness(indM[phi])
    return seqU

def main_mp(t1, partitions=100, cpus=8):
    jobs = []
    splits = split_t1(t1, partitions = 100)
    for part_t1 in splits:
        indM = fml.getIndMatrix(part_t1.index, part_t1)
        job = {'func':mp_func,'indM':indM}
        jobs.append(job)
    if cpus==1: out= fml.processJobs_(jobs)
    else: out= fml.processJobs(jobs,numThreads=cpus)
    return pd.DataFrame(out)