from functools import wraps
import numpy as np

def make_source_spectrum(source,cw,Bm):
    @wraps(source)
    def source_spectrum(c,c0):
        return source(c,c0,cw,Bm)
    return source_spectrum

def uniform_source(c,c0,cw,Bm):
    # This is actually fine as intermittency will normalize this
    return Bm

def gaussian_source(c,c0,cw,Bm):
    return Bm * np.exp(-np.log(2) * ((c- c0) / cw) ** 2)

def convective_source(c,c0,cw,Bm):
    return Bm*((c-c0)/cw)*np.exp(1 - np.abs((c-c0)/cw))

