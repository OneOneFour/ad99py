import numpy as np
from scipy.integrate import cumulative_trapezoid
from constants import GRAV

"""
Helper functions for integrating momentum flux from MiMA profiles
"""




def mima_gwd_net_flux(gwf,p,from_top=True):
    if from_top:
        return cumulative_trapezoid(gwf/GRAV,x=p,axis=-1,initial=0)
    else:
        return cumulative_trapezoid(gwf[...,::-1]/GRAV,x=p[...,::-1],axis=-1,initial=0)[...,::-1]
    
def mima_gwd_abs_flux(gwf,p,from_top=True):
    if from_top:
        return cumulative_trapezoid(np.abs(gwf)/GRAV,x=p,axis=-1,initial=0)
    else:
        return cumulative_trapezoid(np.abs(gwf[...,::-1])/GRAV,x=p[...,::-1],axis=-1,initial=0)[...,::-1]
    
def mima_gwd_abs_flux_half(gwf,p_half,from_top=True):

    if from_top:
        dp = p_half[...,1:] - p_half[..., :-1]
        int_edge = np.cumsum(np.abs(gwf)/GRAV*dp,axis=-1)
        midpoint = (int_edge[...,1:] + int_edge[..., :-1])*0.5
        first = (int_edge[...,0]/2)[...,None]
        return np.concatenate([first, midpoint], axis=-1)

    else:
        p_half = p_half[...,::-1]
        gwf = gwf[...,::-1]
        dp = p_half[...,1:] - p_half[..., :-1]

        int_edge = np.cumsum(np.abs(gwf)/GRAV*dp,axis=-1)
        midpoint = (int_edge[...,1:] + int_edge[..., :-1])*0.5
        first = (int_edge[...,0]/2)[...,None]
        return np.concatenate([first, midpoint], axis=-1)[...,::-1]

def mima_gwd_ptv_flux(gwf,p,from_top=True):
    gwf_ptv = np.where(gwf>0, gwf, 0)
    if from_top:
        return cumulative_trapezoid(gwf_ptv/GRAV,x=p,axis=-1,initial=0)
    else:
        return cumulative_trapezoid(gwf_ptv[...,::-1]/GRAV,x=p[...,::-1],axis=-1,initial=0)[...,::-1]
    
def mima_gwd_ntv_flux(gwf,p,from_top=True):
    gwf_ntv = np.where(gwf<0, gwf, 0)
    if from_top:
        return cumulative_trapezoid(gwf_ntv/GRAV,x=p,axis=-1,initial=0)
    else:
        return cumulative_trapezoid(gwf_ntv[...,::-1]/GRAV,x=p[...,::-1],axis=-1,initial=0)[...,::-1]