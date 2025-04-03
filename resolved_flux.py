from windspharm.xarray import VectorWind # Slower than pyspharm but works well
from scipy.integrate import cumulative_trapezoid
from constants import GRAV
import numpy as np 

def diag_omega(wind:VectorWind,phalf):

    dwdp_full = - wind.divergence().data
    # Implicit assume second dimension is pressure. 
    shape = list(dwdp_full.shape)
    shape[1] += 1
    dwdp_half = np.zeros(shape)
    dwdp_half[:,1:-1] = 0.5*(dwdp_full[:,1:] +dwdp_full[:,:-1])
    dwdp_half[:,0] = 1.5*dwdp_full[:,0] - 0.5*dwdp_half[:,1]    
    dwdp_half[:,-1] = 1.5*dwdp_full[:,-1] - 0.5*dwdp_half[:,-2]

    w_half = cumulative_trapezoid(dwdp_half, phalf, axis=1,initial=0)
    w_full = 0.5*(w_half[:,1:] + w_half[:,:-1]) 
    return w_full


def get_resolved_FxFy(ds):
    wind = VectorWind(ds.ucomp, ds.vcomp)  
    omega = ds.omega 
    omega_pert = omega - omega.mean(dim='lon')
    uchi,vchi,upsi,vpsi = wind.helmholtz()
    uchiT21,vchiT21,upsiT21,vpsiT21 = wind.helmholtz(truncation=21)

    u_pert = uchi - uchiT21
    v_pert = vchi - vchiT21
    F_x = -u_pert*omega_pert/GRAV
    F_y = -v_pert*omega_pert/GRAV
    return F_x,F_y

def get_resolved_FxFy_no_omega(ds):
    wind = VectorWind(ds.ucomp, ds.vcomp)
    omega = diag_omega(wind, 100*ds.phalf)
    omega_pert = omega - omega.mean(axis=-1)
    uchi,vchi,upsi,vpsi = wind.helmholtz()
    uchiT21,vchiT21,upsiT21,vpsiT21 = wind.helmholtz(truncation=21)

    u_pert = uchi - uchiT21
    v_pert = vchi - vchiT21 

    F_x = -u_pert*omega_pert/GRAV
    F_y = -v_pert*omega_pert/GRAV    

    return F_x,F_y  
    
