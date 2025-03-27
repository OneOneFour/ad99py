import xarray as xr
import numpy as np
import dask.array as da
from constants import GRAV,R_DRY,C_P

def dTdz(T:np.ndarray|da.Array,z:np.ndarray | da.Array):
    if isinstance(T,da.Array) or isinstance(z,da.Array):
        return da.gradient(T,axis=-1) / da.gradient(z,axis=-1)
    return np.gradient(T,axis=-1) / np.gradient(z,axis=-1)  
    
def Ns2(T:np.ndarray,z:np.ndarray,bflim=1e-4):
    Ns2unfilter = (GRAV/T)*(dTdz(T,z) + GRAV/C_P)
    return np.where(Ns2unfilter<bflim**2,bflim**2,Ns2unfilter)

def rho(T:np.ndarray|da.Array,p:np.ndarray|da.Array,hectopascal:bool=True):
    pbrd = (99*hectopascal + 1)*p
    if isinstance(T,da.Array):
        pbrd = da.broadcast_to(pbrd,T.shape,chunks=T.chunks)
    else:
        pbrd = np.broadcast_to(pbrd,T.shape)
    return pbrd/(R_DRY*T)
