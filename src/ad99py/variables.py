import xarray as xr
import numpy as np
import dask.array as da
from .constants import GRAV,R_DRY,C_P

def lapserate(T:np.ndarray|da.Array,z:np.ndarray | da.Array):
    if isinstance(T,da.Array) or isinstance(z,da.Array):
        return da.gradient(T,axis=-1) / da.gradient(z,axis=-1)
    return np.gradient(T,axis=-1) / np.gradient(z,axis=-1)  
    
def bouyancy_freq_squared(T:np.ndarray,z:np.ndarray,bflim=5e-3):
    Ns2unfilter = (GRAV/T)*(lapserate(T,z) + GRAV/C_P)
    if isinstance(T,da.Array) or isinstance(z,da.Array):
        return da.where(Ns2unfilter<bflim**2,bflim**2,Ns2unfilter)
    return np.where(Ns2unfilter<bflim**2,bflim**2,Ns2unfilter)

def density(T:np.ndarray|da.Array,p:np.ndarray|da.Array,hectopascal:bool=True):
    pbrd = (99*hectopascal + 1)*p
    if isinstance(T,da.Array):
        pbrd = da.broadcast_to(pbrd,T.shape,chunks=T.chunks)
    else:
        pbrd = np.broadcast_to(pbrd,T.shape)
    return pbrd/(R_DRY*T)

