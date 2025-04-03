from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(queue='serc',memory='18G',cores=4,walltime='96:00:00',processes=4,interface='ib0',log_directory='slurm_worker_logs')
cluster.adapt(minimum_jobs=75,maximum_jobs=125)

from dask.distributed import Client

client = Client(cluster)
print(client)
"""
START PROGRAM NOW
"""


from ad99 import AlexanderDunkerton1999
from constants import GRAV,C_P,R_DRY
import numpy as np 
import matplotlib.pyplot as plt 
import xarray as xr 
import shutil
import matplotlib.pyplot as plt
import os 
import cartopy.crs as ccrs
import dask.array as da

BFLIM = 1e-4
PATH = "/scratch/users/robcking/gwd_u_era5_2010_2020.zarr"

if __name__ == '__main__':
    print("STARTING AT: ")
    print(client.dashboard_link)
    ##SETUP
    ad99 = AlexanderDunkerton1999(damp_top=True,base_wavelength=30e3)

    GCLOUD_ERA5 = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

    ## DASK

    ds = xr.open_dataset(GCLOUD_ERA5,engine='zarr',chunks={},
        storage_options=dict(token='anon'),)


    ds_subset = ds[['u_component_of_wind','temperature','geopotential']]
    ds_subset = ds_subset.sel(time=slice('2010-01-01','2020-01-01',6))
    ds_subset = ds_subset.isel(level=slice(None,None,-1)).astype(np.float32)
    ds_subset = ds_subset.transpose("time","latitude","longitude","level")


    us = ds_subset.u_component_of_wind
    temps = ds_subset.temperature
    height = ds_subset.geopotential / GRAV

    dTdz = da.gradient(temps,axis=-1)/da.gradient(height,axis=-1)
    Ns2 = GRAV/temps*(dTdz + GRAV/C_P) 
    Ns2 = da.where(Ns2 < BFLIM ** 2, BFLIM **2 , Ns2 )
    Ns = Ns2 ** 0.5 
    Ns = xr.DataArray(Ns,dims=temps.dims,coords=temps.coords)
    rho = (100*temps.level.data.astype(np.float32)[None,None,None,:]/(R_DRY*temps))

    gwd = xr.apply_ufunc(ad99.gwd,us,Ns,height,rho,input_core_dims=[['level'],['level'],['level'],['level']],output_core_dims=[['level']],vectorize=True,dask='parallelized')

    gwd_ds = gwd.to_dataset(name='gwd')
    print(f"Writing to:{PATH}")
    if os.path.exists(PATH):
        shutil.rmtree(PATH)

    # dummy = xr.zeros_like(gwd_ds)
    # dummy.to_zarr(PATH,compute=False)
    
    gwd_ds.to_zarr(PATH)