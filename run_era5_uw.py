from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(queue='serc',memory='16GiB',cores=8,processes=8,interface='ib0',walltime='60:00:00')
cluster.scale(jobs=50) 


from dask.distributed import Client

client = Client(cluster)

print(client)

if __name__ == '__main__':
    import xarray as xr 
    import dask.array as da
    from scipy.integrate import cumulative_trapezoid

    ds = xr.open_dataset('/scratch/users/robcking/gwd_u_era5_2010_2020.zarr',chunks={})
    ds = ds.isel(level=slice(None,None,-1))


    uw = da.apply_along_axis(cumulative_trapezoid,-1,ds.gwd.data/9.81,x=100*ds.level.data,initial=0,dtype=ds.gwd.data.dtype,shape=ds.level.shape)
    gwd_ptv = da.where(ds.gwd.data > 0, ds.gwd.data, 0)
    gwd_ntv = da.where(ds.gwd.data < 0, ds.gwd.data, 0)
    uw_ptv = da.apply_along_axis(cumulative_trapezoid,-1,gwd_ptv/9.81,x=100*ds.level.data,initial=0,dtype=ds.gwd.data.dtype,shape=ds.level.shape)
    uw_ntv = da.apply_along_axis(cumulative_trapezoid,-1,gwd_ntv/9.81,x=100*ds.level.data,initial=0,dtype=ds.gwd.data.dtype,shape=ds.level.shape)
    da_uw = xr.DataArray(uw,coords=ds.coords,dims=ds.dims)
    da_uw = da_uw.isel(level=slice(None,None,-1))

    da_uw_ptv = xr.DataArray(uw_ptv,coords=ds.coords,dims=ds.dims)
    da_uw_ptv = da_uw_ptv.isel(level=slice(None,None,-1))

    da_uw_ntv = xr.DataArray(uw_ntv,coords=ds.coords,dims=ds.dims)
    da_uw_ntv = da_uw_ntv.isel(level=slice(None,None,-1))
    ds_uw = xr.Dataset({'uw':da_uw,'uw_ptv':da_uw_ptv,'uw_ntv':da_uw_ntv})
    ds_uw.to_zarr('/scratch/users/robcking/uw_era5_2010_2020.zarr',zarr_format=2)
    print("DONE")
