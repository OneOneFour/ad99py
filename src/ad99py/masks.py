from typing import Optional
import xarray as xr

DEFAULT_MASK_PATH = 'data/loon_masks.nc'
def load_mask(path:Optional[str]=None,recentering:bool=True)->xr.Dataset:
    if path is None:
        path = DEFAULT_MASK_PATH
    ds_mask = xr.open_dataset(path)
    if recentering:
        # map longitudes to be in range 0-360
        ds_mask['lon']= (ds_mask.lon + 360)%360
    return ds_mask


def mask_dataset(ds:xr.Dataset,mask:Optional[xr.Dataset]=None):
    if mask is None:
        mask = load_mask()
    if 'latitude' in ds.variables and 'longitude' in ds.variables:
        # we assume either/or for 'latitude'/'longitude' or 'lat'/'lon
        mask = mask.rename(lat='latitude',lon='longitude')
        interp_mask = mask.interp(latitude=ds.latitude,longitude=ds.longitude,method='nearest')
        total_mask = sum(interp_mask[d] for d in interp_mask.data_vars)
        masked = ds.where(total_mask).stack(points=['latitude','longitude']).dropna('points',how='all')
    else:
        interp_mask = mask.interp(lat=ds.lat,lon=ds.lon,method='nearest')
        total_mask = sum(interp_mask[d] for d in interp_mask.data_vars)
        masked = ds.where(total_mask).stack(points=['lat','lon']).dropna('points',how='all')
    masked = masked.reset_index('points')
    return masked 
   
