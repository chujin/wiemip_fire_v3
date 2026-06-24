
## $source hgpy/bin/activate

import requests
import os
from urllib.parse import urlparse
import gzip
import shutil
import xarray as xr
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import math
import rioxarray as rxr
import calendar
import wget
import random
from cftime import DatetimeNoLeap


import warnings
warnings.filterwarnings("ignore")


indir = '/Users/hgenet/Documents/WEIMIP/data/raw/fire'
mod = 'GFDL-ESM4'
mod = 'IPSL-CM6A-LR'
mod = 'stable'
mod = 'UKESM1-0-LL'

outtemdir = '/Users/hgenet/Documents/WEIMIP/data/processed/TEMinputs'





### Read in the data
ds = xr.open_dataset(os.path.join(indir,mod + '_monthly_prob_Native_combined_rescaled.nc'), decode_coords="all")
ds = ds.to_dataframe()
ds.reset_index(inplace=True)
ds['pred_prob'] = ds['pred_prob'].fillna(0)
ds['pred_prob_no_burn'] = 1 - ds['pred_prob']
ds['Final_Choice'] = 0
choices = [1, 0]
noburn = ds.groupby(['lat','lon'])['pred_prob'].sum().to_frame()
noburn.reset_index(inplace=True)
### Burn pixels based on probability
for lat, lon in zip(ds[['lat','lon']].drop_duplicates()['lat'], ds[['lat','lon']].drop_duplicates()['lon']):
    if noburn[(noburn['lat'] == lat) & (noburn['lon'] == lon)]['pred_prob'].values[0] > 0.0 :
        print(noburn[(noburn['lat'] == lat) & (noburn['lon'] == lon)]['pred_prob'])
        ds.loc[(ds['lat'] == lat) & (ds['lon'] == lon), 'Final_Choice'] = [np.random.choice(choices, p=probs) for probs in ds[(ds['lat'] == lat) & (ds['lon'] == lon)][['pred_prob', 'pred_prob_no_burn']].values]

ds.to_csv(os.path.join(indir,mod + '_burn.csv'), index=False)


### Summarize by pixel 
ds = pd.read_csv(os.path.join(indir,mod + '_burn.csv'))
reburn = ds.groupby(['lat','lon'])['Final_Choice'].sum().to_frame()
reburn.reset_index(inplace=True)
reburn['Final_Choice'].max()
### Extract the year into a new column
ds['date'] = pd.to_datetime(ds['time'])
ds['year'] = ds['date'].dt.year 
### Summarize by pixel and year
reburn_yr = ds.groupby(['lat','lon','year'])['Final_Choice'].sum().to_frame()
reburn_yr.reset_index(inplace=True)
reburn_yr['Final_Choice'].max()
### Randomly select one burn per year
ds1burn = ds[ds['Final_Choice'] == 1].groupby(['lat','lon','year']).sample(n=1, random_state=42)
### Evaluate reburn frequencies
ds1burn_sorted = ds1burn.sort_values(by=['lat','lon','year'], ascending=[True, True, True])
ds1burn_sorted['date_lag1'] = ds1burn_sorted.groupby(['lat','lon'])['date'].shift(1)
ds1burn_sorted['year_lag1'] = ds1burn_sorted.groupby(['lat','lon'])['year'].shift(1)
ds1burn_sorted['delta_year'] = ds1burn_sorted['year'] - ds1burn_sorted['year_lag1']
ds1burn_sorted['delta_year'].max()
### Summarize by pixel and year
reburn_yr1 = ds1burn.groupby(['lat','lon'])['Final_Choice'].sum().to_frame()
reburn_yr1.reset_index(inplace=True)
reburn_yr1['Final_Choice'].max()
### Select burns that are at least 30 years apart 
ds1burn_select = ds1burn_sorted[(ds1burn_sorted['delta_year'] >= 30) | (ds1burn_sorted['date_lag1'].isna())]
tmp = ds1burn_select[['lat','lon','year']]
all_duplicates = tmp[tmp.duplicated(keep=False)]
b1 = ds1burn_select[['date','lat','lon','pred_prob','Final_Choice']]
b2 = ds1burn_select[['date_lag1','lat','lon','pred_prob','Final_Choice']]
b2 = b2.rename(columns={"date_lag1": "date"})
b2 = b2.dropna(subset=['date'])
burn = pd.concat([b1, b2], ignore_index=True)
### Remove duplicates
tmp = burn[['lat','lon','date']]
all_duplicates = tmp[tmp.duplicated(keep=False)]
burn = burn.drop_duplicates(subset=['lat','lon','date'], keep='first')
tmp = burn[['lat','lon','date']]
all_duplicates = tmp[tmp.duplicated(keep=False)]
### grab random day in a month and get day of burn
month = pd.DataFrame({"month": list(range(1,13)), "monthlength": [31,28,31,30,31,30,31,31,30,31,30,31]})
burn['year'] = [date.year for date in burn['date']]
burn['month'] = [date.month for date in burn['date']]
burn = pd.merge(burn,month,how='left',on=['month'])
burn['day'] = burn.apply(lambda row: np.random.choice(range(row['monthlength'])), axis=1)
burn['CF_Date_burn'] = burn.apply(lambda row: DatetimeNoLeap(int(row['year']), int(row['month']), int(row['day']+1)), axis=1)
burn['date_burn'] = pd.to_datetime(dict(year=burn.year, month=burn.month, day=burn.day+1))
burn['day_of_year'] = burn['date_burn'].dt.dayofyear
### Merge to the whole timeseries
dsy = ds[['lat','lon','date']]
dsy['month'] = [date.month for date in dsy['date']]
dsy = dsy[dsy['month']==1]
dsy['year'] = [date.year for date in dsy['date']]
dsy = pd.merge(burn[['lat','lon','year','day_of_year']],dsy[['lat','lon','year']],how='right',on=['lat','lon','year'])
dsy['exp_burn_mask'] = np.where(dsy['day_of_year'] > 0, 1, 0)
dsy['exp_jday_of_burn'] =  np.where(dsy['day_of_year'] > 0, dsy['day_of_year'], 0)
dsy['exp_fire_severity'] = 0
dsy['exp_area_of_burn'] = 0
### Format TEM outputs
dsy.reset_index(inplace=True)
dsy = dsy.drop(['index'],axis=1)
dsy = dsy.drop(['day_of_year'],axis=1)
dsy['time'] = dsy.apply(lambda row: DatetimeNoLeap(int(row['year']), int(1), int(1)), axis=1)
dsy['Y'] = dsy['lat']
dsy['X'] = dsy['lon']
dsy = dsy.drop(['year'],axis=1)
dsy = dsy.sort_values(by=['time','Y','X'])
# convert dataframe to xarray
nc = dsy.set_index(['time', 'Y', 'X']).to_xarray()
nc['lat'] = nc['lat'].astype(np.double)
nc['lon'] = nc['lon'].astype(np.double)
nc['exp_burn_mask'] = nc['exp_burn_mask'].astype(np.intc)
nc['exp_jday_of_burn'] = nc['exp_jday_of_burn'].astype(np.intc)
nc['exp_fire_severity'] = nc['exp_fire_severity'].astype(np.intc)
nc['exp_area_of_burn'] = nc['exp_area_of_burn'].astype(np.intc)
nc['lat'].attrs={'standard_name':'latitude','units':'degree_north'}
nc['lon'].attrs={'standard_name':'longitude','units':'degree_east'}
nc['Y'].attrs={'standard_name':'projection_y_coordinate','long_name':'y coordinate of projection','units':'m'}
nc['X'].attrs={'standard_name':'projection_x_coordinate','long_name':'x coordinate of projection','units':'m'}
nc['exp_burn_mask'].attrs={'standard_name':'exp_burn_mask','units':'','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
nc['exp_jday_of_burn'].attrs={'standard_name':'exp_jday_of_burn','units':'doy','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
nc['exp_fire_severity'].attrs={'standard_name':'exp_fire_severity','units':'','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
nc['exp_area_of_burn'].attrs={'standard_name':'exp_area_of_burn','units':'km2','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
nc.to_netcdf(os.path.join(outtemdir,'explicit-fire_' + mod + '.nc'),unlimited_dims='time')






































monthlist = list(range(12))
monthlengthlist = [31,28,31,30,31,30,31,31,30,31,30,31]
     
### Expend the spatial extent of the dataset
new_lon = np.arange(5, 26, 1) # Define your new steps
new_lat = np.arange(-10, 11, 1)
ds_expanded = ds.reindex(lon=new_lon, lat=new_lat, method=None, fill_value=np.nan)

# 4. Save the expanded dataset to a new NetCDF file
ds_expanded.to_netcdf("your_data_expanded.nc")



df[['Col1', 'Col2']].drop_duplicates()

#####   SPECIFICATIONS   #####

### Climate input directory path
indir = '/Users/hgenet/Documents/WEIMIP/data/raw/1pctCO2'
#indir = '/Volumes/5TI/WEIMIP/data/raw/1pctCO2'
modlist = os.listdir(indir)
modlist.remove(".DS_Store")
#modlist.remove("ndep")
modlist.remove("co2")

### EPSG for climate dataset
clmt_crs = 'EPSG:4326'

### output directory
outdir = '/Users/hgenet/Documents/WEIMIP/data/processed/climate'
os.makedirs(outdir, exist_ok=True)
for subdir in modlist:
    print(subdir)
    os.makedirs(os.path.join(outdir,subdir), exist_ok=True)

### Climate variable list
varlist=['tmp','pre','dswrf','spfh','pres']
# List of variables that are averaged over time
varlist_mean=['tmp','spfh','pres','dswrf']
# List of variables that are summed over time
varlist_sum=['pre']


### Path to PPP mask
mask_path = '/Volumes/5TIV/PROCESSED/MASK/aoi_5k_buff_6931_2_0.tiff'
## Spatial output directory
outmskdir = '/Users/hgenet/Documents/WEIMIP/data/processed/mask'
os.makedirs(outmskdir, exist_ok=True)


### Path to PPP ancillary data
anc_path = '/Volumes/5TIV/PROCESSED/'
## Spatial output directory
outancdir = '/Users/hgenet/Documents/WEIMIP/data/processed/ancillary'
os.makedirs(outancdir, exist_ok=True)
## Topo files
topovar = ['aspect','elevation','slope','tpi','drainage']
topoinfile = ['aspect_4k_6931_mask.tif','elevation_4k_6931_mask.tif','slope_4k_6931_mask.tif','tpi_4k_6931_mask.tif','drainage.tif']
## Text files
textvar = ['sand','silt','clay']
textinfile = ['sand_6931_gf.tif','silt_6931_gf.tif','clay_6931_gf.tif']
### URL to vegetation map
vegurl = 'https://zenodo.org/records/17968808/files/HybridLandCover_1km_V2.tif?download=1'
outvegdir = '/Users/hgenet/Documents/WEIMIP/data/processed/vegetation'


### Path to final TEM inputs
outtemdir = '/Users/hgenet/Documents/WEIMIP/data/processed/TEMinputs'
os.makedirs(outtemdir, exist_ok=True)



#####   GET THE EXTENT OF THE MASK   #####

print('GET THE EXTENT OF OUR MASK')
### Re-project the mask to the epsg used for CRU-JRA:
with rasterio.open(mask_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, clmt_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': clmt_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    with rasterio.open(os.path.join(outmskdir,'aoi_5k_buff_4326_2_0.tiff'), 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=clmt_crs,
                resampling=Resampling.nearest)

### Compute bounds
with rasterio.open(os.path.join(outmskdir,'aoi_5k_buff_4326_2_0.tiff')) as src:
    # Get the bounding box object
    bounds = src.bounds

### Convert to netcdf file
rds = rxr.open_rasterio(os.path.join(outmskdir,'aoi_5k_buff_4326_2_0.tiff'))
rds.to_netcdf(os.path.join(outmskdir,'aoi_5k_buff_4326_2_0.nc'))
msk = rxr.open_rasterio(os.path.join(outmskdir,'aoi_5k_buff_4326_2_0.nc'))
msk.rio.crs







#####   CREATE RUN-MASK FILE   #####

### Get spatial info from climate grid for every model
mod='stable'
print('Create a grid file for ', mod, '...')
var='tmp'
y=1850
if mod == 'stable':
    fn = mod + '.' + var + '.' + str(y) + '_6hr.noleap.nc'
else:
    fn = mod + '_clim3_50perc_1pctCO2.' + var + '.' + str(y) + '_6hr.noleap.nc'
ds = xr.open_dataset(os.path.join(indir,mod,'05deg',fn), decode_coords="all")
ds = ds.assign(cnst=(['lat','lon'],np.full((ds['lat'].values.shape[0],ds['lon'].values.shape[0]),1)))
ds = ds.drop_vars([var])
ds = ds.drop_dims(['time'])
ds.to_netcdf(os.path.join(outmskdir,mod+'_grid.nc'))
print('  extract spatial information from the grid...')
ds = rxr.open_rasterio(os.path.join(outmskdir,mod+'_grid.nc'))
modgrid_res = ds.rio.resolution()
modgrid_bounds = ds.rio.bounds()
modgrid_crs = (ds.rio.crs)
ds.rio.write_crs("epsg:4326", inplace=True)
modgrid_crs = (ds.rio.crs)
## Create a mask with the grid's spatial information
print('  create a mask from the grid...')
veg1k = rxr.open_rasterio(os.path.join(outvegdir,'ecotype_all.tif'))
da = veg1k.where(veg1k == 0, 1).where(veg1k > 0, 0)
da.rio.to_raster(os.path.join(outvegdir,'mask.tif'))
os.system('gdalwarp -overwrite -of netCDF -r sum -s_srs epsg:' + str(dacrs) + ' -t_srs ' + clmt_crs + ' -tr ' + str(modgrid_res[0]) + ' ' + str(modgrid_res[1]) + ' -te ' + str(modgrid_bounds[0]) + ' ' + str(math.floor(bounds.bottom*2)/2) + ' ' + str(modgrid_bounds[2]) + ' ' + str(modgrid_bounds[3]) + ' ' + os.path.join(outvegdir,'mask.tif') + ' ' + os.path.join(outvegdir,'mask_upscaled.nc'))
ds = xr.open_dataset(os.path.join(outvegdir,'mask_upscaled.nc'), decode_coords="all")
ds = ds.fillna(0)
arr=ds['Band1'].values
print('  create a mask variable...')
ds['pct'] = 100*ds['Band1']/ds['Band1'].max()
ds['msk'] = xr.where(ds["Band1"] > 0, 1, 0)
#ds['msk'].values[80,100]
print('  reformate to TEM standards...')
ds = ds.rename({'lat':'Y','lon':'X'})
lon_2d = np.repeat(np.array([ds['X'].values]), ds['Y'].values.shape[0], axis=0)
lat_2d = np.repeat(ds['Y'].values.reshape(-1, 1), ds['X'].values.shape[0], axis=1)
ds = ds.assign(lat=(['Y','X'],lat_2d))
ds = ds.assign(lon=(['Y','X'],lon_2d))
ds = ds.rename({'Band1':'land_km2'})
ds = ds.rename({'msk':'run'})
ds = ds.drop_vars(['crs'])
ds['lat'] = ds['lat'].astype(np.single)
ds['lon'] = ds['lon'].astype(np.single)
ds['X'] = ds['X'].astype(np.single)
ds['Y'] = ds['Y'].astype(np.single)
ds['run'] = ds['run'].astype(np.intc)
ds['pct'] = ds['pct'].astype(np.single)
ds['lat'].attrs={'standard_name':'latitude','units':'degree_north','_FillValue': 1.e+20}
ds['lon'].attrs={'standard_name':'longitude','units':'degree_east','_FillValue': 1.e+20}
ds['Y'].attrs={'standard_name':'degree_north','long_name':'y coordinate of projection','units':'degree','_FillValue': 1.e+20}
ds['X'].attrs={'standard_name':'degree_east','long_name':'x coordinate of projection','units':'degree','_FillValue': 1.e+20}
ds['run'].attrs={'standard_name':'mask','units':'','grid_mapping':'latitude_longitude','_FillValue': -9999}
ds['pct'].attrs={'standard_name':'percent area of land','units':'percent','grid_mapping':'latitude_longitude','_FillValue': -9999}
ds['land_km2'].attrs={'standard_name':'land area in km2','units':'km2','grid_mapping':'latitude_longitude','_FillValue': -9999}
ds.to_netcdf(os.path.join(outtemdir,'run-mask2.nc'))
## Create topo and drainage datasets with the grid's spatial information
print('  create topo and drainage datasets ...')
dss = None
for i in range(len(topovar)):
    var=topovar[i]
    infile=topoinfile[i]
    print('   reproject and resample for ', var)
    rst = os.path.join(anc_path,'TOPO',infile) 
    with rasterio.open(rst) as dst:
        in_crs = dst.crs.to_epsg()
    os.system('gdalwarp -overwrite -of netCDF -r average -s_srs epsg:' + str(in_crs) + ' -t_srs ' + clmt_crs + ' -tr ' + str(modgrid_res[0]) + ' ' + str(modgrid_res[1]) + ' -te ' + str(modgrid_bounds[0]) + ' ' + str(math.floor(bounds.bottom*2)/2) + ' ' + str(modgrid_bounds[2]) + ' ' + str(modgrid_bounds[3]) + ' ' + os.path.join(anc_path,'TOPO',infile) + ' ' + os.path.join(outancdir,var + '_' + mod + '.nc'))
    ds = xr.open_dataset(os.path.join(outancdir,var + '_' + mod + '.nc'), decode_coords="all")
    ds = ds.rename({'Band1':var})
    if dss is None:
        dss = ds
    else:
        dss = xr.merge([dss, ds])
dss = dss.rename({'lat':'Y','lon':'X'})
msk = xr.open_dataset(os.path.join(outtemdir,'run-mask2.nc'))
dss = xr.merge([dss, msk])
dss = dss.drop_vars(['pct','run'])
dss = dss.rename({'drainage':'drainage_class'})
dss['X'] = dss['X'].astype(np.single)
dss['Y'] = dss['Y'].astype(np.single)
dss['lat'] = dss['lat'].astype(np.single)
dss['lon'] = dss['lon'].astype(np.single)
dss['slope'] = dss['slope'].astype(np.double)
dss['aspect'] = dss['aspect'].astype(np.double)
dss['elevation'] = dss['elevation'].astype(np.double)
dss['tpi'] = dss['tpi'].astype(np.double)
dss['drainage_class'] = (("Y","X"),np.where(dss['drainage_class']>0.5,1,0))
dss['drainage_class'] = dss['drainage_class'].astype(np.int_)
dss['lat'].attrs={'standard_name':'latitude','units':'degree_north','_FillValue': 1.e+20}
dss['lon'].attrs={'standard_name':'longitude','units':'degree_east','_FillValue': 1.e+20}
dss['Y'].attrs={'standard_name':'degree_north','long_name':'y coordinate of projection','units':'degree','_FillValue': 1.e+20}
dss['X'].attrs={'standard_name':'degree_east','long_name':'x coordinate of projection','units':'degree','_FillValue': 1.e+20}
dss['aspect'].attrs={'standard_name':'aspect','units':'degree','grid_mapping':'latitude_longitude','_FillValue': -999.0}
dss['elevation'].attrs={'standard_name':'elevation','units':'m','grid_mapping':'latitude_longitude','_FillValue': -999.0}
dss['slope'].attrs={'standard_name':'slope','units':'degree','grid_mapping':'latitude_longitude','_FillValue': -999.0}
dss['tpi'].attrs={'standard_name':'topographic position index','units':'','grid_mapping':'latitude_longitude','_FillValue': -999.0}
dss['drainage_class'].attrs={'standard_name':'drainage_class','units':'yrs','grid_mapping':'latitude_longitude','_FillValue': -999.0}
dss.drop(['crs'])
dss.drop_vars(['aspect','elevation','slope','tpi']).to_netcdf(os.path.join(outtemdir,'drainage.nc'))
dss.drop_vars(['drainage_class']).to_netcdf(os.path.join(outtemdir,'topo.nc'))
## Create texture dataset with the grid's spatial information
print('  create topo and drainage datasets ...')
dss = None
for i in range(len(textvar)):
    var=textvar[i]
    infile=textinfile[i]
    print('   reproject and resample for ', var)
    rst = os.path.join(anc_path,'TEXTURE',infile) 
    with rasterio.open(rst) as dst:
        in_crs = dst.crs.to_epsg()
    in_crs = 6931
    os.system('gdalwarp -overwrite -of netCDF -r average -s_srs epsg:' + str(in_crs) + ' -t_srs ' + clmt_crs + ' -tr ' + str(modgrid_res[0]) + ' ' + str(modgrid_res[1]) + ' -te ' + str(modgrid_bounds[0]) + ' ' + str(math.floor(bounds.bottom*2)/2) + ' ' + str(modgrid_bounds[2]) + ' ' + str(modgrid_bounds[3]) + ' ' + os.path.join(anc_path,'TEXTURE',infile) + ' ' + os.path.join(outancdir,var + '.nc'))
    ds = xr.open_dataset(os.path.join(outancdir,var + '.nc'), decode_coords="all")
    ds = ds.rename({'Band1':var})
    if dss is None:
        dss = ds
    else:
        dss = xr.merge([dss, ds])
dss = dss.rename({'lat':'Y','lon':'X'})
msk = xr.open_dataset(os.path.join(outtemdir,'run-mask2.nc'))
dss = xr.merge([dss, msk])
dss = dss.drop_vars(['pct','run'])
dss = dss.rename({'sand':'pct_sand'})
dss = dss.rename({'silt':'pct_silt'})
dss = dss.rename({'clay':'pct_clay'})
dss['pct_silt'] = 100-(dss['pct_sand'] + dss['pct_clay'])
dss['X'] = dss['X'].astype(np.single)
dss['Y'] = dss['Y'].astype(np.single)
dss['lat'] = dss['lat'].astype(np.single)
dss['lon'] = dss['lon'].astype(np.single)
dss['pct_sand'] = dss['pct_sand'].astype(np.single)
dss['pct_silt'] = dss['pct_silt'].astype(np.single)
dss['pct_clay'] = dss['pct_clay'].astype(np.single)
dss['lat'].attrs={'standard_name':'latitude','units':'degree_north','_FillValue': 1.e+20}
dss['lon'].attrs={'standard_name':'longitude','units':'degree_east','_FillValue': 1.e+20}
dss['Y'].attrs={'standard_name':'degree_north','long_name':'y coordinate of projection','units':'degree','_FillValue': 1.e+20}
dss['X'].attrs={'standard_name':'degree_east','long_name':'x coordinate of projection','units':'degree','_FillValue': 1.e+20}
dss['pct_sand'].attrs={'standard_name':'sand','units':'pct','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
dss['pct_silt'].attrs={'standard_name':'silt','units':'pct','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
dss['pct_clay'].attrs={'standard_name':'clay','units':'pct','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
dss.drop(['crs']).to_netcdf(os.path.join(outtemdir,'texture.nc'))
## Create a vegetation dataset with the grid's spatial information
veg1k = rxr.open_rasterio(os.path.join(outvegdir,'ecotype_all.tif'))
unique_vals, counts = np.unique(veg1k.compute(), return_counts=True)
for v in unique_vals:
    da = veg1k.where(veg1k == v, 0).where(veg1k != v, 1)
    dacrs = da.rio.crs.to_epsg()
    #unique_vals, counts = np.unique(da.compute(), return_counts=True)
    da.rio.to_raster(os.path.join(outvegdir,'msk_' + str(v) + '.tif'))
    os.system('gdalwarp -overwrite -of netCDF -r sum -s_srs epsg:' + str(dacrs) + ' -t_srs ' + clmt_crs + ' -tr ' + str(modgrid_res[0]) + ' ' + str(modgrid_res[1]) + ' -te ' + str(modgrid_bounds[0]) + ' ' + str(math.floor(bounds.bottom*2)/2) + ' ' + str(modgrid_bounds[2]) + ' ' + str(modgrid_bounds[3]) + ' ' + os.path.join(outvegdir,'msk_' + str(v) + '.tif') + ' ' + os.path.join(outvegdir,'msk_' + str(v) + '_upscaled.tif'))
    vegup = rxr.open_rasterio(os.path.join(outvegdir,'msk_' + str(v) + '_upscaled.tif'))
    vegup = vegup.assign_coords(band=['CMT_' + str(v)])
    vegup.coords['band'] = ['CMT_' + str(v)]
    if v == unique_vals[0]:
        stacked = vegup
    else:
        stacked = xr.concat([stacked,vegup], dim="band")
#stacked.rio.to_raster(os.path.join(outvegdir,mod + '_veg_multibands.tif'))
## Extract primary, secondary and tertiary community
cmtdf = pd.DataFrame() 
for fn in os.listdir(outvegdir):
    if '_upscaled.tif' in fn:
        vpath = os.path.join(outvegdir, fn)
        cmt = fn.split("_")[1]
        print(f"Found file at: {vpath}, cmt: {cmt}")
        rst = rasterio.open(vpath)
        if cmt == '0':
            meta = rst.meta.copy()
            meta.update(compress='lzw')
            rstshp = rst.read(1).shape[1]
        df = rst.read(1)
        df = pd.DataFrame(df.flatten())
        df = df.set_axis(['cmt_' + str(cmt)], axis=1)
        cmtdf = pd.concat([cmtdf,df],axis = 1)
cmtlist = cmtdf.columns
cmtdf['tot'] = cmtdf.sum(axis=1)
for cmt in cmtlist:
    print(cmt)
    cmtdf[cmt+'_pct'] = 100*cmtdf[cmt]/cmtdf['tot']
cmtlist_pct = [item + '_pct' for item in cmtlist]
mask = cmtdf[cmtlist_pct].eq(cmtdf[cmtlist_pct].max(axis=1), axis=0)
cmtdf['all_max_cols'] = mask.apply(lambda x: x.index[x].tolist(), axis=1)
cmtdf['len'] = mask.apply(lambda x: len(x.index[x].tolist()), axis=1)
cmtdf['cmt1'] = cmtdf['all_max_cols'].str[0]
cmtdf['cmt1'] = np.where(pd.isna(cmtdf['cmt1']) == True,'cmt_0_pct',cmtdf['cmt1'])
cmtdf['cmt2'] = np.where(cmtdf['len']>1,cmtdf['all_max_cols'].str[1],cmtdf[cmtlist_pct].apply(lambda row: row.nlargest(2).idxmin(), axis=1))
cmtdf['cmt2'] = np.where(pd.isna(cmtdf['cmt2']) == True,'cmt_0_pct',cmtdf['cmt2'])
cmtdf['cmt3'] = np.where(cmtdf['len']>2,cmtdf['all_max_cols'].str[2],cmtdf[cmtlist_pct].apply(lambda row: row.nlargest(3).idxmin(), axis=1))
cmtdf['cmt3'] = np.where(pd.isna(cmtdf['cmt3']) == True,'cmt_0_pct',cmtdf['cmt3'])
cmtdf['pct1'] = cmtdf.apply(lambda row: row[str(row['cmt1'])],axis = 1) 
cmtdf['pct2'] = cmtdf.apply(lambda row: row[str(row['cmt2'])],axis = 1) 
cmtdf['pct3'] = cmtdf.apply(lambda row: row[str(row['cmt3'])],axis = 1) 
cmtdf['cmt1'] = np.where(cmtdf['pct1'] == 0,'cmt_0_pct',cmtdf['cmt1'])
cmtdf['cmt2'] = np.where(cmtdf['pct2'] == 0,'cmt_0_pct',cmtdf['cmt2'])
cmtdf['cmt3'] = np.where(cmtdf['pct3'] == 0,'cmt_0_pct',cmtdf['cmt3'])
cmtdf['cmt11'] = cmtdf['cmt1'].str.split('_').str[1]
cmtdf['cmt21'] = cmtdf['cmt2'].str.split('_').str[1]
cmtdf['cmt31'] = cmtdf['cmt3'].str.split('_').str[1]
## Produce the rasters
cmt1 = np.reshape(cmtdf['cmt11'], (-1, rstshp))
with rasterio.open(os.path.join(indir,'veg_primary.tiff'), mode="w+", **meta,) as out:
    out.write(cmt1, 1)
cmt1pct = np.reshape(cmtdf['pct1'], (-1, rstshp))
cmt1pct[np.isnan(cmt1pct)] = 0
with rasterio.open(os.path.join(indir,'veg_primary_pct.tiff'), mode="w+", **meta,) as out:
    out.write(cmt1pct, 1)
cmt2 = np.reshape(cmtdf['cmt21'], (-1, rstshp))
with rasterio.open(os.path.join(indir,'veg_secondary.tiff'), mode="w+", **meta,) as out:
    out.write(cmt2, 1)
cmt2pct = np.reshape(cmtdf['pct2'], (-1, rstshp))
cmt2pct[np.isnan(cmt2pct)] = 0
with rasterio.open(os.path.join(indir,'veg_secondary_pct.tiff'), mode="w+", **meta,) as out:
    out.write(cmt2pct, 1)
cmt3 = np.reshape(cmtdf['cmt31'], (-1, rstshp))
with rasterio.open(os.path.join(indir,'veg_tertiary.tiff'), mode="w+", **meta,) as out:
    out.write(cmt3, 1)
cmt3pct = np.reshape(cmtdf['pct3'], (-1, rstshp))
cmt3pct[np.isnan(cmt3pct)] = 0
with rasterio.open(os.path.join(indir,'veg_tertiary_pct.tiff'), mode="w+", **meta,) as out:
    out.write(cmt3pct, 1)
## Format and export primary land cover
v1 = xr.open_dataset(os.path.join(indir,'veg_primary.tiff'),engine = "rasterio")
v1 = v1.rename({'Band1':'veg_class'})
vp1 = xr.open_dataset(os.path.join(indir,'veg_primary.tiff'),engine = "rasterio")
vp1 = vp1.rename({'Band1':'veg_pct_cov'})
v1 = xr.merge([v1, vp1])
v1['veg_class'] = v1['veg_class'].squeeze(dim="band", drop=True)
v1['veg_pct_cov'] = v1['veg_pct_cov'].squeeze(dim="band", drop=True)
v1.drop_dims('band')
v1 = v1.rename({'x':'X'})
v1 = v1.rename({'y':'Y'})
msk = xr.open_dataset(os.path.join(outtemdir,'run-mask_' + mod + '.nc'))
v1 = xr.merge([v1, msk])
v1['X'] = v1['X'].astype(np.single)
v1['Y'] = v1['Y'].astype(np.single)
v1['lat'] = v1['lat'].astype(np.single)
v1['lon'] = v1['lon'].astype(np.single)
v1['veg_class'] = v1['veg_class'].astype(np.intc)
v1['veg_pct_cov'] = v1['veg_pct_cov'].astype(np.single)
v1['lat'].attrs={'standard_name':'latitude','units':'degree_north','_FillValue': 1.e+20}
v1['lon'].attrs={'standard_name':'longitude','units':'degree_east','_FillValue': 1.e+20}
v1['Y'].attrs={'standard_name':'degree_north','long_name':'y coordinate of projection','units':'degree','_FillValue': 1.e+20}
v1['X'].attrs={'standard_name':'degree_east','long_name':'x coordinate of projection','units':'degree','_FillValue': 1.e+20}
v1['veg_class'].attrs={'standard_name':'veg_class','long_name':'primary community','units':'','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
v1['veg_pct_cov'].attrs={'standard_name':'veg_pct_cov','long_name':'percent cover of the primary community','units':'percent','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
v1.drop(['crs','run','pct','band']).to_netcdf(os.path.join(outtemdir,'vegetation1_' + mod + '.nc'))
## Format and export secondary land cover
v2 = xr.open_dataset(os.path.join(indir,'veg_secondary.tiff'),engine = "rasterio")
v2 = v2.rename({'Band1':'veg_class'})
vp2 = xr.open_dataset(os.path.join(indir,'veg_secondary_pct.tiff'),engine = "rasterio")
vp2 = vp2.rename({'Band1':'veg_pct_cov'})
v2 = xr.merge([v2, vp2])
v2['veg_class'] = v2['veg_class'].squeeze(dim="band", drop=True)
v2['veg_pct_cov'] = v2['veg_pct_cov'].squeeze(dim="band", drop=True)
v2.drop_dims('band')
v2 = v2.rename({'x':'X'})
v2 = v2.rename({'y':'Y'})
msk = xr.open_dataset(os.path.join(outtemdir,'run-mask_' + mod + '.nc'))
v2 = xr.merge([v2, msk])
v2['X'] = v2['X'].astype(np.single)
v2['Y'] = v2['Y'].astype(np.single)
v2['lat'] = v2['lat'].astype(np.single)
v2['lon'] = v2['lon'].astype(np.single)
v2['veg_class'] = v2['veg_class'].astype(np.intc)
v2['veg_pct_cov'] = v2['veg_pct_cov'].astype(np.single)
v2['lat'].attrs={'standard_name':'latitude','units':'degree_north','_FillValue': 1.e+20}
v2['lon'].attrs={'standard_name':'longitude','units':'degree_east','_FillValue': 1.e+20}
v2['Y'].attrs={'standard_name':'degree_north','long_name':'y coordinate of projection','units':'degree','_FillValue': 1.e+20}
v2['X'].attrs={'standard_name':'degree_east','long_name':'x coordinate of projection','units':'degree','_FillValue': 1.e+20}
v2['veg_class'].attrs={'standard_name':'veg_class','long_name':'secondary community','units':'','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
v2['veg_pct_cov'].attrs={'standard_name':'veg_pct_cov','long_name':'percent cover of the secondary community','units':'percent','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
v2.drop(['crs','run','pct','band']).to_netcdf(os.path.join(outtemdir,'vegetation2_' + mod + '.nc'))
## Format and export tertiary land cover
v3 = xr.open_dataset(os.path.join(indir,'veg_tertiary.tiff'),engine = "rasterio")
v3 = v3.rename({'Band1':'veg_class'})
vp3 = xr.open_dataset(os.path.join(indir,'veg_tertiary_pct.tiff'),engine = "rasterio")
vp3 = vp3.rename({'Band1':'veg_pct_cov'})
v3 = xr.merge([v3, vp3])
v3['veg_class'] = v3['veg_class'].squeeze(dim="band", drop=True)
v3['veg_pct_cov'] = v3['veg_pct_cov'].squeeze(dim="band", drop=True)
v3.drop_dims('band')
v3 = v3.rename({'x':'X'})
v3 = v3.rename({'y':'Y'})
msk = xr.open_dataset(os.path.join(outtemdir,'run-mask_' + mod + '.nc'))
v3 = xr.merge([v3, msk])
v3['X'] = v3['X'].astype(np.single)
v3['Y'] = v3['Y'].astype(np.single)
v3['lat'] = v3['lat'].astype(np.single)
v3['lon'] = v3['lon'].astype(np.single)
v3['veg_class'] = v3['veg_class'].astype(np.intc)
v3['veg_pct_cov'] = v3['veg_pct_cov'].astype(np.single)
v3['lat'].attrs={'standard_name':'latitude','units':'degree_north','_FillValue': 1.e+20}
v3['lon'].attrs={'standard_name':'longitude','units':'degree_east','_FillValue': 1.e+20}
v3['Y'].attrs={'standard_name':'degree_north','long_name':'y coordinate of projection','units':'degree','_FillValue': 1.e+20}
v3['X'].attrs={'standard_name':'degree_east','long_name':'x coordinate of projection','units':'degree','_FillValue': 1.e+20}
v3['veg_class'].attrs={'standard_name':'veg_class','long_name':'tertiary community','units':'','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
v3['veg_pct_cov'].attrs={'standard_name':'veg_pct_cov','long_name':'percent cover of the tertiary community','units':'percent','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
v3.drop(['crs','run','pct','band']).to_netcdf(os.path.join(outtemdir,'vegetation3_' + mod + '.nc'))

### Wetland processing
cmtdf = pd.DataFrame() 
for fn in os.listdir(outvegdir):
    if '_upscaled.tif' in fn:
        vpath = os.path.join(outvegdir, fn)
        cmt = fn.split("_")[1]
        print(f"Found file at: {vpath}, cmt: {cmt}")
        rst = rasterio.open(vpath)
        if cmt == '0':
            meta = rst.meta.copy()
            meta.update(compress='lzw')
            rstshp = rst.read(1).shape[1]
        df = rst.read(1)
        df = pd.DataFrame(df.flatten())
        df = df.set_axis(['cmt_' + str(cmt)], axis=1)
        cmtdf = pd.concat([cmtdf,df],axis = 1)

cmtlist = cmtdf.columns
cmtdf['tot'] = cmtdf.sum(axis=1)
cmtdf['bog'] = cmtdf[['cmt_31','cmt_75','cmt_92']].sum(axis=1)
cmtdf['fen'] = cmtdf['cmt_55']
cmtdf['wetsedge'] = cmtdf[['cmt_6','cmt_73','cmt_76']].sum(axis=1)
cmtdf['wetland'] = cmtdf[['bog','fen','wetsedge']].sum(axis=1)

cmtdf['bog_pct'] = cmtdf['bog']/cmtdf['tot']
cmtdf['fen_pct'] = cmtdf['fen']/cmtdf['tot']
cmtdf['wetsedge_pct'] = cmtdf['wetsedge']/cmtdf['tot']
cmtdf['wetsedge'] = cmtdf[['cmt_6','cmt_73','cmt_76']].sum(axis=1)

wtddf = cmtdf[['tot','bog','fen','wetsedge','bog_pct','fen_pct','wetsedge_pct']]
wtddf['wetland'] = wtddf[['bog','fen','wetsedge']].sum(axis=1)
wtddf['wetland_pct'] = wtddf['wetland']/wtddf['tot']
wtddf['cmt_0'] = wtddf['tot'] - wtddf['wetland']
wtddf['cmt_0_pct'] = wtddf['cmt_0']/wtddf['tot']
wtdlist_pct = [item + '_pct' for item in ['cmt_0','bog','fen','wetsedge']]

maskw = wtddf[wtdlist_pct].eq(wtddf[wtdlist_pct].max(axis=1), axis=0)
wtddf['all_max_cols'] = maskw.apply(lambda x: x.index[x].tolist(), axis=1)
wtddf['len'] = maskw.apply(lambda x: len(x.index[x].tolist()), axis=1)
wtddf['cmt1'] = wtddf['all_max_cols'].str[0]
wtddf['cmt1'] = np.where(pd.isna(wtddf['cmt1']) == True,'cmt_0_pct',wtddf['cmt1'])
wtddf[(wtddf['wetland']>0) & (wtddf['all_max_cols'] == 'fen_pct')]
wtddf['pct1'] = wtddf.apply(lambda row: row[str(row['cmt1'])],axis = 1) 
wtddf['cmt1'] = np.where(wtddf['pct1'] == 0,'cmt_0_pct',wtddf['cmt1'])
wtddf['wtd_cmt'] = np.where(wtddf['cmt1'] == 'wetsedge_pct',6,np.where(wtddf['cmt1'] == 'bog_pct',31,np.where(wtddf['cmt1'] == 'fen_pct',32,0)))
wtddf['wtf_pct'] = np.where(wtddf['wtd_cmt'] > 0,wtddf['wetland_pct'],1)
## Produce the rasters
cmt1 = np.reshape(wtddf['wtd_cmt'], (-1, rstshp))
with rasterio.open(os.path.join(indir,'wetland_primary.tiff'), mode="w+", **meta, ) as out:
    out.write(cmt1, 1)
cmt1pct = np.reshape(wtddf['wtf_pct'], (-1, rstshp))
cmt1pct[np.isnan(cmt1pct)] = 0
meta.update(dtype='float32')
with rasterio.open(os.path.join(indir,'wetland_primary_pct.tiff'), mode="w+", **meta) as out:
    out.write(cmt1pct, 1)
## Format and export primary land cover
v1 = xr.open_dataset(os.path.join(indir,'wetland_primary.tiff'),engine = "rasterio")
v1 = v1.rename({'Band1':'veg_class'})
vp1 = xr.open_dataset(os.path.join(indir,'wetland_primary_pct.tiff'),engine = "rasterio")
vp1 = vp1.rename({'Band1':'veg_pct_cov'})
v1 = xr.merge([v1, vp1])
v1['veg_class'] = v1['veg_class'].squeeze(dim="band", drop=True)
v1['veg_pct_cov'] = v1['veg_pct_cov'].squeeze(dim="band", drop=True)
v1.drop_dims('band')
v1 = v1.rename({'x':'X'})
v1 = v1.rename({'y':'Y'})
msk = xr.open_dataset(os.path.join(outtemdir,'run-mask_' + mod + '.nc'))
v1 = xr.merge([v1, msk])
v1['X'] = v1['X'].astype(np.single)
v1['Y'] = v1['Y'].astype(np.single)
v1['lat'] = v1['lat'].astype(np.single)
v1['lon'] = v1['lon'].astype(np.single)
v1['veg_class'] = v1['veg_class'].astype(np.intc)
v1['veg_pct_cov'] = v1['veg_pct_cov'].astype(np.single)
v1['lat'].attrs={'standard_name':'latitude','units':'degree_north','_FillValue': 1.e+20}
v1['lon'].attrs={'standard_name':'longitude','units':'degree_east','_FillValue': 1.e+20}
v1['Y'].attrs={'standard_name':'degree_north','long_name':'y coordinate of projection','units':'degree','_FillValue': 1.e+20}
v1['X'].attrs={'standard_name':'degree_east','long_name':'x coordinate of projection','units':'degree','_FillValue': 1.e+20}
v1['veg_class'].attrs={'standard_name':'wetland_class','long_name':'primary wetland community','units':'','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
v1['veg_pct_cov'].attrs={'standard_name':'wetland_pct_cov','long_name':'percent cover of the all wetland community','units':'percent','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
v1.drop(['crs','run','pct','band']).to_netcdf(os.path.join(outtemdir,'wetland1_' + mod + '.nc'))








# Sum across bands
summed_band = stacked.sel(band=1) + stacked.sel(band=2)
summed_band = summed_band.expand_dims(band=["summed_band"])
updated_da = xr.concat([stacked, summed_band], dim="band")

# 5. Save the updated raster
updated_da.rio.to_raster("summed_output.tif")



#####   SUMMARIZING THE CLIMATE TIME SERIES     #####

print('SUMMARIZING & CROPING THE DATA')
for mod in modlist:
    print('processing climate data for', mod, '...')
for var in varlist:
    print('  processing variable', var, '...')
    filelist = [f for f in os.listdir(os.path.join(indir,mod,'05deg')) 
                  if var + '.' in f and os.path.isfile(os.path.join(os.path.join(indir,mod,'05deg'), f))]
    yearlist = sorted([int(f.split(".")[2].split("_")[0]) for f in filelist])
    if not os.path.exists(os.path.join(outdir,mod, mod + '_' + var + '.nc')): 
        dsss = None
        for y in yearlist:
            ## filename
            if mod == 'stable':
                fn = mod + '.' + var + '.' + str(y) + '_6hr.noleap.nc'
            else:
                fn = mod + '_clim3_50perc_1pctCO2.' + var + '.' + str(y) + '_6hr.noleap.nc'
            ## Summarize data from 6-hourly to monthly
            print('    summarizing from 6-hourly to monthly for year ', str(y), '...')
            ds = xr.open_dataset(os.path.join(indir,mod,'05deg',fn), decode_coords="all")
            if var in varlist_mean:
                dss = ds[[var]].resample(time='MS').mean()
            elif var in varlist_sum:
                # Converting kg/s to mm
                if var == 'pre':
                    ds[var]*=60*60*6
                dss = ds[[var]].resample(time='MS').sum()
            dss = dss.isel(time=slice(None, -1))
            ## Crop the data to the mask original extent
            print('    croping to area of interest...')
            dss = dss.where(dss['lat'] >= math.floor(bounds.bottom*2)/2, drop=True)
            ## Exporting the data
            print('    concatenating the monthly data...')
            if dsss is None:
                dsss = dss
            else:
                dsss = xr.concat([dsss, dss], dim='time')
        print('  writing the timeseries out for variable ', var)
        dsss.to_netcdf(os.path.join(outdir,mod, mod + '_' + var + '.nc'),unlimited_dims='time',format='NETCDF4', mode='w')






#####   UNIT CONVERSION  &  FORMATING   #####

print('UNIT CONVERSION')
## Merging the variable files
for mod in modlist:
print('  for model ', mod)
print('    merging the data...')
dss = None
for var in varlist:
    ds = xr.open_dataset(os.path.join(outdir,mod, mod + '_' + var + '.nc'))
    if dss is None:
        dss = ds
    else:
        dss = xr.merge([dss, ds])
## Remove unnecessary files
#for var in varlist:
#    if os.path.exists(os.path.join(outdir,mod, mod + '_' + var + '.nc')):
#        os.remove(os.path.join(outdir,mod, mod + '_' + var + '.nc'))
## Unit conversions
print('    unit conversion...')
# temps from K to oC
dss['tmp'] -= 273.15
# vapor pressure in kPa from atm pressure and specific humidity
dss['vapor_press'] = (0.001 * dss['pres'] * dss['spfh']) / (0.622 + 0.378 * dss['spfh'])
dss = dss.drop_vars(['spfh','pres'])
dss = dss.rename({'tmp': 'tair', 'pre': 'precip', 'dswrf':'nirr'})
#dss.to_netcdf(os.path.join(outdir, mod, mod + '.nc'),unlimited_dims='time',format='NETCDF4', mode='w')
### FORMATING TO TEM FORMAT
print('    formatting to TEM...')
#dss = xr.open_dataset(os.path.join(outdir, mod, mod + '.nc'))
dss = dss.assign_coords(time=dss.time.astype("datetime64[us]"))
dss['dse'] = (dss['time'] - np.datetime64('1850-01-01')) / np.timedelta64(1, 'D')
dss['dse2'] = dss['dse'] -  calendar.leapdays(1850, dss.time.dt.year + 1)
dss = dss.assign_coords(time=('time',dss['dse2'].values))
dss = dss.drop_vars(['dse','dse2'])
dss = dss.rename({'lat':'Y','lon':'X'})
#dss = dss.rename_dims({'lat':'Y','lon':'X'})
dss['X'] = dss['X'].astype(np.single)
dss['Y'] = dss['Y'].astype(np.single)
dss['tair'] = dss['tair'].astype(np.single)
dss['precip'] = dss['precip'].astype(np.single)
dss['nirr'] = dss['nirr'].astype(np.single)
dss['vapor_press'] = dss['vapor_press'].astype(np.single)
dss['Y'].attrs={'standard_name':'latitude','units':'degree_north'}
dss['X'].attrs={'standard_name':'longitude','units':'degree_east'}
dss['tair'].attrs={'standard_name':'air_temperature','units':'celsius','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
dss['precip'].attrs={'standard_name':'precipitation_amount','units':'mm month-1','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
dss['nirr'].attrs={'standard_name':'downwelling_shortwave_flux_in_air','units':'W m-2','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
dss['vapor_press'].attrs={'standard_name':'water_vapor_pressure','units':'kPa','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
dss['time'].attrs={'units':'days since 1850-1-1 0:0:0','long_name':'time','calendar':'365_day'}
dss.time.encoding['units'] = 'days since 1850-01-01 00:00:00'
dss.time.encoding['calendar'] = '365_day'
dss.time.encoding['long_name'] = 'time'
##Include lat and lon from runmask
msk = xr.open_dataset(os.path.join(outtemdir,'run-mask_' + mod + '.nc'))
dss = xr.merge([dss, msk])
dss = dss.drop_vars(['pct','run'])
dss.to_netcdf(os.path.join(outtemdir, 'climate_' + mod + '.nc'),unlimited_dims='time',format='NETCDF4', mode='w')





#####   OTHER ANCILLARIES   #####


### Atmospheric CO2 timeseries
data = pd.read_csv(os.path.join(indir,'co2','WIEMIP_1pctco2.txt'), sep=' ', header = None)
data = data.rename(columns={0: "year", 1: "co2"})
ds = data.set_index(['year']).to_xarray()
ds['co2'].attrs={'standard_name':'atmospheric CO2 concentration','units':'ppm'}
ds['year'] = ds['year'].astype(np.int_)
ds['co2'] = ds['co2'].astype(np.single)
ds.to_netcdf(os.path.join(outtemdir,'co2_dyn.nc'),unlimited_dims='year')

data = pd.read_csv(os.path.join(indir,'co2','WIEMIP_1pctco2.txt'), sep=' ', header = None)
data = data.rename(columns={0: "year", 1: "co2"})
data['co2'] = 280
ds = data.set_index(['year']).to_xarray()
ds['co2'].attrs={'standard_name':'atmospheric CO2 concentration','units':'ppm'}
ds['year'] = ds['year'].astype(np.int_)
ds['co2'] = ds['co2'].astype(np.single)
ds.to_netcdf(os.path.join(outtemdir,'co2_cnst.nc'),unlimited_dims='year')





### Empty FRI timeseries
print('Create FRI dataset')
for mod in modlist:
print('  for model ', mod)
ds = xr.open_dataset(os.path.join(outtemdir,'run-mask_' + mod + '.nc'))
df = ds.to_dataframe()
df.reset_index(inplace=True)
fri = df.drop('run', axis=1)
fri['fri'] = 2000
fri['fri_severity'] = 0
fri['fri_jday_of_burn'] = 0
fri['fri_area_of_burn'] = 0
fri_nc = fri.set_index(['Y', 'X']).to_xarray()
fri_nc['lat'] = fri_nc['lat'].astype(np.single)
fri_nc['lon'] = fri_nc['lon'].astype(np.single)
fri_nc['fri'] = fri_nc['fri'].astype(np.intc)
fri_nc['fri_severity'] = fri_nc['fri_severity'].astype(np.intc)
fri_nc['fri_jday_of_burn'] = fri_nc['fri_jday_of_burn'].astype(np.intc)
fri_nc['fri_area_of_burn'] = fri_nc['fri_area_of_burn'].astype(np.intc)
fri_nc['lat'].attrs={'standard_name':'latitude','units':'degree_north'}
fri_nc['lon'].attrs={'standard_name':'longitude','units':'degree_east'}
fri_nc['Y'].attrs={'standard_name':'projection_y_coordinate','long_name':'y coordinate of projection','units':'m'}
fri_nc['X'].attrs={'standard_name':'projection_x_coordinate','long_name':'x coordinate of projection','units':'m'}
fri_nc['fri'].attrs={'standard_name':'fire_return_interval','units':'yrs','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
fri_nc['fri_jday_of_burn'].attrs={'standard_name':'fri_jday_of_burn','units':'doy','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
fri_nc['fri_area_of_burn'].attrs={'standard_name':'fri_area_of_burn','units':'','grid_mapping':'albers_conical_equal_area','_FillValue': -999.0}
fri_nc.to_netcdf(os.path.join(outtemdir,'fri-fire_' + mod + '.nc'))



### Topography and drainage


os.system('gdalwarp -overwrite -of netCDF -r sum -s_srs epsg:' + str(in_crs) + ' -t_srs ' + clmt_crs + ' -tr ' + str(modgrid_res[0]) + ' ' + str(modgrid_res[1]) + ' -te ' + str(modgrid_bounds[0]) + ' ' + str(math.floor(bounds.bottom*2)/2) + ' ' + str(modgrid_bounds[2]) + ' ' + str(modgrid_bounds[3]) + ' ' + os.path.join(anc_path,'TOPO','elevation_4k_6931_mask.tif') + ' ' + os.path.join(outancdir,'elevation.nc'))
os.system('gdalwarp -overwrite -of netCDF -r sum -s_srs epsg:' + str(in_crs) + ' -t_srs ' + clmt_crs + ' -tr ' + str(modgrid_res[0]) + ' ' + str(modgrid_res[1]) + ' -te ' + str(modgrid_bounds[0]) + ' ' + str(math.floor(bounds.bottom*2)/2) + ' ' + str(modgrid_bounds[2]) + ' ' + str(modgrid_bounds[3]) + ' ' + os.path.join(anc_path,'TOPO','slope_4k_6931_mask.tif') + ' ' + os.path.join(outancdir,'slope.nc'))
os.system('gdalwarp -overwrite -of netCDF -r sum -s_srs epsg:' + str(in_crs) + ' -t_srs ' + clmt_crs + ' -tr ' + str(modgrid_res[0]) + ' ' + str(modgrid_res[1]) + ' -te ' + str(modgrid_bounds[0]) + ' ' + str(math.floor(bounds.bottom*2)/2) + ' ' + str(modgrid_bounds[2]) + ' ' + str(modgrid_bounds[3]) + ' ' + os.path.join(anc_path,'TOPO','tpi_4k_6931_mask.tif') + ' ' + os.path.join(outancdir,'tpi.nc'))
os.system('gdalwarp -overwrite -of netCDF -r sum -s_srs epsg:' + str(in_crs) + ' -t_srs ' + clmt_crs + ' -tr ' + str(modgrid_res[0]) + ' ' + str(modgrid_res[1]) + ' -te ' + str(modgrid_bounds[0]) + ' ' + str(math.floor(bounds.bottom*2)/2) + ' ' + str(modgrid_bounds[2]) + ' ' + str(modgrid_bounds[3]) + ' ' + os.path.join(anc_path,'TOPO','drainage.tif') + ' ' + os.path.join(outancdir,'drainage.nc'))

out = os.path.join(outancdir,'aoi_5k_buff_4326_2_0.nc') 

ds = xr.open_dataset(os.path.join(outmskdir,mod+'_mask.nc'), decode_coords="all")
ds = ds.fillna(0)
arr=ds['Band1'].values
print('  create a mask variable...')
ds['pct'] = 100*ds['Band1']/ds['Band1'].max()
ds['msk'] = xr.where(ds["Band1"] > 0, 1, 0)
#ds['msk'].values[80,100]
print('  reformate to TEM standards...')
ds = ds.rename({'lat':'Y','lon':'X'})
lon_2d = np.repeat(np.array([ds['X'].values]), ds['Y'].values.shape[0], axis=0)
lat_2d = np.repeat(ds['Y'].values.reshape(-1, 1), ds['X'].values.shape[0], axis=1)
ds = ds.assign(lat=(['Y','X'],lat_2d))
ds = ds.assign(lon=(['Y','X'],lon_2d))
ds = ds.drop_vars(['Band1','crs'])
ds = ds.rename({'msk':'run'})
ds['lat'] = ds['lat'].astype(np.single)
ds['lon'] = ds['lon'].astype(np.single)
ds['X'] = ds['X'].astype(np.single)
ds['Y'] = ds['Y'].astype(np.single)
ds['run'] = ds['run'].astype(np.intc)
ds['pct'] = ds['pct'].astype(np.single)
ds['lat'].attrs={'standard_name':'latitude','units':'degree_north','_FillValue': 1.e+20}
ds['lon'].attrs={'standard_name':'longitude','units':'degree_east','_FillValue': 1.e+20}
ds['Y'].attrs={'standard_name':'degree_north','long_name':'y coordinate of projection','units':'degree','_FillValue': 1.e+20}
ds['X'].attrs={'standard_name':'degree_east','long_name':'x coordinate of projection','units':'degree','_FillValue': 1.e+20}
ds['run'].attrs={'standard_name':'mask','units':'','grid_mapping':'latitude_longitude','_FillValue': -9999}
ds['pct'].attrs={'standard_name':'percent area of land','units':'percent','grid_mapping':'latitude_longitude','_FillValue': -9999}
ds.to_netcdf(os.path.join(TEMinputs,'run-mask_' + mod + '.nc'))



### Atmospheric CO2 timeseries
data = pd.read_csv(os.path.join(indir,'co2','WIEMIP_1pctco2.txt'), sep=' ', header = None)
data = data.rename(columns={0: "year", 1: "co2"})
ds = data.set_index(['year']).to_xarray()
ds['co2'].attrs={'standard_name':'atmospheric CO2 concentration','units':'ppm'}
ds['year'] = ds['year'].astype(np.int_)
ds['co2'] = ds['co2'].astype(np.single)
ds.to_netcdf(os.path.join(outtemdir,'co2.nc'),unlimited_dims='year')



### Vegetation






os.system('gdalwarp -overwrite -of netCDF -r sum -s_srs epsg:' + str(in_crs) + ' -t_srs ' + clmt_crs + ' -tr ' + str(modgrid_res[0]) + ' ' + str(modgrid_res[1]) + ' -te ' + str(modgrid_bounds[0]) + ' ' + str(math.floor(bounds.bottom*2)/2) + ' ' + str(modgrid_bounds[2]) + ' ' + str(modgrid_bounds[3]) + ' ' + os.path.join(anc_path,'TOPO','elevation_4k_6931_mask.tif') + ' ' + os.path.join(outancdir,'elevation.nc'))










