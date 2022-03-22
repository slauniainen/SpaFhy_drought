# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:46:03 2022

@author: 03081268
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from netCDF4 import Dataset#, date2num, num2date
from spafhy import spafhy

from spafhy.spafhy_io import read_FMI_weather, combine_spafhy_nc

from spafhy.spafhy_parameters_tram import parameters, soil_properties
from spafhy.spafhy_io import create_catchment, read_AsciiGrid, write_AsciiGrid

eps = np.finfo(float).eps

""" paths defined in parameters"""
from harvests_2021 import operations

siteid = 12

# read parameters, list sub-catchment files for site
(pgen, _, _, _) = parameters()
datafolder = pgen['gis_folder']
resultfolder = pgen['ncf_file']

site = 'site' + str(siteid) + '_'
 
dirs = os.listdir(datafolder)
fpaths = [s for s in dirs if site in s]

""" -- run for sub-catchments """
for f in fpaths:
    forcingfile = os.path.join(datafolder, 'weather', 'Site_' + str(siteid) +'_FMIdata_cor.csv')
    #forcingfile = os.path.join(datafolder, 'weather', 'vihti_FMI_10x10.csv')
    print(forcingfile)
    infile = os.path.join(datafolder, f)
    print(infile)
    outfile = os.path.join(resultfolder, f + '.nc')
    print(outfile)
    
    print('*** running site ' + str(siteid) + ' subcatchment ' + str(f) + ' ***')
    
    gisdata = create_catchment(infile, plotgrids=True, plotdistr=False)

    """ set up SpaFHy for the site """
    
    # load parameter dictionaries
    (_, pcpy, pbu, ptop) = parameters()
    psoil = soil_properties()
    
    """ read forcing data and catchment runoff file """
    FORC = read_FMI_weather(str(siteid),
                            pgen['start_date'],
                            pgen['end_date'],
                            sourcefile=forcingfile)
    
    FORC['Prec'] = FORC['Prec'] / pgen['dt']  # mms-1
    FORC['U'] = 2.0 # use constant wind speed ms-1
    Nsteps = len(FORC)
    
    # initialize spafhy
    spa = spafhy.initialize(pgen, pcpy, pbu, ptop, psoil, gisdata, cpy_outputs=False, 
                            bu_outputs=False, top_outputs=False, flatten=True)

    # create netCDF output file

    dlat, dlon = np.shape(spa.GisData['cmask'])

    ncf, ncf_file = spafhy.initialize_netCDF(ID=spa.id, fname=outfile, lat0=spa.GisData['lat0'], 
                                             lon0=spa.GisData['lon0'], dlat=dlat, dlon=dlon, dtime=None)
    
    # run spafhy for spinup
    Nspin = np.where(FORC.index == pgen['spinup_end'])[0][0]
    
    for k in range(0, Nspin):
        forc= FORC[['doy', 'Rg', 'Par', 'T', 'Prec', 'VPD', 'CO2','U']].iloc[k]
        spa.run_timestep(forc, ncf=False)
    
    spa.step_nr = 0
    # run spafhy for Nsteps
    for k in range(Nspin, Nsteps):
        forc= FORC[['doy', 'Rg', 'Par', 'T', 'Prec', 'VPD', 'CO2','U']].iloc[k]
        
        spa.run_timestep(forc, ncf=ncf)
    
    # append gisdata to ncf
    for v in list(ncf['gis'].variables):
        ncf['gis'][v][:,:] = gisdata[v]
        
    # close output file
    ncf.close()

    #del spa, ncf, ncf_file, gisdata


""" -- combine ncfiles and remove separate """
ncfiles = os.listdir(resultfolder)
ncfiles = [os.path.join(resultfolder, s) for s in ncfiles if site in s]

    
#os.chdir(folder)

outfile = os.path.join(resultfolder, 'combined', 'site' + str(siteid) + '.nc')

cf  = combine_spafhy_nc(str(siteid), ncfiles, outfile)
cf.close()

# remove sub-catchment results
for f in ncfiles:
    os.remove(f)
    
#%% save mean vol_moisture and saturation_deficits for time of harvests
# NOTE: assumes data starts from 1.1.2021!

cf = Dataset(outfile, 'r')

print('site ' + str(siteid) + ' harvest from ' + operations['harvest_start'].iloc[siteid-1] + 
      ' to ' + operations['harvest_end'].iloc[siteid-1])
t0 = operations['harvest_start_doy'].iloc[siteid-1]
t1 = operations['harvest_end_doy'].iloc[siteid-1]

mbe = np.nansum(cf['bu']['Mbe'], axis=0)
plt.figure()
plt.imshow(mbe)

# soil moisture and saturation deficits (scaled TWI) for operation dates
#W0 = cf['bu']['Wliq'][t0,:,:]
#W1 = cf['bu']['Wliq'][t1,:,:]

W = np.nanmean(cf['bu']['Wliq'][t0:t1,:,:], axis=0)
dW = np.nanmax(cf['bu']['Wliq'][t0:t1,:,:], axis=0) - np.nanmin(cf['bu']['Wliq'][t0:t1,:,:], axis=0)
               
S0 = cf['top']['Sloc'][t0,:,:]
S1 = cf['top']['Sloc'][t1,:,:]

S = np.nanmean(cf['top']['Sloc'][t0:t1,:,:], axis=0)
dS = np.nanmax(cf['top']['Sloc'][t0:t1,:,:], axis=0) - np.nanmin(cf['top']['Sloc'][t0:t1,:,:], axis=0)
                
plt.figure()

plt.subplot(121)
plt.imshow(W); plt.colorbar(); plt.title('Wliq (m3m-3)')

plt.subplot(122)
plt.imshow(dW); plt.colorbar(); plt.title('dWliq (m3m-3')

plt.figure()

plt.subplot(121)
plt.imshow(S); plt.colorbar(); plt.title('S')

plt.subplot(122)
plt.imshow(dS); plt.colorbar(); plt.title('dS')


#%% save as ascii-grids

# prepare info
gisdata = create_catchment(infile, plotgrids=False, plotdistr=False)
info0 = gisdata['info']

# coordinates
ncols = len(cf['lon'])
nrows = len(cf['lat'])
xll = min(np.array(cf['lon']))
yll = min(np.array(cf['lat']))

info = info0.copy()

info[0] = 'ncols   ' + str(ncols) + '\n'
info[1] = 'nrows   ' + str(nrows) + '\n'
info[2] = 'xllcorner   ' + str(xll) + '\n'
info[3] = 'xllcorner   ' + str(yll) + '\n'


W = np.array(W)
W[np.where(np.isnan(W))] = -9999.0

outfile = os.path.join(resultfolder, 'combined', 'site' + str(siteid) + 
                       '_vol_moisture_doy_' + str(t0) + '_' + str(t1) + '.asc')
write_AsciiGrid(outfile, W, info, fmt='%.3f')

dW = np.array(W)
dW[np.where(np.isnan(dW))] = -9999.0

outfile = os.path.join(resultfolder, 'combined', 'site' + str(siteid) + 
                       '_delta_vol_moisture_doy_' + str(t0) + '_' + str(t1) + '.asc')
write_AsciiGrid(outfile, W, info, fmt='%.3f')

S = np.array(S)
S[np.where(np.isnan(S))] = -9999.0

outfile = os.path.join(resultfolder, 'combined', 'site' + str(siteid) + 
                       '_sat_deficit_doy_' + str(t0) + '_' + str(t1) + '.asc')
write_AsciiGrid(outfile, S, info, fmt='%.3f')

dS = np.array(S)
dS[np.where(np.isnan(dS))] = -9999.0

outfile = os.path.join(resultfolder, 'combined', 'site' + str(siteid) + 
                       '_delta_sat_deficit_doy_' + str(t0) + '_' + str(t1) + '.asc')
write_AsciiGrid(outfile, S, info, fmt='%.3f')

# close nc-file
cf.close()

#data, info1, (xloc, yloc), cellsize, nodata  = read_AsciiGrid('testi.asc')




