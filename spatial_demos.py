# -*- coding: utf-8 -*-
"""
Spatial demonstrations of Spathy

Created on Tue Sep 19 13:21:26 2017

@author: slauniai
"""
import os
#import spotpy
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from spafhy.spafhy_parameters import parameters, soil_properties
from spafhy.spafhy import spafhy_driver

eps = np.finfo(float).eps

# spathy_path = os.path.join('c:', 'c:\repositories\spafhy')
# setupfile = os.path.join(r'c:\repositories\spathy\ini', 'spathy_default.ini')
# setupfile = 'c:\pyspace\spathy\ini\spathy4_ch_1default.ini'

""" load parameter dictionaries """
pgen, pcpy, pbu, ptop = parameters()
psoil = soil_properties()

spa, ncf = spafhy_driver(pgen, pcpy, pbu, ptop, psoil, catchment_id='3',
                         ncf=False, flatten=False, 
                         cpy_outputs=True, bu_outputs=True, top_outputs=True)

gis = spa.GisData
LAIc = gis['LAI_conif']
LAId = gis['LAI_decid']
soil = gis['soil']
twi = gis['twi']

peat_ix = np.where(soil == 4)
med_ix = np.where(soil == 2)
coarse_ix = np.where(soil == 1)
htwi_ix = np.where(twi > 12)
ltwi_ix = np.where(twi < 7)

# cpy results
cres = spa.cpy.results
bres = spa.bu.results
tres = spa.top.results

# plot Qt
Qt = 1e3 * np.array(tres['Qt'])  # mm/d
plt.figure()
plt.plot(Qt)
plt.ylim([0, 20])
del Qt

TR = np.array(cres['Transpi'])  # tranpi
EF = np.array(cres['Efloor'])  # floor evap
IE = np.array(cres['Interc'])  # interception evap

ET = TR + EF + IE

# snow, seek maximum timing
SWE = np.array(cres['SWE'])  # SWE
a = np.nansum(SWE, axis=1)
a = np.nansum(a, axis=1)
swe_max_ix = int(np.where(a == np.nanmax(a))[0][0])

# rootzone water storage; seek maximum and minimum
W = np.array(bres['Wliq'])
a = np.nansum(W, axis=1)
a = np.nansum(a, axis=1)
ix_slow = int(np.where(a == np.nanmin(a))[0])
ix_shi = int(np.where(a == np.nanmax(a))[0])

DR = np.array(bres['Drain'])*1e3

P = np.sum(spa.FORC['Prec'])*spa.dt  # precip


# plot figures of annual ratios
#sns.color_palette('muted')
#plt.figure()
#plt.subplot(221)
#plt.imshow(LAIc + LAId); plt.colorbar(); plt.title('LAI (m$^2$m,^{-2}$)')
#
#plt.subplot(222)
#plt.imshow(np.sum(ET, axis=0)/P); plt.colorbar(); plt.title('ET/P (-)')
#
#plt.subplot(223)
#plt.imshow(np.sum(TR, axis=0)/P); plt.colorbar(); plt.title('T$_r$/P (-)')
#
#plt.subplot(224)
#plt.imshow(np.sum(IE, axis=0)/P); plt.colorbar(); plt.title('E/P (-)')

#%% sns.color_palette('muted')
plt.figure()
plt.subplot(321)
sns.heatmap(LAIc + LAId, cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False); plt.title('LAI (m$^2$m$^{-2}$)')

plt.subplot(322)
sns.heatmap(LAId /(LAIc + LAId), cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False); plt.title('LAI$_d$ fraction (-)')

plt.subplot(323)
sns.heatmap(np.sum(ET, axis=0)/P, cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False); plt.title('ET/P (-)')

plt.subplot(324)
sns.heatmap(np.sum(IE, axis=0)/P, cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False); plt.title('E/P (-)')

plt.subplot(325)
sns.heatmap(np.sum(TR, axis=0)/P, cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False); plt.title('T$_r$/P (-)')

plt.subplot(326)
sns.heatmap(np.sum(EF, axis=0)/P, cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False); plt.title('E$_f$/P (-)')

plt.figure()
sns.heatmap(SWE[swe_max_ix,:,:], cmap='coolwarm',cbar=True, xticklabels=False, yticklabels=False); plt.title('SWE (mm)')

plt.figure()
Wliq = np.array(bres['Wliq'])
ss = np.array(bres['Wliq'] / spa.bu.poros)
S = np.array(tres['S'])
s_hi = spa.top.local_s(S[ix_shi])
s_hi[s_hi<0] = 0.0
s_low = spa.top.local_s(S[ix_slow])
s_low[s_low<0] = 0.0


plt.subplot(321)
sns.heatmap(Wliq[ix_slow,:,:], cmap='coolwarm',cbar=True, vmin=0.1, vmax=0.9, xticklabels=False, yticklabels=False); plt.title('Wliq (m3/m3)')
plt.subplot(322)
sns.heatmap(Wliq[ix_shi,:,:], cmap='coolwarm',cbar=True, vmin=0.1, vmax=0.9, xticklabels=False, yticklabels=False); plt.title('Wliq (m3/m3)')
plt.subplot(323)
sns.heatmap(ss[ix_slow,:,:], cmap='coolwarm',cbar=True, vmin=0.0, vmax=1.0, xticklabels=False, yticklabels=False); plt.title('root zone sat. ratio (-)')
plt.subplot(324)
sns.heatmap(ss[ix_shi,:,:], cmap='coolwarm',cbar=True, vmin=0.0, vmax=1.0, xticklabels=False, yticklabels=False); plt.title('root zone sat. ratio (-)')
plt.subplot(325)
sns.heatmap(s_low, cmap='coolwarm',cbar=True, vmin=0.0, vmax=0.08, xticklabels=False, yticklabels=False); plt.title('topmodel sat deficit (m)')
plt.subplot(326)
sns.heatmap(s_hi, cmap='coolwarm',cbar=True, vmin=0.0, vmax=0.08, xticklabels=False, yticklabels=False); plt.title('topmodel sat deficit (m)')

plt.figure()
plt.subplot(221)
sns.heatmap(Wliq[ix_slow,:,:], cmap='coolwarm_r',cbar=True, vmin=0.1, vmax=0.9, xticklabels=False, yticklabels=False); plt.title('Wliq (m3/m3)')
plt.subplot(222)
sns.heatmap(Wliq[ix_shi,:,:], cmap='coolwarm_r',cbar=True, vmin=0.1, vmax=0.9, xticklabels=False, yticklabels=False); plt.title('Wliq (m3/m3)')
plt.subplot(223)
sns.heatmap(s_low, cmap='coolwarm',cbar=True, vmin=0.0, vmax=0.08, xticklabels=False, yticklabels=False); plt.title('topmodel sat deficit (m)')
plt.subplot(224)
sns.heatmap(s_hi, cmap='coolwarm',cbar=True, vmin=0.0, vmax=0.08, xticklabels=False, yticklabels=False); plt.title('topmodel sat deficit (m)')

# plot scaled s

a1 = s_low.copy()

a1[a1 >=0.03] = 2.0
a1[(a1 < 0.03) & (a1>=0.01)] = 1.0
a1[a1 <=0.01] = 0.0

a2 = s_hi.copy()
a2[a2 >=0.03] = 2.0
a2[(a2 < 0.03) & (a2 >= 0.01)] = 1.0
a2[a2 <=0.01] = 0.0
plt.figure()
plt.subplot(221)
sns.heatmap(a1, cmap='PiYG',cbar=True, xticklabels=False, yticklabels=False); plt.title('wetness class')
plt.subplot(222)
sns.heatmap(a2, cmap='PiYG',cbar=True, xticklabels=False, yticklabels=False); plt.title('wetness class')



# plot LAI -relations

plt.figure()
x = LAId + LAIc

with sns.color_palette('muted'):
    plt.subplot(221)
    plt.plot(x[peat_ix], np.sum(ET, axis=0)[peat_ix]/P, 'o', alpha=0.3, label='peat')
    plt.plot(x[med_ix], np.sum(ET, axis=0)[med_ix]/P, 'o', alpha=0.3, label='medium textured')
    plt.plot(x[coarse_ix], np.sum(ET, axis=0)[coarse_ix]/P, 'o', alpha=0.3, label='coarse textured')
    plt.ylabel('ET/P'); plt.xlabel('LAI (m$^2$m$^{-2}$)')
    plt.legend(loc=4)

    plt.subplot(222)
    plt.plot(x[peat_ix], np.sum(TR, axis=0)[peat_ix]/P, 'o', alpha=0.3, label='peat')
    plt.plot(x[med_ix], np.sum(TR, axis=0)[med_ix]/P, 'o', alpha=0.3, label='medium textured')
    plt.plot(x[coarse_ix], np.sum(TR, axis=0)[coarse_ix]/P, 'o', alpha=0.3, label='coarse textured')
    plt.ylabel('T$_r$/P'); plt.xlabel('LAI (m$^2$m$^{-2}$)')
    # plt.legend()

    plt.subplot(223)
    plt.plot(x[peat_ix], np.sum(IE, axis=0)[peat_ix]/P, 'o', alpha=0.3, label='peat')
    plt.plot(x[med_ix], np.sum(IE, axis=0)[med_ix]/P, 'o', alpha=0.3, label='medium textured')
    plt.plot(x[coarse_ix], np.sum(IE, axis=0)[coarse_ix]/P, 'o', alpha=0.3, label='coarse textured')
    plt.ylabel('E/P'); plt.xlabel('LAI (m$^2$m$^{-2}$)')
    # plt.legend()
    
    plt.subplot(224)
    plt.plot(x[peat_ix], np.sum(EF, axis=0)[peat_ix]/P, 'o', alpha=0.3, label='peat')
    plt.plot(x[med_ix], np.sum(EF, axis=0)[med_ix]/P, 'o', alpha=0.3, label='medium textured')
    plt.plot(x[coarse_ix], np.sum(EF, axis=0)[coarse_ix]/P, 'o', alpha=0.3, label='coarse textured')
    # plt.plot(x[htwi_ix], np.sum(EF, axis=0)[htwi_ix]/P, 's', alpha=0.3, label='wetcells')
    plt.ylabel('E$_f$/P'); plt.xlabel('LAI (m$^2$m$^{-2}$)')
    # plt.legend()
    
    plt.figure()
    y = LAId / (LAIc + LAId)
    
    plt.plot(x[peat_ix], y[peat_ix], 'o', alpha=0.3, label='peat')
    plt.plot(x[med_ix], y[med_ix], 'o', alpha=0.3, label='peat')
    plt.plot(x[coarse_ix], y[coarse_ix], 'o', alpha=0.3, label='peat')

#%% plot figure of relative max swe

snw = SWE[swe_max_ix,:,:]
laiw = LAId * 0.1 + LAIc  # wintertime lai
f = snw > 0
snw = snw[f]
laiw = laiw[f]
vol = spa.GisData['vol']
vol = vol[f]
g = np.where(laiw == min(laiw))
snw_max = np.mean(snw[g])
plt.figure()
plt.subplot(211)
plt.plot(laiw, snw / snw_max , 'ro'); plt.xlabel('LAI')
plt.subplot(212)
plt.plot(vol, snw / snw_max , 'go'); plt.xlabel('vol')

#r = np.size(LAI)
#x = np.reshape(LAIc + LAId,r)
#tr = np.reshape(np.sum(TR, axis=0) / P, r)
#et = np.reshape(np.sum(TR + EF + IE, axis=0) / P, r)
#ef = np.reshape(np.sum(EF, axis=0) / P, r)
#ie = np.reshape(np.sum(IE, axis=0) / P, r)
#c = np.reshape(LAId / (LAIc + LAId), r)
#s = np.reshape(soil, r)
#
#
#dat = {'LAI': x, 'dfrac': c, 'TR': tr, 'ET': et, 'IE': ef,
#       'IE': ie, 'soil': s}
#       
#data = pd.DataFrame(dat, index=np.arange(r))
#data = data.dropna()
#
#plt.figure()
#sns.stripplot(x='LAI', y='ET', data=data, hue='soil')