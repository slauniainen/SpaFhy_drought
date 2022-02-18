# -*- coding: utf-8 -*-
"""
testing raster processing

Created on Wed Feb 16 12:43:37 2022

@author: 03081268
"""
import rasterio
from rasterio.plot import show
import os
from pathlib import Path
import numpy as np
from PIL import Image
from raster_utils import convert_peruskarttarasteri, show_raster

# Data dir
data_dir = r"c:\projects\tram\data\pkrasteri"

fps = list(Path(data_dir).glob('*.png'))

# convert png to geotiff
outfiles = []
for fp in fps:
    print(fp)
    f = convert_peruskarttarasteri(str(fp), epsg_code='3067')
    outfiles.append(f)

    
# # crs
# #crs = rasterio.crs.CRS.from_epsg(3067) # ETRS-TM35FIN
# crs = rasterio.crs.CRS({"init": "epsg:3067"}) # ETRS-TM35FIN
# # Open the file:
# r = rasterio.open(fp, 'r+')
# #r = rasterio.open('testi.tif', 'r+')
# r.crs = crs 
# info = r.meta
# r.close()
# # -- read data

# pkras = r.read()

#%%

# img = Image.open(fp)
# rgbimg = Image.new("RGB", img.size)
# rgbimg.paste(img)
# rgbimg.save('testi.tif')