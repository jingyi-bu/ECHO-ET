# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:08:04 2023

@author: bujin
"""
import rasterio
import numpy as np
import glob,os
import pandas as pd
from tqdm import tqdm
import re
from datetime import datetime
from osgeo import gdal


df = pd.read_excel('para.xlsx')

# et_folder='I:\\ET_RF_2020_s_update'
et_folder='G:\\ET_RF_2018_s_update\\New folder'
year='A2018'

file_et=[f for f in os.listdir(et_folder) if f.startswith('ECO_ET_hourly_s_2018') and f.endswith('.tif')]

ssrd_folder='F:\\Data\\ERA5land_01\\ssrd'

folder_path = './nirv_indicator/'
files_nirv = [f for f in os.listdir(folder_path) if year in f and f.endswith('.tif')]
file_nirv_old = 'example'
output_extent = [-180, -90, 180, 90]

with rasterio.open('MCD12C1_LUCC_Type1.tif') as src:
    lucc=src.read(1).astype(np.float32)    

for file in tqdm(file_et):
    with rasterio.open(os.path.join(et_folder, file)) as src:
        et_ori=src.read(1)
        src_meta = src.meta
    
    file_ssrd=file.replace('ECO_ET_hourly_s','ERA_Land_hourly_ssrd_s')
    
    outfile_ssrd="./tem/" +file_ssrd
    gdal.Warp(outfile_ssrd, os.path.join(ssrd_folder, file_ssrd), format="GTiff", dstSRS="EPSG:4326", targetAlignedPixels='MCD12C1_LUCC_Type1.tif', xRes=0.05,
          yRes=0.05, outputBounds=output_extent, resampleAlg=gdal.GRA_NearestNeighbour)    
    
    
    with rasterio.open(outfile_ssrd) as src:
        ssrd_ori=src.read(1)   
        ssrd_ori[ssrd_ori<= 5]=0
        ssrd_ori[ssrd_ori> 5]=1
        

    date_match = re.search(r'(\d{10})', file)
    if date_match:
        date_str = date_match.group(1)  
        date_obj = datetime.strptime(date_str, '%Y%m%d%H')
        year_str = date_obj.strftime('%Y')
        doy_str = date_obj.strftime('%j')
        file_nirv = f"MCD43C4_A{year_str}{doy_str}_nirv_indicator.tif"
        file_nirv_up=os.path.join(folder_path, file_nirv)
    
    if file_nirv_up != file_nirv_old:
        with rasterio.open(file_nirv_up) as src:
            nirv_ori=src.read(1)
            nirv_ori[:] = nirv_ori.round(1)
            nirv_ori=nirv_ori*10
        file_nirv_old=file_nirv_up
 
    temp_df = pd.DataFrame({
        'Value': lucc.ravel(),
        'NIRv_num': nirv_ori.ravel(),
        'day_night_num': ssrd_ori.ravel()
    })
    

    merged = pd.merge(temp_df, df, how='left', on=['Value', 'NIRv_num', 'day_night_num'])
    
    shape = lucc.shape
    best_k_result = merged['best_k'].values.reshape(shape)
    best_b_result = merged['best_b'].values.reshape(shape)
    
    best_k_result[np.isnan(best_k_result)] = 1
    best_b_result[np.isnan(best_b_result)] = 0        
 

    et_update=best_k_result*et_ori+best_b_result
    et_update[et_ori==-32767]=-32767
    
    output_filepath ='./ET_update/'+file.replace('ECO_ET_hourly_s','ECO_ET_hourly_s_corr')
    with rasterio.open(output_filepath, 'w', **src_meta) as dst:
        dst.write(et_update, 1)    

    os.remove(outfile_ssrd)