# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:09:40 2023

@author: caubu
"""

from osgeo import gdal
import numpy as np
import numpy.ma as ma
import os
import joblib
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool, cpu_count


# wd=os.getcwd()
# filelist_t2m=glob.glob(wd+"\\t2m\\*.tif")
# filelist_d2m=glob.glob(wd+"\\d2m\\*.tif")

geo= "I:/MCD43A3/NIRv/geo/MCD12C1_LUCC_Majority_Land_Cover_Type_1.tif" 
#geo = "D:/UNH_MCD12C1/MCD12C1_LUCC_Majority_Land_Cover_Type_1.tif"
output_extent = [-180, -90, 180, 90]

# Load the saved model from a file
model = joblib.load("RF.joblib")

# Define the feature names
feature_names = ['NIRv', 't2m', 'vpd', 'ssrd', 'sw', 'skt']
    
# Set the path to the folders containing the geotiffs

folder_skt = './ERA5/skt'
folder_ssrd = './ERA5/ssrd'
folder_sw = './ERA5/sw'
folder_t2m = './ERA5/t2m/New folder'
folder_vpd = './ERA5/vpd'
folder_nirv = './MCD43A3/NIRv_filled_final'

# Get a list of the geotiff files in each folder
# filelist_t2m = glob.glob(folder_t2m+"/ERA_Land_hourly_t2m_2020*.tif") 
# filelist_skt = glob.glob(folder_skt+"/ERA_Land_hourly_skt_2020*.tif") 
# filelist_ssrd = glob.glob(folder_ssrd+"/ERA_Land_hourly_ssrd_2020*.tif") 
# filelist_sw = glob.glob(folder_sw+"/ERA_Land_hourly_sw_2020*.tif") 
# filelist_vpd = glob.glob(folder_vpd+"/ERA_Land_hourly_vpd_2020*.tif") 
# filelist_nirv = glob.glob(folder_nirv+"/MCD43C4_A2020*.tif") 

# filelist_t2m = [f for f in os.listdir(folder_t2m) if f.endswith('.tif')]
# filelist_skt = [f for f in os.listdir(folder_skt) if f.endswith('.tif')]
# filelist_ssrd = [f for f in os.listdir(folder_ssrd) if f.endswith('.tif')]
# filelist_sw = [f for f in os.listdir(folder_sw) if f.endswith('.tif')]
# filelist_vpd = [f for f in os.listdir(folder_vpd) if f.endswith('.tif')]
# filelist_nirv = [f for f in os.listdir(folder_nirv) if f.endswith('.tif')]

filelist_t2m = [f for f in os.listdir(folder_t2m) if (f.endswith('.tif') and f.startswith('ERA_Land_hourly_t2m_2022'))]
filelist_skt = [f for f in os.listdir(folder_skt) if (f.endswith('.tif') and f.startswith('ERA_Land_hourly_skt_2022'))]
filelist_ssrd = [f for f in os.listdir(folder_ssrd) if (f.endswith('.tif') and f.startswith('ERA_Land_hourly_ssrd_2022'))]
filelist_sw = [f for f in os.listdir(folder_sw) if (f.endswith('.tif') and f.startswith('ERA_Land_hourly_sw_2022'))]
filelist_vpd = [f for f in os.listdir(folder_vpd) if (f.endswith('.tif') and f.startswith('ERA_Land_hourly_vpd_2022'))]
filelist_nirv = [f for f in os.listdir(folder_nirv) if (f.endswith('.tif') and f.startswith('MCD43C4_A2022'))]



# Iterate through the files in folder A
def process_file(file_t2m):
    # Get the corresponding file in folder B
    file_skt = file_t2m.replace('t2m', 'skt')
    file_ssrd = file_t2m.replace('t2m', 'ssrd')
    file_sw = file_t2m.replace('t2m', 'sw')
    file_vpd = file_t2m.replace('t2m', 'vpd')
    
    dt_object = datetime.strptime(file_t2m.split("_")[-1][0:10], '%Y%m%d%H')
    day_of_year = str(dt_object.year) + str(dt_object.strftime('%j'))
    file_nirv = list(filter(lambda s: day_of_year in s, filelist_nirv))[0]
    
    outfile_t2m="./tem/" +file_t2m.split("/")[-1]
    outfile_skt="./tem/" +file_skt.split("/")[-1]
    outfile_ssrd="./tem/" +file_ssrd.split("/")[-1]
    outfile_sw="./tem/" +file_sw.split("/")[-1]
    outfile_vpd="./tem/" +file_vpd.split("/")[-1]
    
    
    # gdal.Warp(outfile_t2m, os.path.join(folder_t2m, file_t2m), format="GTiff", dstSRS="EPSG:4326", targetAlignedPixels=geo, xRes=0.05,
    #       yRes=0.05, outputBounds=output_extent, resampleAlg=gdal.GRA_Bilinear)
    # gdal.Warp(outfile_skt, os.path.join(folder_skt, file_skt), format="GTiff", dstSRS="EPSG:4326", targetAlignedPixels=geo, xRes=0.05,
    #       yRes=0.05, outputBounds=output_extent, resampleAlg=gdal.GRA_Bilinear)
    # gdal.Warp(outfile_ssrd, os.path.join(folder_ssrd, file_ssrd), format="GTiff", dstSRS="EPSG:4326", targetAlignedPixels=geo, xRes=0.05,
    #       yRes=0.05, outputBounds=output_extent, resampleAlg=gdal.GRA_Bilinear)
    # gdal.Warp(outfile_sw, os.path.join(folder_sw, file_sw), format="GTiff", dstSRS="EPSG:4326", targetAlignedPixels=geo, xRes=0.05,
    #       yRes=0.05, outputBounds=output_extent, resampleAlg=gdal.GRA_Bilinear)
    # gdal.Warp(outfile_vpd, os.path.join(folder_vpd, file_vpd), format="GTiff", dstSRS="EPSG:4326", targetAlignedPixels=geo, xRes=0.05,
    #       yRes=0.05, outputBounds=output_extent, resampleAlg=gdal.GRA_Bilinear)

    gdal.Warp(outfile_t2m, os.path.join(folder_t2m, file_t2m), format="GTiff", dstSRS="EPSG:4326", targetAlignedPixels=geo, xRes=0.05,
          yRes=0.05, outputBounds=output_extent, resampleAlg=gdal.GRA_NearestNeighbour)
    gdal.Warp(outfile_skt, os.path.join(folder_skt, file_skt), format="GTiff", dstSRS="EPSG:4326", targetAlignedPixels=geo, xRes=0.05,
          yRes=0.05, outputBounds=output_extent, resampleAlg=gdal.GRA_NearestNeighbour)
    gdal.Warp(outfile_ssrd, os.path.join(folder_ssrd, file_ssrd), format="GTiff", dstSRS="EPSG:4326", targetAlignedPixels=geo, xRes=0.05,
          yRes=0.05, outputBounds=output_extent, resampleAlg=gdal.GRA_NearestNeighbour)
    gdal.Warp(outfile_sw, os.path.join(folder_sw, file_sw), format="GTiff", dstSRS="EPSG:4326", targetAlignedPixels=geo, xRes=0.05,
          yRes=0.05, outputBounds=output_extent, resampleAlg=gdal.GRA_NearestNeighbour)
    gdal.Warp(outfile_vpd, os.path.join(folder_vpd, file_vpd), format="GTiff", dstSRS="EPSG:4326", targetAlignedPixels=geo, xRes=0.05,
          yRes=0.05, outputBounds=output_extent, resampleAlg=gdal.GRA_NearestNeighbour)
    


    t2m_tif=gdal.Open(outfile_t2m)
    skt_tif=gdal.Open(outfile_skt)
    ssrd_tif=gdal.Open(outfile_ssrd)
    sw_tif=gdal.Open(outfile_sw)
    vpd_tif=gdal.Open(outfile_vpd)
    nirv_tif=gdal.Open(os.path.join(folder_nirv, file_nirv))
    
    NIRv=nirv_tif.ReadAsArray()
    t2m=t2m_tif.ReadAsArray()
    vpd=vpd_tif.ReadAsArray()
    ssrd=ssrd_tif.ReadAsArray()
    sw=sw_tif.ReadAsArray()
    skt=skt_tif.ReadAsArray()

    NIRv[NIRv==32767]=np.nan
    t2m[t2m==-32767]=np.nan
    vpd[vpd==-32767]=np.nan
    ssrd[ssrd==-32767]=np.nan
    sw[sw==-32767]=np.nan
    skt[skt==-32767]=np.nan
    
    ET=NIRv*0
    
    # for j in tqdm(range(t2m.shape[0])):
    #     new_data = np.stack((NIRv[j][:], t2m[j][:], vpd[j][:], ssrd[j][:], sw[j][:], skt[j][:]), axis=-1)
    #     if np.isnan(NIRv[j][:]).all():
    #         ET[j][:]=np.nan
    #     else:
    #         df = pd.DataFrame(data=new_data, columns=feature_names)
    #         ET[j][:] = model.predict(df)
        
    new_data = np.stack((NIRv.flatten(), t2m.flatten(), vpd.flatten(), ssrd.flatten(), sw.flatten(), skt.flatten()), axis=-1)


    # Create a pandas DataFrame with the feature names as the column headers
    df = pd.DataFrame(data=new_data, columns=feature_names)
    
    # save the original dataframe without NaN rows
    df_clean = df.dropna()
    
    # Use the model to make predictions on new data
    ET_predictions = model.predict(df_clean)
    
    ET_full = np.full((len(df)), np.nan)
    
    
    ET_full[df_clean.index] = ET_predictions
    
    ET = ET_full.reshape(NIRv.shape)
    ET = np.nan_to_num(ET, nan=-32767)
    
    
    driver = gdal.GetDriverByName("GTiff")
    outfile="./ET_RF_2022/" +file_t2m.replace('ERA_Land_hourly_t2m', 'ECO_ET_hourly')
    outdataset = driver.Create(outfile, nirv_tif.RasterXSize, nirv_tif.RasterYSize, 1, gdal.GDT_Float32)
    outdataset.SetGeoTransform(nirv_tif.GetGeoTransform())
    outdataset.SetProjection(nirv_tif.GetProjection())
    outdataset.GetRasterBand(1).SetNoDataValue(-32767)
    outdataset.GetRasterBand(1).WriteArray(ET)
    outdataset.FlushCache()

    t2m_tif.FlushCache()
    skt_tif.FlushCache()
    ssrd_tif.FlushCache()
    sw_tif.FlushCache()
    vpd_tif.FlushCache()
    nirv_tif.FlushCache()
    
    del t2m_tif, skt_tif, ssrd_tif, sw_tif, vpd_tif, nirv_tif
    
    os.remove(outfile_t2m)
    os.remove(outfile_skt)
    os.remove(outfile_ssrd)
    os.remove(outfile_sw)
    os.remove(outfile_vpd)

if __name__ == '__main__':
    # Create a pool of worker processes
    # num_processes = cpu_count()
    num_processes = 3
    with Pool(num_processes) as pool:
        # Process each file using the worker processes in parallel
        for _ in tqdm(pool.imap_unordered(process_file, filelist_t2m), total=len(filelist_t2m)):
            pass

