#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import utm
import csv


# In[2]:


def readFusedata_toDF(namedf):
    data = []    
    with open(namedf) as f:  
        line = f.readline()
        while line:
            d = line.split()
            data.append(d)
            line = f.readline()
    a = np.array(data)    
    df_PointCloud = pd.DataFrame()
    df_PointCloud["Latitude"] = a[:,0]
    df_PointCloud["Longitude"] = a[:,1]
    df_PointCloud["Altitude"] = a[:,2]
    df_PointCloud["Intensity"] = a[:,3]

    return df_PointCloud
pass


# In[3]:


def convert_fusedata(df_PointCloud, x_min = 0.0, y_min = 0.0, z_min = 0.0):
    df_PointCloud["Latitude"] = pd.to_numeric(df_PointCloud["Latitude"])
    df_PointCloud["Longitude"] = pd.to_numeric(df_PointCloud["Longitude"])
    df_PointCloud["Altitude"] = pd.to_numeric(df_PointCloud["Altitude"])
    df_PointCloud["Intensity"] = pd.to_numeric(df_PointCloud["Intensity"])
    df_PointCloud["East"] = df_PointCloud.apply(lambda x: utm.from_latlon(x["Latitude"], x["Longitude"])[0], axis = 1)
    df_PointCloud["North"] = df_PointCloud.apply(lambda x: utm.from_latlon(x["Latitude"], x["Longitude"])[1], axis = 1)    
    if y_min == 0:
        y_min = df_PointCloud["North"].min()
    if x_min == 0:
        x_min = df_PointCloud["East"].min()   
    if z_min == 0:
        z_min = df_PointCloud["Altitude"].min()        
    utm_coordinates = utm.from_latlon(df_PointCloud.loc[0,"Latitude"], df_PointCloud.loc[0,"Longitude"])
    zoneN = utm_coordinates[2]
    zoneL = utm_coordinates[3]        
    df_PointCloud["East"] = df_PointCloud["East"] - x_min    
    df_PointCloud["North"] = df_PointCloud["North"] - y_min
    df_PointCloud["Altitude"] = df_PointCloud["Altitude"] - z_min
    
    return df_PointCloud, (x_min, y_min, z_min), (zoneN, zoneL)
pass


# In[4]:


def main():
    df_PointCloud = readFusedata_toDF('./final_project_data/final_project_point_cloud.fuse')
    df_PointCloud, (x_min, y_min, z_min), (number, letter) = convert_fusedata(df_PointCloud)
    df_xyz = df_PointCloud[["East", "North", "Altitude", "Intensity"]]
    df_xyz.to_csv("./final_project_point_cloud.xyz", sep=" ", header=False, index=False)
    with open("minvalues.csv", "w",newline='') as fp:
        wr = csv.writer(fp, delimiter=',')
    #     wr.writerow(["x_min","y_min","z_min"])
        wr.writerow([x_min,y_min,z_min])

    print("FIle one Completed!!")
    # df_xyz.head()
    # df_PointCloud.head()
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()


# In[5]:




