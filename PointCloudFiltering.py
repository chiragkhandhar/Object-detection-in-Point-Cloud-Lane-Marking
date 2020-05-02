#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import utm
from shapely import wkt
import csv
from PointCloudVisualization import convert_fusedata,readFusedata_toDF


# In[2]:


def readxyz(filename,extra = []):
    data = []
    with open(filename) as f: 
        line = f.readline()
        while line:
            line = line.rstrip("\n")
            d = line.split(",")
            data.append(d)
            line = f.readline()
    a = np.array(data)[0] 
    x_min = float(a[0])
    y_min = float(a[1])
    z_min = float(a[2])
    number = letter = 0
    if len(extra) > 0:
        number = float(a[3])
        letter = str(a[4])
    
    return x_min,y_min,z_min,number,letter


# In[3]:


def readXYZdata_toDF(namedf,ignoreline = False, delimeter = " "):
    data = []    
    with open(namedf) as f:  
        line = f.readline()            
        while line:
            if ignoreline == True:
                line = f.readline()
                ignoreline = False
                continue
            d = line.split(delimeter)
            data.append(d)
            line = f.readline()
    a = np.array(data)    
    df_PointCloud = pd.DataFrame()
    df_PointCloud["East"] = pd.to_numeric(a[:,0])
    df_PointCloud["North"] = pd.to_numeric(a[:,1])
    df_PointCloud["Altitude"] = pd.to_numeric(a[:,2])
    df_PointCloud["Intensity"] = pd.to_numeric(a[:,3])

    return df_PointCloud
pass


# In[4]:


def shapeConversion(row):
    return wkt.loads("POINT("+str(row["East"])+" "+str(row["North"])+" " +str(row["Altitude"])+")")
pass


# In[5]:


def filtering_byMeanValue(df_PointCloud):
    mean = df_PointCloud["Intensity"].mean()
    std = df_PointCloud["Intensity"].std()
    dfLanes = df_PointCloud[df_PointCloud["Intensity"] > mean + 1 * std]
    dfLanes = dfLanes[dfLanes["Intensity"] < mean + 7 * std ]
    print("Filtering By Mean:")
    print("===============================================================================")
    print("Mean value for Intensity:      ", mean)
    print("Std value for Intensity:       ", std)
    print("Lower bound for Intensity:     ", mean + 1 * std)
    print("Upper bound for Intensity:     ", mean + 7 * std)
    print("Filtered points for Intensity: ", len(dfLanes))
    print("Original points for Intensity: ", len(df_PointCloud))
    print("Percentage Reduction for Intensity:  ", (len(dfLanes)/len(df_PointCloud))*100)    
    return dfLanes
pass


# In[6]:


def getTrajectoryLine(traj_file='./final_project_data/trajectory.fuse',x_min=0.0,y_min=0.0,z_min=0.0):
    dfTraj = readFusedata_toDF(traj_file)
    x_min,y_min,z_min,number,letter = readxyz("minvalues.csv")
    dfTraj, (x_min,y_min,z_min),(number,letter)=convert_fusedata(dfTraj,x_min,y_min,z_min)
    dfTraj[["East", "North", "Altitude", "Intensity"]].to_csv("./trajectory.xyz", index=False)
    line = "LINESTRING("
    for index,row in dfTraj.iterrows():
        line = line + str(row["East"]) + " " + str(row["North"]) + " " + str(row["Altitude"]) + ", "
    line = line[:-2] + ")"
    trajLine = wkt.loads(line)    
    return trajLine, (x_min, y_min, z_min), (number, letter) 
pass


# In[7]:


def filter_by_trajLine(df_PointCloud, trajLine):
    rows = []
    for index, row in df_PointCloud.iterrows():
        distance = trajLine.distance(row["Shape"])
        if distance <= 20:
            rows.append(row)
    filtered_dfLanes = pd.DataFrame(rows,columns=['East','North','Altitude','Intensity','Shape'])    
    print("\n\n Filtering By Trajectory:")
    print("=================================================================================")
    print("Filtered points for Intensity: ", len(filtered_dfLanes))
    print("Original points for Intensity: ", len(df_PointCloud))
    print("Percentage Reduction for Intensity:  ", (len(filtered_dfLanes)/len(df_PointCloud))*100)
    return filtered_dfLanes
pass


# In[8]:


def main():
    x_min,y_min,z_min,number,letter = readxyz("minvalues.csv")
    trajLine,(x_min, y_min, z_min),(number, letter)=getTrajectoryLine('./final_project_data/trajectory.fuse',x_min, y_min, z_min)
    dfLanes = readXYZdata_toDF('./final_project_point_cloud.xyz')
    dfLanes = filtering_byMeanValue(dfLanes)
    dfLanes[['East', 'North', 'Altitude', 'Intensity']].to_csv("./filter_mean.xyz", index=False)
    dfLanes['Shape'] = dfLanes.apply(shapeConversion, axis=1)
    dfLanes = filter_by_trajLine(dfLanes, trajLine)
    dfLanes[['East', 'North', 'Altitude', 'Intensity']].to_csv("./filter_trajectory.xyz", index=False)
    with open("minvalues1.csv", "w",newline='') as fp:
        wr = csv.writer(fp, delimiter=',')
    #     wr.writerow(["x_min","y_min","z_min"])
        wr.writerow([x_min,y_min,z_min,number, letter])
    pass



if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()


# In[ ]:




