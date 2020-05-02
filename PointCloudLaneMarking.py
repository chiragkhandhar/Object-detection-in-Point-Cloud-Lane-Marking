#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import utm
from shapely import wkt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import import_ipynb
from PointCloudVisualization import convert_fusedata,readFusedata_toDF
from PointCloudFiltering import readXYZdata_toDF, readxyz
import random


# In[40]:


dfLanes = readXYZdata_toDF("./filter_trajectory.xyz",ignoreline = True, delimeter = ",")
Traj = dfLanes[["East", "North", "Altitude"]].values
Traj = StandardScaler().fit_transform(Traj)
db = DBSCAN(eps=0.06, min_samples=30).fit(Traj)
labels = db.labels_
nClusters = len(set(labels)) - (1 if -1 in labels else 0)
nNoise = list(labels).count(-1)
dfLanes["Group"] = labels
dfClusters = dfLanes[["East","North","Altitude","Group"]]
dfClusters = dfClusters[dfClusters["Group"]>=0]
dfClusters.to_csv("./clustering.xyz",index=False)
print("Total Number of Clusters:           ", nClusters)
print("Total Number of Noise points:       ", nNoise)
print("Total Number of clustered points:   ", len(dfClusters))


# In[41]:


lines = []
for cluster in range(nClusters):
    sub_dfClusters = dfClusters[dfClusters["Group"] == cluster]
    points = sub_dfClusters[["East","North", "Altitude"]].values
    distances = squareform(pdist(points))
    for i in range(0,15):
        max_index = np.argmax(distances)
        i1, i2 = np.unravel_index(max_index, distances.shape)
        distances[i1,i2] = 0.0
    max_dist = np.max(distances)
    max_index = np.argmax(distances)
    i1, i2 = np.unravel_index(max_index, distances.shape)
    p1 = sub_dfClusters.iloc[i1]
    p2 = sub_dfClusters.iloc[i2]
    lines.append(([p1["East"],p2["East"]],[p1["North"],p2["North"]],[p1["Altitude"],p2["Altitude"]]))


# In[42]:


plt.figure(figsize=(15,5))
plt.xlim(70, 130), plt.ylim(20,120)
colors = ["red","blue","green","purple","orange","yellow","lightblue","black"]
for l in lines:
    plt.plot(l[0], l[1], l[2],color = colors[random.randint(0,7)])    
# plt.show()
plt.savefig('Prototype_LaneMarking1.png')
print("Number of lane markings: ", len(lines))


# In[43]:


def createLine(coords):
    return wkt.loads("LINESTRING("+str(coords[0][0])+" "+str(coords[1][0])+", " +str(coords[0][1])+ " " +str(coords[1][1])+")")    
distMatrix = []
for l1 in range(len(lines)):
    prototype = []
    distances = []
    line1 = createLine(lines[l1])
    for l2 in range(len(lines)):
        line2 = createLine(lines[l2])
        d = line1.distance(line2)
        if d == 0:
            d = 100
        distances.append(d)        
    distMatrix.append(distances)
distances = np.array(distMatrix)
pass


# In[44]:


fusion = set()
for row in range(distances.shape[0]):
    for column in range(distances.shape[1]):
        value = distances[row,column]
        if value < 1.0:
            min_index = min(row, column)
            max_index = max(row, column)
            fusion.add((min_index, max_index))            
fusion = list(fusion)
fClusters = []
finsih = []
fusion.sort()
for i in range(len(fusion)):
    new_cluster = list(fusion[i])
    if new_cluster[0] not in finsih and new_cluster[1] not in finsih:
        for j in range(i+1, len(fusion)):
            f = list(fusion[j])
            if (f[0] in new_cluster or f[1] in new_cluster) and f[0] not in finsih:
                new_cluster.extend(f)
        fClusters.append(list(set(new_cluster)))
        finsih.extend(list(set(new_cluster)))        
fused_lines = [item for sublist in fClusters for item in sublist]
for i in range(distances.shape[0]):
    if i not in fused_lines:
        fClusters.append([i])


# In[45]:


final_LaneM = []
for cluster in fClusters:
    if len(cluster) == 1:
        final_LaneM.append(lines[cluster[0]])
    else:
        points = []
        for segment in cluster:
            l = lines[segment]
            points.append([l[0][0], l[1][0], l[2][0]])
            points.append([l[0][1], l[1][1], l[2][1]])
        points = np.array(points)
        distances = squareform(pdist(points))
        max_dist = np.max(distances)
        max_index = np.argmax(distances)
        i1, i2 = np.unravel_index(max_index, distances.shape)        
        p1 = points[i1]
        p2 = points[i2]
        final_LaneM.append(([p1[0], p2[0]],[p1[1], p2[1]], [p1[2], p2[2]]))


# In[46]:


plt.figure(figsize=(15,5))
plt.xlim(70, 130), plt.ylim(20,120)
for l in final_LaneM:
    plt.plot(l[0], l[1], l[2], color = colors[random.randint(0,7)])    
plt.savefig('Prototype_LaneMarking2.png')
print("Total Number of final lane markings: ", len(final_LaneM))


# In[47]:


op_rows = []
x_min,y_min,z_min,number,letter = readxyz("minvalues1.csv", extra = ["number","letter"])
for line in final_LaneM:
    xstart = line[0][0] + x_min
    ystart = line[1][0] + y_min
    zstart = line[2][0] + z_min
    xend = line[0][1] + x_min
    yend = line[1][1] + y_min
    zend = line[2][1] + z_min
    (start_lat, start_lon) = utm.to_latlon(xstart,ystart,number, letter)
    (end_lat, end_lon) = utm.to_latlon(xend,yend,number, letter)
    op_rows.append([start_lat, start_lon, zstart, end_lat, end_lon, zend])    
res = pd.DataFrame(op_rows, columns=[" Start_Latitude", "Start_Longitude", "Start_Altitude","End_latitude", "End_Longitude", "End_Altitude"])
res.to_csv("./res.csv", header=True, index=False)

