
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from math import log

const_LambdaList = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

labels = ["N Cities=500", "N Cities=5,000", "N Cities=50,000", "N Cities=500,000"]
df = pd.read_csv("ClusterTimingData.csv", header=None, skiprows=1)
myTitle = "AWS Spark Cluster Performance (1 Master Node, N Slave Nodes)"
myYlabel = "execution time in seconds"
myYlabel_log = "execution time in log(seconds)"
myXlabel = "number of slave nodes (m4.large = 2 vCPU, 8GB Mem)"

cities500 = df[0:6]
cities5000 = df[6:12]
cities50000 = df[12:18]
cities500000 = df[18:24]

fig, ax = plt.subplots()

x1=cities500[1]
y1=np.log(cities500[2])

x2=cities5000[1]
y2=np.log(cities5000[2])

x3=cities50000[1]
y3=np.log(cities50000[2])

x4=cities500000[1]
y4=np.log(cities500000[2])

#plt.figure(1, figsize=(16,16))

ax.plot(x1,y1, marker='o')
ax.plot(x2,y2, marker='o')
ax.plot(x3,y3, marker='o')
ax.plot(x4,y4, marker='o')

plt.title(myTitle)
plt.xlabel(myXlabel)
plt.ylabel(myYlabel_log)

legend = ax.legend(labels, loc="upper right", shadow=True)
#legend.get_frame().set_facecolor('#00FFCC')
plt.show()

fig, ax = plt.subplots()
x1=cities500[1]
y1=cities500[2]

x2=cities5000[1]
y2=cities5000[2]

x3=cities50000[1]
y3=cities50000[2]

#x4=cities500000[1]
#y4=cities500000[2]

#plt.figure(1, figsize=(16,16))
newlabels = labels[0:3]

ax.plot(x1,y1, marker='o')
ax.plot(x2,y2, marker='o')
ax.plot(x3,y3, marker='o')
#ax.plot(x4,y4, marker='o')

plt.title(myTitle)
plt.xlabel(myXlabel)
plt.ylabel(myYlabel)

legend = ax.legend(newlabels, loc="upper right", shadow=True)
plt.show()


fig, ax = plt.subplots()
x1=cities500[1]
y1=cities500[2]

x2=cities5000[1]
y2=cities5000[2]

x3=cities50000[1]
y3=cities50000[2]

x4=cities500000[1]
y4=cities500000[2]

#plt.figure(1, figsize=(16,16))

ax.plot(x1,y1, marker='o')
ax.plot(x2,y2, marker='o')
ax.plot(x3,y3, marker='o')
ax.plot(x4,y4, marker='o')

plt.title(myTitle)
plt.xlabel(myXlabel)
plt.ylabel(myYlabel)

legend = ax.legend(labels, loc="upper right", shadow=True)
plt.show()