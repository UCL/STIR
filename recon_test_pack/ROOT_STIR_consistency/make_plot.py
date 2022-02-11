import numpy as npy
import matplotlib.pyplot as plt
import itertools

def extract_coordinates(filename):
    with open(filename,'r') as f:
        line_content = f.readlines()
    f.close()
    line_content = [x.split() for x in line_content]
    xlist = [x[0] for x in line_content ]
    ylist = [x[1] for x in line_content ]
    zlist = [x[2] for x in line_content ]
    weightlist = [x[3] for x in line_content ]
    return [xlist,ylist,zlist,weightlist]

def extract_coordinates2(filename):
    with open(filename,'r') as f:
        line_content = f.readlines()
    f.close()
    line_content = [x.split() for x in line_content]
    listcont = [[x[0],x[1],x[2]] for x in line_content ]
    return listcont


def make_scatter_plot(xlist,ylist):
    plt.scatter(xlist,ylist)
    plt.xlabel("x value (mm)")
    plt.ylabel("y value (mm)")
    plt.title("TOF kernel")
    plt.legend()
    
def threshold_values(xsource,ysource,zsource,listcont):
    distance=[]
    boollist=[]
    for x in listcont:
        distance.append(npy.sqrt(npy.power(float(x[0])-xsource,2)+npy.power(float(x[1])-ysource,2)+npy.power(float(x[2])-zsource,2)))
    for d in distance:
        boollist.append(d > 45)
    return boollist

#%%

[xlist,ylist,zlist,weightlist] = extract_coordinates("stir_image1.txt")
plt.figure()
make_scatter_plot(xlist,ylist)
#%%
[xlist,ylist,zlist,weightlist] = extract_coordinates("stir_image2.txt")
plt.figure()
make_scatter_plot(xlist,ylist)
#%%
[xlist,ylist,zlist,weightlist] = extract_coordinates("stir_image3.txt")
plt.figure()
make_scatter_plot(xlist,ylist)
#%%
[xlist,ylist,zlist,weightlist] = extract_coordinates("stir_image4.txt")
plt.figure()
make_scatter_plot(xlist,ylist)
#%%
[xlist,ylist,zlist,weightlist] = extract_coordinates("stir_image5.txt")
plt.figure()
make_scatter_plot(xlist,ylist)
#%%
[xlist,ylist,zlist,weightlist] = extract_coordinates("stir_image6.txt")
plt.figure()
make_scatter_plot(xlist,ylist)
#%%
[xlist,ylist,zlist,weightlist] = extract_coordinates("stir_image7.txt")
plt.figure()
make_scatter_plot(xlist,ylist)
#%%
[xlist,ylist,zlist,weightlist] = extract_coordinates("stir_image8.txt")
plt.figure()
make_scatter_plot(xlist,ylist)
#%%
[xlist,ylist,zlist,weightlist] = extract_coordinates("stir_image9.txt")
plt.figure()
make_scatter_plot(xlist,ylist)
#%%
[xlist,ylist,zlist,weightlist] = extract_coordinates("stir_image10.txt")
plt.figure()
make_scatter_plot(xlist,ylist)
#%%
[xlist,ylist,zlist,weightlist] = extract_coordinates("stir_image11.txt")
plt.figure()
make_scatter_plot(xlist,ylist)
#%%
[xlist,ylist,zlist,weightlist] = extract_coordinates("stir_image12.txt")
plt.figure()
make_scatter_plot(xlist,ylist)
#%%