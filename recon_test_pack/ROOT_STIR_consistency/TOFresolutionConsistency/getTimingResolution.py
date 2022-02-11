import numpy as npy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal

def extract_timeStamps(filename):
    with open(filename,'r') as f:
        line_content = f.readlines()
    f.close()
    line_content = [x.split() for x in line_content]
    line_content = [[x[6],x[29],x[12],x[35]] for x in line_content ]
    return line_content   

def get_timeDifference(time_list):
    timeDiff=[]
    for x in time_list:
        if (int(x[2])<int(x[3])):
            timeDiff.append((float(x[1])-float(x[0]))*10**12) #in ps
        else:
            timeDiff.append((float(x[0])-float(x[1]))*10**12) #in ps
    return timeDiff

def sample_timeDifference_distribution(timeDiff):
    timeDiff.sort()
    timeDiff=npy.around(timeDiff,decimals=3)
    rangemin=timeDiff[0]-0.001
    rangemax=timeDiff[-1]+0.001
    x=npy.linspace(rangemin,rangemax,10000)
    y=npy.zeros(npy.size(x))
    for deltaT in timeDiff:
        nb=0
        for xindex in x:
            if (deltaT < xindex):
                y[nb]=y[nb]+1
                break
            nb=nb+1
    #recentering:
    x=x-(x[1]-x[0])/2
    return [x,y]
    
def plot_timeDifference_distribution(timedifferencearray,nbofevents):
    plt.plot(timedifferencearray,nbofevents)
    plt.xlabel("Difference in arrival time (ps)")
    plt.ylabel("Number of events")
    plt.title("Time difference distribution")
    plt.legend()
    
def gauss_function(x, a, x0, sigma):
    return (a/(sigma*npy.sqrt(npy.pi)))*npy.exp(-(x-x0)**2/(2*sigma**2))    
    
def get_gaussianFit(timedifferencearray,nbofevents):
    m = 0#sum(timedifferencearray * nbofevents)/len(timedifferencearray)
    sigma = 233.55#npy.sqrt(sum(nbofevents * (timedifferencearray - m)**2))
    params = curve_fit(gauss_function, timedifferencearray, nbofevents, p0=[1,m,sigma])
    return [gauss_function(timedifferencearray, params[0][0],params[0][1],params[0][2]),params]
 
def get_FWHM(sigma):
    return sigma*2*npy.sqrt(2*npy.log(2))

def response_noTimeRes(x, a, x0, b):
    y=npy.zeros(npy.size(x))
    nb=0
    for xindex in x:
        if (xindex>(x0-b) and xindex<x0):
            y[nb]=xindex*a/b +a
        if (xindex>= x0 and xindex <(x0+b)):
            y[nb]=a-xindex*a/b
        nb=nb+1
    return y

def get_responseFit(timedifferencearray,nbofevents):
    m=0
    a=12000
    b=100
    params = curve_fit(response_noTimeRes, timedifferencearray, nbofevents, p0=[a,m,b])
    return [response_noTimeRes(timedifferencearray, params[0][0],params[0][1],params[0][2]),params]

def reload_data_timing(name):
    npzfile=npy.load(name)
    # Order should be: timeDiff, nbEvents, nbEventsFit, firstparamfit, secondparamfit, tirdparamfit
    return [npzfile[npzfile.files[0]],npzfile[npzfile.files[1]],npzfile[npzfile.files[2]],npzfile[npzfile.files[3]],npzfile[npzfile.files[4]],npzfile[npzfile.files[5]]]

#%%
time_list = extract_timeStamps("Output_centreCoincidences.dat")
timeDiff=get_timeDifference(time_list)
[sampledTimeDiffTim,nbEventsTim]=sample_timeDifference_distribution(timeDiff)
plot_timeDifference_distribution(sampledTimeDiffTim,nbEventsTim)
[nbEventsFitTim,paramsTim] = get_gaussianFit(sampledTimeDiffTim,nbEventsTim)
plot_timeDifference_distribution(sampledTimeDiffTim,nbEventsFitTim)
FWHM=get_FWHM(paramsTim[0][2])

npy.savez("centreWithTimeResolution.npz",sampledTimeDiffTim=sampledTimeDiffTim,
          nbEventsTim=nbEventsTim,nbEventsFitTim=nbEventsFitTim, a=paramsTim[0][0],x0=paramsTim[0][1],sigma=paramsTim[0][2])
#%%
time_list = extract_timeStamps("Output_centre2Coincidences.dat")
timeDiff=get_timeDifference(time_list)
[sampledTimeDiffNoTim,nbEventsNoTim]=sample_timeDifference_distribution(timeDiff)
plt.figure()
plot_timeDifference_distribution(sampledTimeDiffNoTim,nbEventsNoTim)
[nbEventsFitNoTim,paramsNoTim]= get_responseFit(sampledTimeDiffNoTim,nbEventsNoTim)
plot_timeDifference_distribution(sampledTimeDiffNoTim,nbEventsFitNoTim)

npy.savez("centreWithoutTimeResolution.npz",sampledTimeDiffNoTim=sampledTimeDiffNoTim,
          nbEventsNoTim=nbEventsNoTim,nbEventsFitNoTim=nbEventsFitNoTim, a=paramsNoTim[0][0],x0=paramsNoTim[0][1],b=paramsNoTim[0][2])
