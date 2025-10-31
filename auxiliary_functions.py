import pygmsh
import numpy as np
from collections import defaultdict
from constants import G, au, speed_of_light, sigma, S0
from scipy.optimize import fsolve
from scipy.interpolate import griddata
import time
from datetime import timedelta


def imitate_sample(X, n_bins, N):
    '''
    Takes arbitrary array X and generates sample of N elements that imitates the 
    distribution of X uses inverse transform sampling method and spline interpoalation. 
    This can be used to generate sample of arbitrary size of, say, albedos that 
    imitates some known sample of albedos, e.g. from WISE.
    
    input:
        X - array whose distribution is imitated
        nbins - number of bins for array X to make discrete distribution
        N - number of elements in the output sample
        
    output:
        sample - sample of N elements with the distribution similar to X 
    '''

    # binning data
    y, xx= np.histogram(X, bins=n_bins)
    x = np.zeros(len(y))
    
    for i in range(len(x)):
        x[i]=(xx[i+1]+xx[i])/2
    
 
    # natural spline interpolation
    a=y[:-1]
    b=np.zeros(len(a), dtype='float')
    d=np.zeros(len(a), dtype='float')
    h=np.zeros(len(x)-1, dtype='float')
    for i in range(0,len(x)-1):
        h[i]=x[i+1]-x[i]
        
    A=np.zeros([len(x), len(x)], dtype='float')
    v=np.zeros(len(x))
    
    for i in range(2,len(A)):
    
        A[i-1,i-1]=2*(h[i-2]+h[i-1])
        A[i,i-1]=h[i-1]
        A[i-2,i-1]=h[i-2]
        
        v[i-1]=3*((y[i]-y[i-1])/h[i-1]-(y[i-1]-y[i-2])/h[i-2])
    
    A[0,0]=1; A[-1][-1]=1; A[-1,-2]=0; A[0,1]=0; A[1,0]=h[0]; A[-2,-1]=h[-1] 
                  
    c = np.linalg.solve(A,v)
    
    for i in range(len(a)):
        b[i]=(y[i+1]-y[i])/h[i]-h[i]/3*(2*c[i]+c[i+1])
        d[i]=(c[i+1]-c[i])/3/h[i]
        
    c=c[:-1]
    Y0=np.zeros(len(x)) # integral in nodes
    
    for i in range(1,len(x)):
        Y0[i]=np.polyval([d[i-1]/4,c[i-1]/3,b[i-1]/2,a[i-1],Y0[i-1]],x[i]-x[i-1]) # integral in nodes
    
    # Generating sample using Inverse Transform Sampling
    r=np.random.random(N)*np.max(Y0) # uniform sample
    sample=np.zeros_like(r) # output sample
    
    for i in range(len(r)):
        ind=np.argwhere(Y0<=r[i])[-1]
        if ind==len(Y0):
            ind=ind-1
        
        t=np.roots((np.array([d[ind]/4,c[ind]/3,b[ind]/2,a[ind],Y0[ind]-r[i]])).flatten()) 
        t=t[(np.argwhere(np.imag(t)==0)).flatten()] 
        t=t[(np.argwhere(t.real>0)).flatten()]
        t=t[(np.argwhere(t.real<x[ind+1]-x[ind])).flatten()]
        t=t+x[ind]
        sample[i]=np.real(t)    
    
    return sample


def sample_from_curve(x,y, N):
    '''
    Takes arbitrary array X and generates sample of N elements that imitates the 
    distribution of X uses inverse transform sampling method and spline interpoalation. 
    This can be used to generate sample of arbitrary size of, say, albedos that 
    imitates some known sample of albedos, e.g. from WISE.
    
    input:
        X - array whose distribution is imitated
        nbins - number of bins for array X to make discrete distribution
        N - number of elements in the output sample
        
    output:
        sample - sample of N elements with the distribution similar to X 
    '''
    # natural spline interpolation
    a=y[:-1]
    b=np.zeros(len(a), dtype='float')
    d=np.zeros(len(a), dtype='float')
    h=np.zeros(len(x)-1, dtype='float')
    for i in range(0,len(x)-1):
        h[i]=x[i+1]-x[i]
        
    A=np.zeros([len(x), len(x)], dtype='float')
    v=np.zeros(len(x))
    
    for i in range(2,len(A)):
    
        A[i-1,i-1]=2*(h[i-2]+h[i-1])
        A[i,i-1]=h[i-1]
        A[i-2,i-1]=h[i-2]
        
        v[i-1]=3*((y[i]-y[i-1])/h[i-1]-(y[i-1]-y[i-2])/h[i-2])
    
    A[0,0]=1; A[-1][-1]=1; A[-1,-2]=0; A[0,1]=0; A[1,0]=h[0]; A[-2,-1]=h[-1] 
                  
    c = np.linalg.solve(A,v)
    
    for i in range(len(a)):
        b[i]=(y[i+1]-y[i])/h[i]-h[i]/3*(2*c[i]+c[i+1])
        d[i]=(c[i+1]-c[i])/3/h[i]
        
    c=c[:-1]
    Y0=np.zeros(len(x)) # integral in nodes
    
    for i in range(1,len(x)):
        Y0[i]=np.polyval([d[i-1]/4,c[i-1]/3,b[i-1]/2,a[i-1],Y0[i-1]],x[i]-x[i-1]) # integral in nodes
    
    # Generating sample using Inverse Transform Sampling
    r=np.random.random(N)*np.max(Y0) # uniform sample
    sample=np.zeros_like(r) # output sample
    
    for i in range(len(r)):
        ind=np.argwhere(Y0<=r[i])[-1]
        if ind==len(Y0):
            ind=ind-1
        
        t=np.roots((np.array([d[ind]/4,c[ind]/3,b[ind]/2,a[ind],Y0[ind]-r[i]])).flatten()) 
        t=t[(np.argwhere(np.imag(t)==0)).flatten()] 
        t=t[(np.argwhere(t.real>0)).flatten()]
        t=t[(np.argwhere(t.real<x[ind+1]-x[ind])).flatten()]
        t=t+x[ind]
        sample[i]=np.real(t)    
    
    return sample

def true2ecc(f, e):
    # =============================================================================
    # converts true anomaly to eccentric (or hyperbolic) anomaly
    # Input:
    # f [radians] - true anomaly
    # Output:
    # eccentric (or hyperbolic anomaly) [radians]
    # =============================================================================
    if e > 1:
        return 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(f / 2))

    else:
        return np.arctan2(np.sqrt(1 - e ** 2) * np.sin(f), e + np.cos(f))