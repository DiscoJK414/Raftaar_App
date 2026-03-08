import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from  trackbound import build
import matplotlib.pyplot as plt


leftb, rightb, centerline = build("cones2.txt", show_plot=False)

def resample(line, N=200):

    d=np.sqrt(np.sum(np.diff(line,axis=0)**2,axis=1))
    s=np.insert(np.cumsum(d),0,0)
    s=s/s[-1]
    fx=interp1d(s,line[:,0])
    fy=interp1d(s,line[:,1])
    snew=np.linspace(0,1,N)
    new = np.zeros((N,2))
    new[:,0] = fx(snew)
    new[:,1] = fy(snew)

    return new

def normals(center):

    t = np.gradient(center,axis=0)
    t = t/np.linalg.norm(t,axis=1,keepdims=True)
    n = np.zeros_like(t)
    n[:,0] = -t[:,1]
    n[:,1] =  t[:,0]

    return n

def bounds(center,left,right):

    N=len(center)

    amin=np.zeros(N)
    amax=np.zeros(N)

    for i in range(N):

        dl = np.linalg.norm(center[i]-left[i])
        dr = np.linalg.norm(center[i]-right[i])
        w = min(dl,dr)
        amin[i] = -w
        amax[i] =  w

    return amin,amax

def path(center,n,alpha):
    return center + n*alpha[:,None]

def curvature_cost(alpha,center,n):
    p = path(center,n,alpha)
    a = p[1:-1]-p[:-2]
    b = p[2:]-p[1:-1]
    cross = a[:,0]*b[:,1]-a[:,1]*b[:,0]
    denom = np.linalg.norm(a,axis=1)**3 + 1e-6
    k = cross/denom
    return np.sum(k**2)


def cost(alpha,center,n):
    k = curvature_cost(alpha,center,n)
    smooth = np.sum((alpha[1:] - alpha[:-1])**2)

    return k+0.04*smooth


def racing_line(center,left,right,N=200):

    center = resample(center,N)
    left   = resample(left,N)
    right  = resample(right,N)

    n = normals(center)
    amin,amax = bounds(center,left,right)
    bnds = list(zip(amin,amax))
    alpha0 = np.zeros(N)
    res = minimize(cost,alpha0,args=(center,n),bounds=bnds,method="L-BFGS-B",options={"maxiter":300})
    alpha = res.x
    r = path(center,n,alpha)
    return center,r

def plot(left,right,center,r):

    plt.figure(figsize=(7,7))
    plt.plot(left[:,0],left[:,1],'k',label="left boundary")
    plt.plot(right[:,0],right[:,1],'k',label="right boundary")
    plt.plot(center[:,0],center[:,1],'--',label="centerline")
    plt.plot(r[:,0],r[:,1],linewidth=3,label="racing line")
    plt.axis("equal")
    plt.legend()
    plt.title("Racing Line")
    plt.show()

center,r = racing_line(centerline,leftb,rightb,200)
plot(leftb,rightb,center,r)