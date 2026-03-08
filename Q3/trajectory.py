import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

data=pd.read_csv("imu_gps_log.csv")
time=data["timestamp_ns"].values*0.000000001
gyro_z=data["imu_ang_vel_z"].values
acc_x=data["imu_lin_acc_x"].values
acc_y=data["imu_lin_acc_y"].values
lat=data["gps_lat"].values
lon=data["gps_lon"].values

Radius=6378137
lat0=math.radians(lat[0])
lon0=math.radians(lon[0])

gpsx=[]
gpsy=[]
for i in range(len(lat)):
    lat_r=math.radians(lat[i])
    lon_r=math.radians(lon[i])

    x=(lon_r-lon0)*Radius*math.cos(lat0)
    y=(lat_r-lat0)*Radius

    gpsx.append(x)
    gpsy.append(y)

gps_x = np.array(gpsx)
gps_y = np.array(gpsy)

n=5
a=0.03
b=2
k=0
lambda_=a**2*(n+k)-n

Wm=np.zeros(2*n+1)
Wc=np.zeros(2*n+1)
Wm[0]=lambda_/(n+lambda_)
Wc[0]=lambda_/(n+lambda_) +(1-a**2+b)
for i in range(1,2*n+1):

    Wm[i] = 1/(2*(n+lambda_))
    Wc[i] = 1/(2*(n+lambda_))

#Process noise
Q = np.diag([0.1,0.1,0.5,0.5,0.01])
#Measurement noise
R = np.diag([5,5])

#State initial
x= np.array([gps_x[0], gps_y[0], 0, 0, 0])
P=np.eye(n)

def sigma_points(x,P):
    sigma=np.zeros((2*n+1,n))
    sigma[0]=x
    #Cholesky is square root of a matrix.
    #AAT=(n+λ)P
    A=np.linalg.cholesky((n+lambda_)*P)
    for i in range(n):
        sigma[i+1]=x+A[:,i]
        sigma[n+1+i]=x-A[:,i]

    return sigma

def motion(x,ax,ay,gyro,dt):
    px,py,vx,vy,yaw=x

    #yaw update
    yaw=yaw+gyro*dt
    c=math.cos(yaw)
    s=math.sin(yaw)

    Rot_matrix= np.array([
        [c,-s],
        [s,c]
    ])

    #body frame acc
    a_body=np.array([ax,ay])
    a_world=Rot_matrix@a_body
    vx=vx+a_world[0]*dt
    vy=vy+a_world[1]*dt

    px=px+vx*dt+0.5*a_world[0]*dt*dt
    py=py+vy*dt+0.5*a_world[1]*dt*dt

    return np.array([px,py,vx,vy,yaw])

def prediction(x,P,ax,ay,gyro,dt):
    sigma=sigma_points(x,P)
    sigma_prediction=[]
    for s in sigma:
        sigma_prediction.append(motion(s,ax,ay,gyro,dt))
    sigma_prediction=np.array(sigma_prediction)
    x_pred=np.zeros(n)
    for i in range(2*n+1):
        x_pred+=Wm[i]*sigma_prediction[i]

    P_pred=np.zeros((n,n))

    for i in range(2*n+1):
        diff=sigma_prediction[i]-x_pred
        P_pred+=Wc[i]*np.outer(diff,diff)

    P_pred+=Q
    return x_pred,P_pred,sigma_prediction

def measurement(x):
    return np.array([x[0],x[1]])

def update(x_pred,P_pred,sigma_pred,z):
    m = 2
    Z = []
    for s in sigma_pred:
        Z.append(measurement(s))

    Z = np.array(Z)
    z_pred = np.zeros(m)
    for i in range(2*n+1):
        z_pred+=Wm[i]*Z[i]

    S = np.zeros((m,m))
    C = np.zeros((n,m))

    for i in range(2*n+1):

        dz = Z[i]-z_pred
        S+=Wc[i]*np.outer(dz,dz)

    for i in range(2*n+1):
        dx = sigma_pred[i] - x_pred
        dz = Z[i] - z_pred

        C += Wc[i] * np.outer(dx,dz)

    S += R
    K=C@np.linalg.inv(S)
    x_new = x_pred + K@(z-z_pred)
    P_new = P_pred - K@S@K.T

    return x_new,P_new

trajectory=[]
velocity=[]
for i in range(1,len(time)):
    dt=time[i]-time[i-1]
    x_pred,P_pred,sigma=prediction(x,P,acc_x[i],acc_y[i],gyro_z[i],dt)
    z=np.array([gps_x[i],gps_y[i]])

    x,P=update(x_pred,P_pred,sigma,z)
    trajectory.append([x[0],x[1]])

    speed= math.sqrt(x[2]**2 + x[3]**2)
    velocity.append(speed)

trajectory_mat=np.array(trajectory)

plt.figure(figsize=(12,5))


plt.subplot(1,2,1)

plt.plot(trajectory_mat[:,0], trajectory_mat[:,1], label="UKF")
plt.plot(gps_x, gps_y, '--', label="GPS")

plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.title("Trajectory of vehicle")
plt.axis("equal")
plt.grid()
plt.legend()


plt.subplot(1,2,2)

plt.plot(time[1:], velocity)

plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("Vehicle Speed")
plt.grid()


plt.tight_layout()
plt.show()