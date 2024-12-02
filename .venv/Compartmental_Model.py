import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.optimize import minimize
from sympy import DiracDelta
from sympy.abc import x
from numpy import exp
#Set the time and observation data from assignment

t=np.array([0,5,15,30,90,180,360,720,1440,2160,2880,3600,4320])
data=np.array([0,3.33,10.09,18.04,46.19,70.94,92.51,99.67,96.33,96.22,94.25,94.04,93.21])

#Initial conditions
q1_0=0
q2_0=0
y0=np.array([q1_0,q2_0])
t_span=([t[0],t[-1]])


def Model(t,y,params):
    dqdt=np.zeros(2)
    k12,k2e=params
    dqdt[0]=-k12*y[0]+125*Diracdelta(x).subs(x,t)
    dqdt[1]=(k12*y[0]-k2e*y[1])
    return dqdt


def obj(params,t,data):
    a1,a2,l1,l2=params
    return np.sum((data-(a1*np.exp(-l1*t)+a2*np.exp(-l2*t)))**2)

# Initial guess for parameters [damping coefficient, spring constant]
initial_guess = [-60,10,0.1,1e-4]  # Adjust these based on expected values

# Bounds on the parameters (e.g., both should be positive)
bounds = [(None,None),(None,None),(0,None), (0,None)]  # c, k <= 0

# Set options for the optimization
options = {
    'maxiter': 10000,    # Maximum number of iterations
    'disp': True,       # Display convergence messages
}


# Minimize the objective function
result = minimize(obj, initial_guess, args=(t,data),options=options,method="Nelder-Mead")

a1,a2,l1,l2 =result.x

print(result.x)

plt.figure()
plt.scatter(t,data,color="black")
plt.plot(t,a1*np.exp(-l1*t)+a2*np.exp(-l2*t),color="red")
plt.plot(t,a1*exp(-l2*t)+a2*exp(-l1*t))
plt.legend(["Data","Solution","Reversed exponents"])
plt.show()

