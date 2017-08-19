__author__ = "Steven Li"
__version__ = "1.0.1"
__email__ = "lixin78@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2
import statsmodels.api as sm
%matplotlib inline

# Milstein scheme
def simulateCIR_Milstein(x0,dt,Nt,NSample):
    x = np.repeat(x0,NSample)
    Z = np.random.randn(Nt,NSample)*np.sqrt(dt)
    sigma2 = sigma * sigma
    for i in range(Nt):
        dx = (kappa*(theta-x) - 0.25*sigma2)*dt + sigma*np.sqrt(x)*Z[i]+0.25*sigma2*(Z[i]**2)
        x = x+dx
        x = np.maximum(x, 0.0001)
    return x

# Simulation Parameters
kappa = 0.1
theta = 0.5
sigma = 0.2

# Pseudo parameters
M=1000
N=10000
dt=0.01 
x0=0.45
T=M*dt

#CIR Parameters
chi_c = (2*kappa)/(sigma*sigma*(1-np.exp(-kappa*T)))
chi_u = chi_c*x0*np.exp(-kappa*T)
chi_v = chi_c*x0
chi_q = (2*kappa*theta/(sigma*sigma))-1.0
chi_dof = 4*kappa*theta/(sigma*sigma)
chi_ncent = 2*chi_c*x0*np.exp(-kappa*T)

# Simulate and plot Milstein simulated distribution
xm = simulateCIR_Milstein(x0,dt,M,N)
_ = plt.hist(xm,normed=True,label = 'Simulated(Milstein)')
xgrid=np.linspace(0,2.5,100)
plt.plot(xgrid, ncx2.pdf(xgrid*2*chi_c, chi_dof, chi_ncent)*2*chi_c,label='Theoretical')
plt.title('Simulated(Milstein) vs. Theoretical Distrbution')
plt.legend()

# Simulate using WA method (developed by N. Ninomiya & N. Victoir, 2003)
def CIRSimWA(x0,dt,Nt, NSample):
    x = np.repeat(x0,NSample)
    Z = np.random.randn(Nt,NSample)*np.sqrt(dt)
    sigma2 = sigma * sigma
    
    #Xt = np.zeros(N)
    #Xt = np.zeros(N)
    #kappaPrime = kappa - 0.5*(sigma**2)    
    sigma2 = sigma**2
    thetaPrime = theta - sigma2/(4*kappa)
    Xt = x
    for i in range(Nt):
        X1 = np.maximum(0.0,thetaPrime + (Xt - thetaPrime)*np.exp(-0.5*kappa*dt))
        X2 = 0.25*(2*np.sqrt(X1)+sigma*Z[i])**2
        Xt = np.maximum(0.0,thetaPrime + (X2 - thetaPrime)*np.exp(-0.5*kappa*dt))
        #print d,s,sx,dx,Wt[i],Xt
    return Xt

# Simulate and plot WA method
xv = CIRSimWA(x0,dt,M,N)
plt.figure()
plt.hist(xv,normed=True,label='Simulated(WA)')
plt.plot(xgrid, ncx2.pdf(xgrid*2*chi_c, chi_dof, chi_ncent)*2*chi_c,color='r',label='Theoretical')
plt.title('Simulated(WA) vs. Theoretical Distrbution')
plt.legend()

# QQ plots
ecdf_wa = sm.distributions.ECDF(xv*2*chi_c)
ecdf_ms = sm.distributions.ECDF(xm*2*chi_c)
plt.figure()
xgrid=np.linspace(0,50,100)
ecdf_ms_sim = ecdf_ms(xgrid)
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),color='red',label='theoretical')
plt.scatter(ecdf_ms_sim, ncx2.cdf(xgrid, chi_dof, chi_ncent), label='sim(Milstein)')
plt.title('QQ-plot: Simulation (Milstein) vs. Theoretical NC Distribution')
plt.legend(loc=0)


ecdf_wa_sim = ecdf_wa(xgrid)
plt.figure()
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),color='red',label='theoretical')
plt.scatter(ecdf_wa_sim, ncx2.cdf(xgrid, chi_dof, chi_ncent),label='sim(WA)')
plt.title('QQ-plot: Simulation (WA) vs. Theoretical NC Distribution')
plt.legend(loc=0)
