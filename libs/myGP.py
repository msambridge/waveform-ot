#modified from Andrew Valentines script
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# define Correlation functions
def sqExp(x,xp,s1,rho):
    return (s1**2) * np.exp(-(x-xp)**2/(2.*rho**2))
def matern0(x,xp,s1,rho):
    return (s1**2)*np.exp(-np.abs(x-xp)/rho)
def matern1(x,xp,s1,rho):
    return (s1**2)*(1.+np.sqrt(3)*abs(x-xp)/rho)*np.exp(-np.sqrt(3)*abs(x-xp)/rho)
def matern2(x,xp,s1,rho):
    return (s1**2)*(1.+np.sqrt(5)*abs(x-xp)/rho+5.*(x-xp)**2/(3.*rho**2))*np.exp(-np.sqrt(5)*abs(x-xp)/rho)
def periodic(x,xp,s1,rho,period):
    return (s1**2) *np.exp(-(2*np.sin(abs(x-xp)*np.pi/period)**2)/rho**2)

def Createcurve(plotyn,nx=250,x0=[-3,3],corr=0.2):    
    k = lambda x,xp:matern2(x,xp,0.2,0.1) # pick a Correlation function
    k = lambda x,xp:sqExp(x,xp,0.2,corr) # pick a Correlation function

    # plot the covariance function

    #nx=250
    xx = np.linspace(-1,1,nx)
    if(plotyn):
        plt.figure(figsize=(10,8))
        plt.plot(xx,[k(x,0) for x in xx])
        plt.show()

# calculate the covariance matrix

    K = np.zeros([nx,nx])
    for i in range(nx):
        for j in range(nx):
            K[i,j] = k(xx[i],xx[j])
    if(plotyn):
        plt.figure(figsize=(10,8))
        plt.imshow(K,cmap=plt.cm.cubehelix)
        plt.colorbar()
        plt.contour(K,10,colors='k')
        plt.show()

# generate multiple random deviatives from this PDF

    if(plotyn):
        plt.figure(figsize=(10,8))
        np.random.seed(1726151)
        for i in range(0,10):
            plt.plot(xx,np.random.multivariate_normal(np.zeros(nx),K))
        plt.show()
    
# Generate and plot a single PDF

    xxx = np.linspace(x0[0],x0[1],nx)
    #ye = np.random.multivariate_normal(np.sin(xxx),3*K)
    ye = np.random.multivariate_normal(np.zeros(nx),K)
    if(plotyn):
        plt.figure(figsize=(10,8))
        #plt.plot(xxx,np.sin(xxx))
        plt.title(' A Gaussian Process')
        plt.plot(xxx,ye)
        plt.show()
    return xxx,ye
    
#--------------------------------------------------------
if __name__ == "__main__":

    xcurve,ycurve = Createcurve(1)