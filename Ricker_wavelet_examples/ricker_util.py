#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:52:30 2022

@author: Malcolm Sambridge
"""
import numpy as np
import myGP as gp               # Routines to calculate Gaussian processes used to generate correlated noisy time series.
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import time as timer
import pickle
import json

import FingerprintLib as fp     # My main library for calculating waveform Fingerprint density functions
import OTlib as OT              # My main library for all Optimal Transport calculations 
import ricker_util_opt

def ricker(f, length=0.128, dt=0.001,deriv=False):
    t = np.arange(-length/2, (length-dt)/2, dt) # sigma = sqrt(2)*pi/f
    a = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2))
    b = np.exp(-(np.pi**2)*(f**2)*(t**2))
    y = a * b
    if(deriv):
        dw = b*(-4.0*(np.pi**2)*(f)*(t**2)) + a*(-(np.pi**2)*(2*f)*(t**2)*b)
        return t, y, dw
    return t, y

# Double Ricker wavelet creation code
# Variables:
#      tpert: Centre of double Ricker wavelet in time
#      amp  : Amplitude factor so that the maximum amplitude peak as at amp
#      f    : Dominant frequency factor

def rickerwavelet(tpert,amp,f, trange=[-2.,2.],sigma_amp=0.,sigma_cor=0.,deriv=False,seed=0.,removejitter=True):
    '''
    
        Create a double Ricker wavelet
        
        See eqn. (36) of Sambridge et al. (2022) for details of all parameters.

        Inputs:
            tpert - float; Centre of double Ricker wavelet in time
            amp   - float; Amplitude factor so that the maximum amplitude peak as at amp
            f     - float; Dominant frequency factor
            deriv - logical; switch to calculate and derivatives of the waveform amplitudes and return as an ndarray
            trange - ndarray, shape(nt); time limits of output waveform
            sigma_amp - float; Amplitude of Gaussian noise to be added to waveform
            sigma_cor - float; correlation length (passed to Gaussian process code gp.Createcurve)

        Outputs:
            t - ndarray; shape(nt);
            wpnew - ndarray; shape (nt); amplitude of double ricker wavelet
            dwpd - ndarray; shape(3,nt); derivative of amplitude with respect to time offset, amplitude and frequency parameters. 

            nt the number of samples in the output is determined by 2*length/dt.
    '''

    freq = f*25*4/128 # A wavelength of 25 Hz
    tr, w = ricker(freq,length=4,dt=4/128)
    if(deriv): tr, w, dw = ricker(freq,length=4,dt=4/128,deriv=True)
    wp = amp*np.concatenate((w,w)) # create double ricker
    if(seed != 0.):
        np.random.seed(seed)
    wpn = wp + 0.001*np.max(wp)*np.random.rand(len(wp))
    if(removejitter): wpn = wp 
    tp = np.linspace(trange[0],trange[1],len(wpn)) # define time axis in trange
    #Nt = len(wp)

    if(sigma_cor==0.):
        sn = sigma_amp*np.max(np.abs(wpn))
        noise = sn*np.random.randn(len(wpn))
    else:
        xc,yc = gp.Createcurve(0,nx=len(wpn),corr=sigma_cor)
        noise = yc*sigma_amp/np.std(yc)
        
    wpnew = wpn + noise
    
    if(deriv):
        dwpd = np.zeros((3,len(wp)))
        dwpd[0] = -np.gradient(wpn,tp[1]-tp[0]) # derivative with respect to time offset
        dwpd[1] = np.concatenate((w,w)) # derivative with respect to amplitude
        dwpd[2] = amp*np.concatenate((dw,dw))*25*4/128 # derivative with respect to frequency
        return tp+tpert,wpnew,dwpd
    
    return tp+tpert,wpnew

def datawindowunion(tref,wref,t,w): # prunes one waveform into the time window of the other
    f0 = interpolate.interp1d(t,w,fill_value=(0.0,0.0),bounds_error=False)
    f1 = interpolate.interp1d(tref,wref,fill_value=(0.0,0.0),bounds_error=False)
    
    t0 = np.min((tref[0],t[0]))
    t1 = np.max((tref[-1],t[-1]))
    dt = t[1]-t[0]
    #nt = np.int((t1-t0)/dt)
    nt = int((t1-t0)/dt)
    tnew = np.linspace(t0,t1,nt)
    wout1 = f0(tnew)
    wout2 = f1(tnew)
    return wout1,wout2  

# plot ricker wavelets
def plotrickers(t1,w1,t2,w2,tlim=(False,False),ulim=(False,False),clean=False,
                title='Ricker Wavelets',ref=[False,False],filename='Figures/Obs_pred_ricker.pdf'):
    fig = plt.figure(figsize=(10,4))
    if(not clean): 
        plt.title(title)
        plt.xlabel('t')
    plt.plot(t1,w1,label='Predicted')
    plt.plot(t2,w2,label='Observed')
    #plt.plot(tb,wb,'g-')
    if(not clean): plt.legend()
    if(tlim[0] != False):
        plt.xlim(tlim[0],tlim[1])
    if(ulim[0] != False):
        plt.ylim(ulim[0],ulim[1])
    if(type(ref[0]) is np.ndarray): plt.plot(ref[0],ref[1],color='grey',lw=1.0)
    if(clean):
        plt.tick_params(left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False)

    plt.plot([np.min((t1[0],t2[0])),np.max((t1[-1],t2[-1]))],[0.,0.],'k-',lw=0.5)
    plt.savefig(filename)

    plt.show()
    
# plot ricker wavelets similar to plotrickers with some added features
def plotrickers_special(t1,w1,t2,w2,tlim=(False,False),ulim=(False,False),clean=False,
                title='Ricker Wavelets',ref=[False,False],xlab=False,offset=''):
    '''
    
            Plot pair of ricker wavelets
    
    '''
    if(offset==''):
        plt.plot(t1,w1,lw=0.75)
    else:
        plt.plot(t1,w1,lw=0.75,label=offset)
    plt.plot(t2,w2,lw=0.75)
    plt.ylabel('Amplitude')
    if(xlab): plt.xlabel('Time')
    #if(not clean): plt.legend()
    if(tlim[0] != False):
        plt.xlim(tlim[0],tlim[1])
    if(ulim[0] != False):
        plt.ylim(ulim[0],ulim[1])
    if(type(ref[0]) is np.ndarray): plt.plot(ref[0],ref[1],color='grey',lw=1.0)
    if(clean):
        plt.tick_params(left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False)
    plt.plot([np.min((t1[0],t2[0],tlim[0])),np.max((t1[-1],t2[-1],tlim[-1]))],[0.,0.],
                 'k:',lw=0.5)

# create 3D plot of amplitude vs time shift for W2 surface
def plotsurface(source,x,y,xtrue,ytrue,xlab='x',ylab='y',lw=1.0,base=False,
                 l=0.5,alt=55,lstyle='c--',zorder=99,levels=10,offset=0,lcmap=cm.cubehelix,
                dpi=600,filename='3Dsurface.png'):
    '''
    
            Plot misfit surface
    
    '''
    xv, yv = np.meshgrid(x,y)
    # Set viewer position 
    azimuth = -45.0
    #altitude = 50
    altitude = alt
    col = 'g'
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')
    light = LightSource(315, 25)

    illuminated_surface = light.shade(source, cmap=cm.cubehelix_r, blend_mode='soft')

    ax.view_init(azim = azimuth, elev=altitude)

    ax.plot_surface(xv, yv, source, rstride=1, cstride=1, linewidth=0,
                    antialiased=True, facecolors=illuminated_surface)
                        # do not put color map on both shade and surface - makes too bright & low contrast
    ax.plot([xtrue,xtrue],[ytrue,ytrue],[np.max(source),l*np.max(source)],lstyle,lw=lw,zorder=zorder)
    ax.plot([xtrue],[ytrue],[np.max(source)],'r^',zorder=zorder)
    if(base): ax.plot([xtrue],[ytrue],[offset],'r^',zorder=zorder)
    
    ax.set_xlabel(xlab)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel(ylab)
    ax.set_ylim(y[0], y[-1])
    ax.set_zlabel('Waveform misfit')
    #if(base): ax.contour(xv, yv, source, levels, linewidths=2,  cmap=lcmap, linestyles="solid", offset=offset)
    if(base): ax.contour(xv, yv, source, levels, linewidths=2,  colors='grey', linestyles="solid", offset=offset)

    plt.savefig(filename,dpi=dpi)
    plt.show()
    
# routine to build OT objects from a waveform using a fixed window
# This is used in the ricker wavlet fitting example
def BuildOTobjfromWaveform(t,wave,grid,norm=False,verbose=False,lambdav=None,deriv=False,transform=False):
    '''
        Create two objects:
            - A fingerprint object containing the input time series, window parameters and its nearest distance density field.
            - An Optimal Transport PDF object based on the 2D density field from the first object.

        Inputs:
            t - 1d float array, shape (Nt,): time values of time series.
            wave - 1d float array, shape (Nt,): amplitudes of time series.
            grid - tuple; (t0,t1,u0,u1,Nug,Ntg)
                    du = amplitude limits of waveform
                    u0 = lower limit on amplitude box in un-normalized co-ordinates
                    u1 = upper limit on amplitude box in un-normalized co-ordinates
                    t0 = lower limit on time axis in un-normalized co-ordinates
                    t1 = upper limit on time axis in un-normalized co-ordinates
                    Nug = number of grid points along amplitude axis in time-amplitude window
                    Ntg = number of grid points along time axis in time-amplitude window
            norm -  logical to optionally calculate grid from, t, wave; Default(False).
            verbose - logilcal: write out sone timing information.
            lambdav - float: the distance scaling parameter used in the density field estimation.
            deriv - logical; switch to calculate and derivatives of the 2D density field with respect to the waveform amplitudes and return as a parameter of the class object.
            transform - logical; switch to turn on transform of amplitudes using inverse arctan.
        Outputs:
            wf - Fingerprint class object; 
            OT.OTpdf - Optimal Transport probability density function object for 2D density field.
        
    '''
    if(norm):
        du = np.max(wave)-np.min(wave)     # amplitude limits of waveform
        u0 = np.min(wave)-0.2*du           # lower limit on amplitude box in un-normalized co-ordinates
        u1 = np.max(wave)+0.2*du           # upper limit on amplitude box in un-normalized co-ordinates
        t0 = np.min(t)                     # lower limit on time axis in un-normalized co-ordinates
        t1 = np.max(t)                     # upper limit on time axis in un-normalized co-ordinates
        Nu = int(1.3*len(wave))            # number of grid points in each window
        Nt = len(wave)
    elif(transform): # perform arc tan transform of amplitude
        (t0,t1,u0,u1,Nu,Nt) = grid
        wave = arctan_trans(wave,u0,u1)
        u0,u1 = 0.,1.
    else:
        (t0,t1,u0,u1,Nu,Nt) = grid

    ta = timer.time()
    wf = fp.waveformFP(t,wave,(t0,t1,u0,u1,Nu,Nt)) # create Fingerprint waveform object
    tb = timer.time() - ta
    ta = timer.time()
    if(lambdav is None ): # Calculate PDF of distance field using default spatial scaling
        wf.calcpdf(verbose=verbose,deriv=deriv)
    else:                 # Calculate PDF of distance field using imposed spatial scaling
        wf.calcpdf(verbose=verbose,lambdav=lambdav,deriv=deriv)
        
    tpdf = timer.time() - ta
    xa, xb = np.meshgrid(np.linspace(wf.tlimn[0], wf.tlimn[1], wf.ntg), np.linspace(0., 1., wf.nug))  
    pos = np.dstack((xa, xb))
    if(verbose): print(' BuildOTobjfromWaveform: Time to build fp object: ',tb,' Time to calculate pdf field: ',tpdf)
    if(verbose): print(' Ricker Fingerprint grid ',wf.ntg,wf.nug)    
    if(norm): 
        return wf, OT.OTpdf((wf.pdf,pos)), (t0,t1,u0,u1,Nu,Nt)
    else:
        return wf, OT.OTpdf((wf.pdf,pos))  
    
def arctan_trans(u,u0,u1,deriv=False): #perform arctan transformation of amplitude window
    up = ((u-u0) + (u-u1))/(u1-u0)
    un = 0.5 + np.arctan(up)/np.pi
    und = 2/((u1-u0)*np.pi*(1+up*up))
    if(deriv): return un,und # return transformed 
    return un
    
def CalcWasserWaveform_old(wfsource,wftarget,wf,distfunc='W2',deriv=False,Nproj=10):
    # Calculate derivatives of Wasserstein wrt unormalized amplitudes of waveform
    if(deriv):
        w, dw, dwg = OT.MargWasserstein(wfsource,wftarget,derivatives=True,distfunc=distfunc) # calculate Wasserstein and derivatives
    else:
        [w] = OT.MargWasserstein(wfsource,wftarget,distfunc=distfunc) # calculate Wasserstein and derivatives
    if(deriv): 
        wf.PDFderiv(chainmatrix=dw)
        return w,wf.pdfd,dwg/(wf.tlim[1]-wf.tlim[0])
    else:
        return w

def CalcWasserWaveform(wfsource,wftarget,wf,distfunc='W2',deriv=False,returnmarg=False):
    # Calculate derivatives of Wasserstein wrt unormalized amplitudes of waveform
    '''
    
        Calculates Wasserstein distances between time and amplitude marginals of 2D waveform density functions.
        Optionally calculates derivaties as well with respect to waveform amplitude and origin time of window.
        
        Inputs:
            wfsource - OTpdf class object containing the Fingerprint of the predicted (source) waveform.
            wftarget - OTpdf class object containing the Fingerprint of the observed (target) waveform.
            wf - Fingerprint class object containing the predicted waveform.
            distfunc - string; defined the type of Wasserstein distance (options `W1' or 'W2' for W_p^p, p=1,2)
            deriv - logical; switch to calculate derivatives of W_p^p with respect to the waveform amplitudes.
            returnmarg - logical; switch to return all results for both time and amplitude marginals (otherwise return average results) 
            transform - logical; switch to turn on transform of amplitudes using inverse arctan.

        Outputs:
            w - list of floats; length 1 or 2; Wasserstein between (time,amplitude) marginals 
                for source and target (returnmarg=True) or their average (returnmarg=False)
                
            wf.pdfdMarg - list of 1 or 2 ndarrays; each length (nt,); Derivatives of Wasserstein distances with 
                         respect to waveform amplitudes for each marginal (if deriv=True,returnmarg=True); 
                         else their average (returnmarg=False).
                         
            dwg - list of floats; shape 1 or 2; Derivatives of Wasserstein distances with 
                         respect to origin time of window (if deriv=True,returnmarg=True); 
                         else their average (returnmarg=False).
                         
            nt is the number of points along the predicted time series (wf.nt)
            
    '''

    if(deriv):
        out = OT.MargWasserstein(wfsource,wftarget,derivatives=True,distfunc=distfunc,returnmargW=returnmarg) # calculate Wasserstein and derivatives
        w, dw, dwg = out
    else:
        out = OT.MargWasserstein(wfsource,wftarget,distfunc=distfunc,returnmargW=returnmarg) # calculate Wasserstein and derivatives
        if(returnmarg):
            w = out
        else:
            [w] = out
    if(deriv): 
        if(returnmarg):
            wf.PDFderivMarg(dw)  # get derivative of W marginals wrt to unormalized waveform amplitudes
            return w,wf.pdfdMarg,[dwg[0]/(wf.tlim[1]-wf.tlim[0]),dwg[1]/(wf.tlim[1]-wf.tlim[0])]
        else:
            wf.PDFderiv(chainmatrix=dw)  # get derivative of W wrt to unormalized waveform amplitudes
            return w,wf.pdfd,dwg/(wf.tlim[1]-wf.tlim[0])
    else:
        return w
    
def LSmisfit(tref,wref,tpred,wpred):
    w1,w2 = datawindowunion(tref,wref,tpred,wpred)
    return np.dot(w1-w2,w1-w2)

def writepickle(filename,listOfStr,listOfdata): # Write out a pickle file of objects
    zipbObj = zip(listOfStr, listOfdata)
    # Create a dictionary from zip object
    dictdata = dict(zipbObj)
    pickle.dump( dictdata, open( filename, "wb" ) )
    return
def readpickle(filename):
    return pickle.load( open( filename, "rb" ) )

def writejson(filename,listOfStr,listOfdata): # Write out a pickle file of objects
    for i,data in  enumerate(listOfdata):
        if(isinstance(data, np.ndarray)):
            listOfdata[i] = data.tolist()
    zipbObj = zip(listOfStr, listOfdata)
    # Create a dictionary from zip object
    dictdata = dict(zipbObj)
    json.dump( dictdata, open( filename, "w" ,encoding="utf8") )
    return

def readjson(filename):
    return pickle.load( open( filename, "rb" ) )

#----------------------------------------------------------------------------------------------
#
# routines from iterative inversion
#
#----------------------------------------------------------------------------------------------
# objective function for minimization used by scipy.optimize
def optfunc(x,data):
    '''
        Routine to act as an interface with scipy.minimize(). Actions
            - takes model parameters and builds rickerwavelet (forward problem) plus derivatives
            - takes rickerwavelet and builds fingerprint waveform object and OT object of 2D fingerprint density function.
            - takes OT object and calculates Wasserstein misfits between time and amplitude marginals together with derivatives.
            - Combines derivatives using the chain rule and returns average Wasserstein misfit for time-amplitude marginals plus derivatives. 
    
    '''
    [wfobs_target,distfunc,trange,grid,lambdav,transform,alpha] = data # get data block
    
    tpos,wpos,dw = rickerwavelet(x[0],x[1],x[2],trange=trange,deriv=True)                # model parameters to ricker wavelets

    wfsp,wfsourcep = BuildOTobjfromWaveform(tpos,wpos,grid,lambdav=lambdav,deriv=True,transform=transform)   # ricker wavelets to PDF of fingerprint
 
    w2M,dr,dgM = CalcWasserWaveform(wfsourcep,wfobs_target,wfsp,distfunc=distfunc,deriv=True,returnmarg=True) # PDF of fringerprints to Wasserstein distances  
    
    w2 = alpha*w2M[0] + (1-alpha)*w2M[1] # weighted sum of marginal Wassersteins [see eqn. 21 of Sambridge et al. (2022)]
    
    dg = alpha*dgM[0] + (1-alpha)*dgM[1] # weighted sum of origin time derivatives
    if(transform): # correct amplitude derivatives for transform
        un,dundu = arctan_trans(wpos,grid[2],grid[3],deriv=True)
        #print(len(dr),dr)
        dr[0] = dr[0]*dundu # correct time-marginal Wasserstein misfit for amplitude transform
        dr[1] = dr[1]*dundu # correct amplitude-marginal Wasserstein misfit for amplitude transform

    derivt = dw.dot(dr[0])
    derivu = dw.dot(dr[1])
    deriv = alpha*derivt + (1-alpha)*derivu # weighted sum of marginal derivatives
    deriv[0] = dg # derivative of W_p wrt to window position along time axis.    return w2p
    ricker_util_opt.Wdata.append([w2,x,wfsp,deriv,wfsourcep])
    return w2,deriv

#----------------------------------------------------------------------------------------------
#
# routines from iterative inversion
#
#----------------------------------------------------------------------------------------------
# objective function for minimization used by scipy.optimize
def optfunc_orig(x,data):
    '''
        Routine to act as an interface with scipy.minimize(). Actions
            - takes model parameters and builds rickerwavelet (forward problem) plus derivatives
            - takes rickerwavelet and builds fingerprint waveform object and OT object of 2D fingerprint density function.
            - takes OT object and calculates Wasserstein misfits between time and amplitude marginals together with derivatives.
            - Combines derivatives using the chain rule and returns average Wasserstein misfit for time-amplitude marginals plus derivatives. 
    
    '''
    [wfobs_target,distfunc,trange,grid,lambdav] = data # get data block
    
    tpos,wpos,dw = rickerwavelet(x[0],x[1],x[2],trange=trange,deriv=True)                # model parameters to ricker wavelets

    wfsp,wfsourcep = BuildOTobjfromWaveform(tpos,wpos,grid,lambdav=lambdav,deriv=True,transform=transform)   # ricker wavelets to PDF of fingerprint
 
    w2p,dr,dg = CalcWasserWaveform(wfsourcep,wfobs_target,wfsp,distfunc=distfunc,deriv=True) # PDF of fringerprints to Wasserstein distances  
    # chain rule for dw/dm

    deriv = dw.dot(dr)
    deriv[0] = dg # derivative of W_p wrt to window position along time axis.    return w2p
    ricker_util_opt.Wdata.append([w2p,x,wfsp,deriv,wfsourcep])
    return w2p,deriv

# callback function for recording each iteration of models used by scipy.optimize
def recordresult(x):
    #global Wits
    ricker_util_opt.Wits.append(x)
    print(x)
    return

def findres(Wits,Wdata):
    ind = []
    np.zeros(len(Wits))
    for i in range(len(Wits)):
        for j in range(len(Wdata)):
            if (all(Wits[i] == Wdata[j][1])): 
                #print(i,j)
                ind.append(j)
                pass
    u = np.unique(np.array(ind))
    was = [Wdata[u[x]][0] for x in range(len(u))]
    models = [Wdata[u[x]][1] for x in range(len(u))]
    waves = [Wdata[u[x]][2] for x in range(len(u))]
    return was,models,waves

def plotmisfit(ws,title='Wasserstein distance vs iteration',filename='Figures/wasser_opt.pdf',second=None,log=False,style1='co-',style2='co-'): # plot misfits
    '''
    
            Plot misfit over iterations 
    
    '''
    
    fig = plt.figure(figsize=(6,4))
    plt.title(title)
    plt.xlabel('Iteration')
    if(second is not None): 
        it1 = np.linspace(0,len(second),len(ws))
        it2 = np.linspace(0,len(second),len(second))
    if(log): 
        if(second is not None): 
            plt.semilogy(it1,ws,style1)
            plt.semilogy(it2,second,style2)
        else:
            plt.semilogy(ws,style1)
    else:
        if(second is not None):  
            plt.plot(it1,ws,style1) 
            plt.plot(it2,second,style2)        
        else:
            plt.plot(ws,style1) 
    plt.savefig(filename)
    plt.show()
    
def plotwfit(tobs,wobs,i,wfplot,was,it,w,xlim=[-2.1,7.1],ylim=[-1.0,1.9],title='Waveform fit'):
    fig = plt.figure(figsize=(14,4))
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax1.set_title(title)    
    ax1.set_xlabel('Time')
    ax1.plot(wfplot[i].p.T[0],wfplot[i].p.T[1],label='Synthetic')
    ax1.plot(tobs,wobs,label='Observed')
    ax1.legend()
    ax1.set_xlim(xlim[0],xlim[1])
    ax1.set_ylim(ylim[0],ylim[1])
    ax1.plot(xlim,[0.0,0.0],'k:',lw=0.5)

    ax2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
    ax2.set_title('Misfit reduction')    
    ax2.set_ylabel('Wasserstein distance')
    ax2.semilogy(was,'c-')
    ax2.semilogy(it,w,'ro')
    ax2.set_xlabel('Iteration')
    fig.tight_layout()
#plt.show()

def plotwfit_3panels(tobs,wobs,i,wfplot,was,ls,it,w,l2,xlim=[-2.1,7.1],ylim=[-1.0,1.9],title='Waveform fit'):
    fig = plt.figure(figsize=(14,8))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax1.set_title(title)    
    ax1.set_xlabel('Time')
    ax1.plot(wfplot[i].p.T[0],wfplot[i].p.T[1],label='Synthetic')
    ax1.plot(tobs,wobs,label='Observed')
    ax1.legend()
    ax1.set_xlim(xlim[0],xlim[1])
    ax1.set_ylim(ylim[0],ylim[1])
    ax1.plot(xlim,[0.0,0.0],'k:',lw=0.5)

    ax2 = plt.subplot2grid((2, 3), (0, 2), colspan=1)
    ax2.set_title('Misfit reduction')    
    ax2.set_ylabel('Wasserstein distance')
    ax2.semilogy(was,'w-')
    ax2.semilogy(was[:int(it)+1],'c-')
    ax2.semilogy(it,w,'ro')
    ax2.set_xlabel('Iteration')

    ax3 = plt.subplot2grid((2, 3), (1, 2), colspan=1)
    ax3.set_title('Misfit reduction')    
    ax3.set_ylabel('L2 distance')
    ax3.plot(ls,'w-')
    ax3.plot(ls[:int(it)+1],'c-')
    ax3.plot(it,l2,'ro')
    ax3.set_xlabel('Iteration')

    fig.tight_layout()
#plt.show()

def plotMarginals(wfwave,wf,tag='_',fxsize=None,fysize=None):
    '''
    
            Plot marginals of 2D density function from waveform Fingerpeint and OTpdf classes. 
    
    '''

    wf.setMarginals()  # Calculate marginals
    

    if(fxsize is None):
        figs = (9,1)
    else:
        figs = (fxsize,1)
    # plot time marginal
    fig = plt.figure(figsize=figs)
    plt.plot(wf.marg[1].x,wf.marg[1].pdf)
    plt.fill_between(wf.marg[1].x, 0, wf.marg[1].pdf)
    plt.xlim(wf.marg[1].x[0],wf.marg[1].x[-1])
    plt.tick_params(left=False,bottom=True,labelleft=False,labelbottom=False)
    if(tag != '-'): 
        plt.savefig('Figures/Marginal_u'+tag+'.png',dpi=300)
    else:
        plt.savefig('Figures/Marginal_u.png',dpi=300)
        
    plt.show
        
    fig = plt.figure(figsize=figs)
    plt.plot(wf.marg[0].x,wf.marg[0].pdf)
    plt.fill_between(wf.marg[0].x, 0, wf.marg[0].pdf)
    plt.xlim(wf.marg[0].x[0],wf.marg[0].x[-1])
    plt.tick_params(left=False,bottom=True,labelleft=False,labelbottom=False)
    if(tag != '-'): 
        plt.savefig('Figures/Marginal_t'+tag+'.png',dpi=300)
    else:
        plt.savefig('Figures/Marginal_t.png',dpi=300)
    plt.show

    if(tag != '-'): 
        filename = 'Figures/Ricker_start_fingerprint'+tag+'.pdf'
    else:
        filename = 'Figures/Ricker_start_fingerprint.pdf'

    fp.plot_LS(wfwave.dfield,wfwave,None,None," ",'black','grey',aspect=True,filename=filename,pdf=False,fxsize=fxsize,fysize=fysize)
    return

def check_dwduFD(i,t,RF,dufd,grid,lambdav,wfobs_target,transform=False):
    '''
            Calculate derivatives of Wasserstein distance between time and amplitude marginals 
            with respect to waveform amplitudes by finite difference
    '''
    RFp = np.copy(RF)
    dufdu = dufd*RF[i]/100.
    RFp[i]+=dufdu # pertub waveform +ve
    wfsp,wfsourcep = BuildOTobjfromWaveform(t,RFp,grid,lambdav=lambdav,transform=transform)  # ricker wavelets to PDF of fingerprint
    w2tp,w2up = CalcWasserWaveform(wfsourcep,wfobs_target,wfsp,distfunc='W2',returnmarg=True)[0] # PDF of fringerprints to Wasserstein distances  
    
    RFm = np.copy(RF)
    RFm[i]-=dufdu # pertub waveform +ve
    wfsn,wfsourcen = BuildOTobjfromWaveform(t,RFm,grid,lambdav=lambdav,transform=transform)  # ricker wavelets to PDF of fingerprint
    w2tn,w2un = CalcWasserWaveform(wfsourcen,wfobs_target,wfsn,distfunc='W2',returnmarg=True)[0] # PDF of fringerprints to Wasserstein distances    

    dwtdufd = (w2tp-w2tn)/(2*dufdu)
    dwudufd = (w2up-w2un)/(2*dufdu)
    return dwtdufd,dwudufd

# Finite difference of derivatives of W2 with respect to model parameters
def check_dwdmFD(k,tpred,wpred,dm,mref,grid,lambdav,wfobs_target,trange,transform=False,returnmarg=True):
    '''
            Calculate derivatives of Wasserstein distance between time and amplitude marginals 
            with respect ricker model parameters by finite difference
    '''
    #wfs,wfsource = ru.BuildOTobjfromWaveform(tpred,wpred,grid,lambdav=lambdav,deriv=True)
    #w2,dr,dg = ru.CalcWasserWaveform(wfsource,wfobs_target,wfs,distfunc='W2',deriv=True)
    # chain rule for dw/dm
    #deriv = dw.dot(dr)
    #deriv[0] = dg # derivative of W_p wrt to window position along time axis.

    m = np.copy(mref)
    ds = dm*m[k]
    m[k] += ds
    if(returnmarg):
        tpos,wpos = rickerwavelet(m[0],m[1],m[2],trange=trange)                  # model parameters to ricker wavelets
        wfsp,wfsourcep = BuildOTobjfromWaveform(tpos,wpos,grid,lambdav=lambdav,transform=transform)  # ricker wavelets to PDF of fingerprint
        w2tp,w2up = CalcWasserWaveform(wfsourcep,wfobs_target,wfsp,distfunc='W2',returnmarg=True)[0] # PDF of fringerprints to Wasserstein distances  
        m = np.copy(mref)
        m[k] -= ds
        tneg,wneg = rickerwavelet(m[0],m[1],m[2],trange=trange)                  # model parameters to ricker wavelets
        wfsn,wfsourcen = BuildOTobjfromWaveform(tneg,wneg,grid,lambdav=lambdav,transform=transform)  # ricker wavelets to PDF of fingerprint
        w2tn,w2un = CalcWasserWaveform(wfsourcen,wfobs_target,wfsn,distfunc='W2',returnmarg=True)[0] # PDF of fringerprints to Wasserstein distances    
        fd0t = (w2tp-w2tn)/(2*ds)
        fd0u = (w2up-w2un)/(2*ds)
        return fd0t,fd0u
    else:
        tpos,wpos = rickerwavelet(m[0],m[1],m[2],trange=trange)                  # model parameters to ricker wavelets
        wfsp,wfsourcep = BuildOTobjfromWaveform(tpos,wpos,grid,lambdav=lambdav,transform=transform)  # ricker wavelets to PDF of fingerprint
        w2p = CalcWasserWaveform(wfsourcep,wfobs_target,wfsp,distfunc='W2') # PDF of fringerprints to Wasserstein distances  
        m = np.copy(mref)
        m[k] -= ds
        tneg,wneg = rickerwavelet(m[0],m[1],m[2],trange=trange)                  # model parameters to ricker wavelets
        wfsn,wfsourcen = BuildOTobjfromWaveform(tneg,wneg,grid,lambdav=lambdav,transform=transform)  # ricker wavelets to PDF of fingerprint
        w2n = CalcWasserWaveform(wfsourcen,wfobs_target,wfsn,distfunc='W2') # PDF of fringerprints to Wasserstein distances    
        fd0 = (w2p-w2n)/(2*ds)
        return fd0




    