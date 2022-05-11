#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:15:40 2022

@author: Malcolm Sambridge
"""

import pyprop8 as pp
import numpy as np
import matplotlib.pyplot as plt
from pyprop8.utils import rtf2xyz,make_moment_tensor,stf_trapezoidal,clp_filter
from tqdm import tqdm 
from scipy.interpolate import griddata
import time as timer
from matplotlib import cm
import copy
import time
from . import FingerprintLib as fp
from . import OTlib as OT
import pickle as pickle
#global opt_history, opt_history_data
# import ot as POT
from scipy import interpolate
from . import myGP as gp
from . import loc_cmt_util_opt

def prop8seis(x,y,z,prop8data,Mxyz=None,drv=None,show_progress=True,nt=61,timestep=1.0,returndata=False):
    Nm2moment = 1.0E-13 # Conversion factor from Nm to moment argument value
    strike,dip,rake,Mo = prop8data['sdrm']                             # Source characteristics
 
    if(Mxyz is None): # take Mxyz from prop8data disctionary
        Mxyz = rtf2xyz(make_moment_tensor(strike,dip,rake,Mo*Nm2moment,0,0))
    source = pp.PointSource(x,y,z,Mxyz,np.zeros((3,1)),0.0)
    stations = pp.ListOfReceivers(xx = prop8data['recx'].flatten(),yy = prop8data['recy'].flatten(),depth=0.)
    nr = stations.nstations
    nc = 3
    
    #print(' nr = ',nr,' nc =',nc)
        
    if(drv is None):
        t,s = pp.compute_seismograms(prop8data['model'],source,stations,nt,timestep,0.023,
                                     source_time_function=lambda om:clp_filter(om,2*np.pi*0.05,2*np.pi*0.2),
                                     derivatives=drv,show_progress=show_progress)                    
    else:
        t,s,d = pp.compute_seismograms(prop8data['model'],source,stations,nt,timestep,0.023,
                                       source_time_function=lambda om:clp_filter(om,2*np.pi*0.05,2*np.pi*0.2),
                                       derivatives=drv,show_progress=show_progress)
        if(nr==1 and nc>1): d = d[np.newaxis,:,:]
        if(nr==1 and nc==1): d = d[np.newaxis,np.newaxis,:]
            
    if(nr==1 and nc>1): s = s[np.newaxis,:,:]
    if(nr==1 and nc==1): s = s[np.newaxis,np.newaxis,:]

    if(returndata and drv is None): return t,s, source, stations
    if(returndata and drv is not None): return t,s,d, source, stations
    if(drv is None): return t,s
    return t,s,d

def misfitfunc(so,sp): # L2 norm misfir function between waveforms
    r = (so-sp).flatten()
    return np.dot(r.T,r)

def plotseis(splot,tt,splot0=None,splot1=None,splot2=None,title=None,filename='seis.pdf'):
    if(splot.ndim==3):
        nr = np.shape(splot)[0]
        nc = np.shape(splot)[1]
        s = splot
        if(splot0 is not None): s0 = splot0
        if(splot1 is not None): s1 = splot1
        if(splot2 is not None): s2 = splot2
    elif(splot.ndim==2):
        nr = 1
        nc = np.shape(splot)[0]
        s = splot[np.newaxis,:,:]
        if(splot0 is not None): s0 = splot0[np.newaxis,:,:]
        if(splot1 is not None): s1 = splot1[np.newaxis,:,:]
        if(splot2 is not None): s2 = splot2[np.newaxis,:,:]
    else:
        nr = 1
        nc = 1
        s = splot[np.newaxis,np.newaxis,:]
        if(splot0 is not None): s0 = splot0[np.newaxis,np.newaxis,:]
        if(splot1 is not None): s1 = splot1[np.newaxis,np.newaxis,:]
        if(splot2 is not None): s2 = splot2[np.newaxis,np.newaxis,:]

    fig, axs = plt.subplots(nr,3,figsize=(12,1.+1.5*nr),sharex=True)
    
    if(nr==1):
        ax = axs[np.newaxis,:]
    elif(nr==1 and nc==1):
        ax = axs[np.newaxis,np.newaxis,:]
    else:
        ax = axs
    if(title!=None): fig.suptitle(title, fontsize=16,y=1.1)
    lab = ['North','East','Vertical']
    col = ['b','r','g']
    for i in range(nr):      # loop over receivers
        for j in range(nc):  # loop over components
            ax[0,j].set_title(lab[j],fontsize=16)
            ax[i,j].tick_params(axis='both', which='major', labelsize=15)
            if(splot0 is not None): ax[i,j].plot(tt,s0[i,j,:],linestyle=':',color=col[j],lw=1.0)
            if(splot1 is not None): ax[i,j].plot(tt,s1[i,j,:],linestyle='--',color='k',lw=1.0)
            if(splot2 is not None): ax[i,j].plot(tt,s2[i,j,:],linestyle='-',color='k',alpha=0.5,lw=1.0)
            ax[i,j].plot(tt,s[i,j,:],linestyle='-',color=col[j])
    plt.tight_layout()
    if(nr==1):
        fig.subplots_adjust(top=0.9)
    plt.savefig(filename)
    plt.show()
    
# objective function for minimization used by scipy.optimize
def optfunc(m,optdata,returnseis=False,return2W=False,precon=True): # wrapper routine to call either L2 norm misfit version optfunc_L2, or OT based misfit version `optfunc_OT'
    invopt        = optdata['invopt']
    if(invopt['mistype'] == 'OT'):
        return optfunc_OT(m,optdata,returnseis=returnseis,return2W=return2W,precon=precon)
    elif(invopt['mistype'] == 'L2'):
        return optfunc_L2(m,optdata,returnseis=returnseis,precon=precon)

def optfunc_L2(m_in,optdata,returnseis=False,returnseisd=False,noderiv=False,geometry='cartesian',precon=True): # L2 norm misfit function
    #global opt_history_data
    
    #OTdata        = optdata['OTdata']
    invopt        = optdata['invopt']
    prop8data     = optdata['prop8data']      # read the pyprop8 data block for passing to prop8 seismogram calculations
    seis_obs      = prop8data['obs_seis']     # observed seismograms
    fac = 1.
    if(invopt['precon'] and precon): # use preconditioning if set globally and not turned off locally
        m = m_in*invopt['mscal']
    else:
        m = m_in
    
    if(not invopt['loc'] and not invopt['cmt']): return 0.,np.zeros_like(m)
    
    if(invopt['loc']):            
        x,y,z = m[:3]   # source location
    else:
        x,y,z = invopt['mref']   # source location is fixed and passed via data block
    
    if(z < 0.001): z = 0.001 # stop source going above receivers
    
   # if cartesian geometry then derivatives returned by pyprop8 will be wrt source (x,y), otherwise (r,phi,z)
    
    Mxyz = None # prop8 gets fixed Mxyz from prop8data dictionary (for source location at fixed cmt)
    cmtd = False # switch for calculating cmt derivatives
    if(invopt['loc'] and invopt['cmt']): # We have moment tensor inversion parameters active
        Mxyz = fac*buildMxyzfromupper(m[3:]) # Mxyz comes from input model parameters during optimisation
        cmtd = True
        #print(' my Mxyz\n',Mxyz)
    elif(invopt['cmt']): # We have only moment tensor inversion parameters active
        Mxyz = fac*buildMxyzfromupper(m)
        cmtd = True

    if(geometry=='cartesian'): # return source location derivatives in cartesian co-ordinates from pyprop8
        drv = pp.DerivativeSwitches(x=invopt['loc'],y=invopt['loc'],depth=invopt['loc'],moment_tensor=cmtd,structure=prop8data['model']) # choose which derivatives to make
    else:                      # return source location derivatives in spherical co-ordinates from pyprop8
        drv = pp.DerivativeSwitches(r=invopt['loc'],phi=invopt['loc'],depth=invopt['loc'],moment_tensor=cmtd,structure=prop8data['model']) # choose which derivatives to make

    #print(m)
    t, seis_pred, deriv, sourceL, stations = prop8seis(x,y,z,prop8data,Mxyz=Mxyz,drv=drv,show_progress=False,returndata=True) # calculate displacement seismograms

    # derivatives
    
    derivxyz = drv_rpd2xyz(drv,deriv,stations,geometry=geometry) # use chain rule to find derivatives wrt of source location (3,nr,nc,nt)

    dr = (seis_pred-seis_obs).flatten() # seismogram residuals
    mis = np.dot(dr.T,dr)            # L2 misfit function

    # derivatives
    
    if(invopt['loc'] and invopt['cmt']):        
        d = derivxyz.reshape(9,-1)
    else:
        d = derivxyz.reshape(3,-1)
    dmis = 2*d.dot(dr)    
    
    loc_cmt_util_opt.opt_history_data.append([mis,m,dmis,seis_pred,Mxyz])
    
    if(returnseis): return mis,dmis,t, seis_pred # return misfit, derivative of misfit wrt model parameters, and seismograms
    if(returnseisd): return mis,dmis,t, seis_pred,derivxyz # return misfit, derivative of misfit wrt model parameters, and seismograms
    if(noderiv): return mis
    if(invopt['precon'] and precon): dmis*=invopt['mscal'] 
    return mis,dmis                             # return misfit and derivative of misfit wrt model parameters

# OT objective function for minimization
def optfunc_OT(m_in,optdata,returnseis=False,returnwobj=False,returngrid=False,noderiv=False,returnderiv=False,return2W=False,geometry='cartesian',precon=True): # Wasserstein misfit function
    #global opt_history_data
    
    OTdata        = optdata['OTdata']
    invopt        = optdata['invopt']
    prop8data     = optdata['prop8data']      # read the pyprop8 data block for passing to prop8 seismogram calculations
    seis_obs      = prop8data['obs_seis']     # observed seismograms
    fac = 1.
    if(invopt['precon'] and precon): # use preconditioning if set globally and not turned off locally
        m = m_in*invopt['mscal']
    else:
        m = m_in
    
    if(not invopt['loc'] and not invopt['cmt']): return 0.,np.zeros_like(m)
    
    if(invopt['loc']):            
        x,y,z = m[:3]   # source location
    else:
        x,y,z = invopt['mref']   # source location is fixed and passed via data block
    
    if(z < 0.001): z = 0.001 # stop source going above receivers
    
   # if cartesian geometry then derivatives returned by pyprop8 will be wrt source (x,y), otherwise (r,phi,z)
    
    Mxyz = None # prop8 gets fixed Mxyz from prop8data dictionary (for source location at fixed cmt)
    cmtd = False # switch for calculating cmt derivatives
    if(invopt['loc'] and invopt['cmt']): # We have moment tensor inversion parameters active
        Mxyz = fac*buildMxyzfromupper(m[3:]) # Mxyz comes from input model parameters during optimisation
        cmtd = True
        #print(' my Mxyz\n',Mxyz)
    elif(invopt['cmt']): # We have only moment tensor inversion parameters active
        Mxyz = fac*buildMxyzfromupper(m)
        cmtd = True

    if(geometry=='cartesian'): # return source location derivatives in cartesian co-ordinates from pyprop8
        drv = pp.DerivativeSwitches(x=invopt['loc'],y=invopt['loc'],depth=invopt['loc'],moment_tensor=cmtd,structure=prop8data['model']) # choose which derivatives to make
    else:                      # return source location derivatives in spherical co-ordinates from pyprop8
        drv = pp.DerivativeSwitches(r=invopt['loc'],phi=invopt['loc'],depth=invopt['loc'],moment_tensor=cmtd,structure=prop8data['model']) # choose which derivatives to make

    #print(m)
    t, seis_pred, deriv, sourceL, stations = prop8seis(x,y,z,prop8data,Mxyz=Mxyz,drv=drv,show_progress=False,returndata=True) # calculate displacement seismograms

    # derivatives
    
    derivxyz = drv_rpd2xyz(drv,deriv,stations,geometry=geometry) # use chain rule to find derivatives wrt of source location (3,nr,nc,nt)

    # OT starts here
    
    # set up OT grid and objects for observations and reference source
    nr,nc,nt = np.shape(seis_obs) # get number of receivers, components and length of seismograms
    dr       = np.zeros((nr,nc,nt))
    returnmarg = OTdata['Wopt'] != 'Wavg' or return2W # condition for returning both Marginal W-distances

    mis = 0.
    if(returnmarg): # return two sets of Wasserstein and derivatives one for each marginal (only applies if mthd='Marginal').
        misW = np.zeros(2)
        dr  = np.zeros((2,nr,nc,nt))
        dg  = np.zeros(2)

    # use (0,1) range for predicted and observed seismogram windows after linear-arctan transform
    
    obs_grids = OTdata['obs_grids']
    wfobs_target = OTdata['wfobs_target']
    wfo = OTdata['wfobs']
    fpgrid=None
    un,dundu = arctan_trans(seis_pred,obs_grids,deriv=True)
            
    wfp, wfpred_source = BuildOTobjfromWaveform(t,seis_pred,obs_grids,OTdata,lambdav=OTdata['plambda'],deriv=True,fpgrid=fpgrid) # Calculate fingerprints for each predicted waveform

    if(returnmarg):
        for i in range(nr): # loop over receivers
            for j in range(nc): # loop over components
                w2pl,drl,dgl = CalcWasserWaveform(wfpred_source[i][j],wfobs_target[i][j], wfp[i][j], distfunc=OTdata['distfunc'],deriv=True,returnmarg=returnmarg) # PDF of fingerprints to Wasserstein distances  
                #print(' mis',mis,' w2pl',w2pl)
                misW[0]+= w2pl[0]      
                misW[1]+= w2pl[1]    
                dr[0,i,j,:] = drl[0]
                dr[1,i,j,:] = drl[1]
                dg[0] = dgl[0]
                dg[1] = dgl[1]
    else:
        for i in range(nr): # loop over receivers
            for j in range(nc): # loop over components
                w2p,dr[i,j,:],dg = CalcWasserWaveform(wfpred_source[i][j],wfobs_target[i][j], wfp[i][j],distfunc=OTdata['distfunc'],deriv=True,returnmarg=returnmarg) # PDF of fingerprints to Wasserstein distances  

                mis+= w2p
    
    if(invopt['loc'] and invopt['cmt']):  # derivative of W wrt to seismograms
        d = derivxyz.reshape(9,-1)
    else:
        #d = np.array([derivxyz[0].flatten(),derivxyz[1].flatten(),derivxyz[2].flatten()]) 
        d = derivxyz.reshape(3,-1)
            
    if(returnmarg):
        dr[0] = dr[0]*dundu
        dr[1] = dr[1]*dundu

        dmis0 = d.dot(dr[0].flatten())
        dmis1 = d.dot(dr[1].flatten())
        if(return2W):     #return both Marginal W_p distances and (optionally) derivatives
            dmis = [dmis0,dmis1]
            mis = misW
        elif(OTdata['Wopt'] == 'Wt'): #return t-Marginal W_p distance and (optionally) derivatives
            dmis = dmis0
            mis = misW[0]
        elif(OTdata['Wopt'] == 'Wu'): #return u-Marginal W_p distance and (optionally) derivatives
            dmis = dmis1
            mis = misW[1]
    else:
        dr = dr*dundu
        dmis = d.dot(dr.flatten()) # derivative of Wasserstein distance misfit wrt source x,y,z (using chain rule)
                
    loc_cmt_util_opt.opt_history_data.append([mis,m,dmis,seis_pred,Mxyz])
    
    if(returnseis): return mis,dmis,t,seis_pred # return misfit, derivative of misfit wrt model parameters, and seismograms
    if(returnwobj): return mis,dmis,wfo,wfp,wfpred_source,wfobs_target # return misfit, derivative of misfit wrt model parameters, fingerprint objects, OT objects
    if(returngrid): return mis,dmis,obs_grids # return misfit, derivative of misfit wrt model parameters, and seismograms
    if(noderiv): return mis
    if(returnderiv): return mis,dmis,derivxyz,dr
    if(invopt['precon'] and precon): dmis*=invopt['mscal'] 
    return mis,dmis                              # return misfit and derivative of misfit wrt model parameters

# Least squares solution to moment tensor for given location
def Moment_LS(xyz,prop8data):
    
    diagorder = [0,3,4,1,5,2]   # reorder derivatives from `diag first' order to upper triangular order
        
    seis_obs = prop8data['obs_seis'] # observed seismograms
    x,y,z = xyz   # source location
    
    if(z < 0.001): z = 0.001 # stop source going above receivers
    
    drv = pp.DerivativeSwitches(moment_tensor=True,structure=prop8data['model']) # choose which derivatives to make

    #t, seis_pred, deriv, sourceL, stations = prop8seis(x,y,z,prop8data,Mxyz=np.eye(3),drv=drv,show_progress=False,returndata=True) # calculate displacement seismograms
    t, seis_pred, deriv = prop8seis(x,y,z,prop8data,Mxyz=np.eye(3),drv=drv,show_progress=False) # calculate displacement seismograms
 
    # solve for moment tensor
            
    Gt = np.array([deriv[:,drv.i_mt+diagorder[0],:,:].flatten(),deriv[:,drv.i_mt+diagorder[1],:,:].flatten(),deriv[:,drv.i_mt+diagorder[2],:,:].flatten(),
                          deriv[:,drv.i_mt+diagorder[3],:,:].flatten(),deriv[:,drv.i_mt+diagorder[4],:,:].flatten(),deriv[:,drv.i_mt+diagorder[5],:,:].flatten()])
    #Gt = Gt[diagorder]
        
    Gtd = Gt.dot(seis_obs.flatten())
    #print('Gtd',np.shape(Gtd))
    GtG = Gt.dot(Gt.T)
    #print('GtG',np.shape(GtG))
     
    return np.linalg.inv(GtG).dot(Gtd)

# callback function for recording each iteration of models used by scipy.optimize

def recordresult(x):
    #global opt_history
    invopt        = loc_cmt_util_opt.optdata['invopt']

    mis = loc_cmt_util_opt.opt_history_data[-1][0]
    Mxyz = loc_cmt_util_opt.opt_history_data[-1][-1]
    index = len(loc_cmt_util_opt.opt_history_data)
    if(invopt['precon']):
        loc_cmt_util_opt.opt_history.append([x*invopt['mscal'],mis,index,Mxyz])
    else:
        loc_cmt_util_opt.opt_history.append([x,mis,index,Mxyz])
    #print(x)
    return

def buildMxyzfromupper(vals):
    new = np.zeros((3,3))
    inds = np.triu_indices_from(new)
    new[inds] = vals
    new[(inds[1], inds[0])] = vals
    return new
#print(buildMxyzfromupper(m))

def drv_rpd2xyz(drv,deriv,stations,geometry='spherical'):
    # source location derivatives provided by pyprop
    diagorder = [0,3,4,1,5,2]   # reorder derivatives from `diag first' order from pyprop8 to upper triangular order
    if(geometry=='spherical'):
        dseisdr   = deriv[:,drv.i_r,:,:]   # derivative of seismograms wrt range of receiver from source, r
        dseisdp   = deriv[:,drv.i_phi,:,:] # derivative of seismograms wrt azimuth of receiver from source along x-axis ccw, phi.
        dseisdd   = deriv[:,drv.i_dep,:,:] # derivative of seismograms wrt depth of receiver, z_r
        # use chain rule to find derivatives of source location in Cartesian co-ordinates
        dseisdx = ((dseisdr.T)*(-np.cos(stations.pp)) + (dseisdp.T)*(np.sin(stations.pp)/stations.rr)).T # derivative of seismograms wrt x co-ordinate of source
        dseisdy = -((dseisdr.T)*(np.sin(stations.pp)) + (dseisdp.T)*(np.cos(stations.pp)/stations.rr)).T # derivative of seismograms wrt y co-ordinate of source
        dseisdz = -dseisdd   # derivative of seismograms wrt y co-ordinate of source
    else:
        dseisdx   = deriv[:,drv.i_x,:,:]   # derivative of seismograms wrt x-co-ordinate of source
        dseisdy   = deriv[:,drv.i_y,:,:]   # derivative of seismograms wrt y-co-ordinate of source
        dseisdz   = -deriv[:,drv.i_dep,:,:] # derivative of seismograms wrt depth of receiver, z_r
    if(drv.moment_tensor):
        dseismt0   = deriv[:,drv.i_mt+diagorder[0],:,:]   # derivative of seismograms wrt moment tensor components       
        dseismt1   = deriv[:,drv.i_mt+diagorder[1],:,:] # derivative of seismograms wrt moment tensor components       
        dseismt2   = deriv[:,drv.i_mt+diagorder[2],:,:] # derivative of seismograms wrt moment tensor components       
        dseismt3   = deriv[:,drv.i_mt+diagorder[3],:,:] # derivative of seismograms wrt moment tensor components       
        dseismt4   = deriv[:,drv.i_mt+diagorder[4],:,:] # derivative of seismograms wrt moment tensor components       
        dseismt5   = deriv[:,drv.i_mt+diagorder[5],:,:] # derivative of seismograms wrt moment tensor components       
        return np.array([dseisdx,dseisdy,dseisdz,dseismt0,dseismt1,dseismt2,dseismt3,dseismt4,dseismt5])
    return np.array([dseisdx,dseisdy,dseisdz])

def BuildMxyz(A):
    X = np.zeros((3,3))
    X[np.triu_indices(3)] = A
    return X + X.T - np.diag(np.diag(X))

# utility routine to setup inversion parameters
def setmref(invopt,source,mtrue):
    cmtfactor = 1.
    if(invopt['loc']): mref = mtrue
    if(invopt['loc'] and invopt['cmt']): mref = [mtrue[0],mtrue[1],mtrue[2],source.Mxyz[0][np.triu_indices(3)]/cmtfactor]
    if(not invopt['loc'] and invopt['cmt']): mref = source.Mxyz[0][np.triu_indices(3)]/cmtfactor #  True solution of location and cmt
    return mref

# analaysis
def checkconverge(solutions,dlimit=1.,mlimit=None,verbose=False,):
    misfitcondition = False
    ns = 0.
    mistrue = solutions[0][5]
    mistrue = -10.
    mlimit = 10. # % upper bound in relative misfit before convergence
    #dlimit =  1. # distance to truth source for convergence
    mtrue = solutions[0][4][:3]
    #if(po['invopt']['cmt'] == True):  mtrue = solutions[0][4][:3]
    con = [False for i in range(len(solutions))]
    d = np.zeros(len(solutions))
    gcon = [False for i in range(len(solutions))]

    for i in range(len(solutions)):
        mstart = solutions[i][0]
        d[i] = np.linalg.norm(solutions[i][2][:3]-mtrue)
        if(np.abs(mstart[0]) !=80.): # restrict analysis to smaller square
            gcon[i] = True
            ns +=1.
            s = 100*np.abs(solutions[i][3]-mistrue)/mistrue # percentage difference in misfit
            #print(s)
            if(misfitcondition and (s < mlimit)): 
                con[i] = True
            elif(d[i]<dlimit or (solutions[i][3] < mistrue)):
                con[i] = True   
            if(verbose):print(i,':',' start: ',solutions[i][0][:3],' mis start ',solutions[i][1],' mis final:',solutions[i][3],' mfinal',solutions[i][2][:3],con[i])
            #print(d)
    if(verbose): print('\n',np.sum(con),' of ',ns,' converged: ',100*np.sum(con)/ns,'%')
    return con,d,gcon

# routine to perfom all set up and be used in the loops below
def buildFingerprintwindows(t,wave,Nu=None,Nt=None,u0=None,u1=None):
    nr,nc,nt = np.shape(wave)
    grid = np.zeros((nr,nc)).tolist()
    for i in range(nr):      # loop over receivers
        for j in range(nc):  # loop over components
            du = np.max(wave[i,j])-np.min(wave[i,j])     # amplitude limits of waveform
            u0out,u1out = u0,u1
            if(u0 is None): u0out = np.min(wave[i,j])-0.3*du                # lower limit on amplitude box in un-normalized co-ordinates
            if(u1 is None): u1out = np.max(wave[i,j])+0.3*du                # upper limit on amplitude box in un-normalized co-ordinates
            t0 = np.min(t)                               # lower limit on time axis in un-normalized co-ordinates
            t1 = np.max(t)                               # upper limit on time axis in un-normalized co-ordinates
            Nu_used = Nu
            if(Nu_used is None): Nu_used = int(1.3*len(wave[i,j]))        # number of grid points along amplitude axis
            Nt_used = Nt
            if(Nt_used is None): Nt_used = len(wave[i,j])                  # number of grid points along time axis
            grid[i][j] = [t0,t1,u0out,u1out,Nu_used,Nt_used]
    return grid    

def BuildOTobjfromWaveform(t,wavein,gridin,OTdata,norm=False,verbose=False,lambdav=None,deriv=False,fpgrid=None):
    '''
        Create two objects:
            - A fingerprint object containing the input time series, window parameters and its nearest distance density field.
            - An Optimal Transport PDF object based on the 2D density field from the first object.

        Inputs:
            t - 1d float array, shape (Nt,): time values of time series.
            wave - 1d float array, shape (Nt,): amplitudes of time series.
            gridin - tuple; (t0,t1,u0,u1,Nug,Ntg)
                    du = amplitude limits of waveform
                    u0 = lower limit on amplitude box in un-normalized co-ordinates
                    u1 = upper limit on amplitude box in un-normalized co-ordinates
                    t0 = lower limit on time axis in un-normalized co-ordinates
                    t1 = upper limit on time axis in un-normalized co-ordinates
                    Nug = number of grid points along amplitude axis in time-amplitude window
                    Ntg = number of grid points along time axis in time-amplitude window
            OTdata - list, list of optimisation data (used here to supply all transformed grids, OTdata['obs_grids01')
            norm -  logical to optionally calculate grid from, t, wave; Default(False).
            verbose - logilcal: write out sone timing information.
            lambdav - float: the distance scaling parameter used in the density field estimation.
            deriv - logical; switch to calculate and derivatives of the 2D density field with respect to the waveform amplitudes and return as a parameter of the class object.

        Outputs:
            wf - Fingerprint class object; 
            OT.OTpdf - Optimal Transport probability density function object for 2D density field.
        
    '''

    fpgrid=None
    
    if(wavein.ndim==1): # input is a single waveform (Not quite working)
        nr,nc = 1,1
        wave = np.array([[wavein]])
        grid = np.array([[gridin]]).tolist()
        if(fpgrid is not None): fpgridused = np.array([[fpgrid]]).tolist()
    elif(wavein.ndim==3): # input is an array of waveforms, (grid is a list)
        nr,nc,nt = np.shape(wavein)
        wave = wavein
        grid = gridin
        if(fpgrid is not None): fpgridused = fpgrid
    else:
        print(' Do not understand input waveform dimension is not 1 or 3 ?')
        raise
    twf,tpdf = 0.,0.
    wflist   = np.zeros((nr,nc)).tolist()
    wfolist  = np.zeros((nr,nc)).tolist()
    fpgridu = fpgrid
    
    u = arctan_trans(wave,grid)
    grid = OTdata['obs_grids01'] # Here amplitudes ranges are (0,1) because u has been transformed.

    q=2    # PDF = exp(-d**q)/lambda
    q=None # PDF = exp(-d**q)/lambda
                
    for i in range(nr):      # loop over receivers
        for j in range(nc):  # loop over components             (t0,t1,u0,u1,Nu,Nt) = grid[i][j]  
            #print(u0,Nu)
            if(fpgrid is not None): fpgridu = fpgridused[i][j]
            ta = timer.time()
            wf = fp.waveformFP(t,u[i][j],grid[i][j],fpgrid=fpgridu) # create Fingerprint waveform object
            twf += timer.time() - ta
            ta = timer.time()
            if(lambdav is None ): # Calculate PDF of distance field using default spatial scaling
                wf.calcpdf(deriv=deriv,q=q) # default value of lamba is 0.04
            else:                 # Calculate PDF of distance field using imposed spatial scaling
                wf.calcpdf(lambdav=lambdav,deriv=deriv,q=q)
            wflist[i][j] = wf # create list of Fingerprint waveform objects
            tpdf += timer.time() - ta
            #xa, xb = np.meshgrid(np.linspace(wf.tlimn[0], wf.tlimn[1], wf.ntg), np.linspace(0., 1., wf.nug))  
            #pos = np.dstack((xa, xb))
            wfolist[i][j] = OT.OTpdf((wf.pdf,wf.pos))  
            # here the OT library normalises the PDF in 2D and set Marginals returns 1D PDFs normalised again.
            
    if(verbose): print(' BuildOTobjfromWaveform: Time to build fp objects: ',twf,' Time to calculate pdf fields: ',tpdf)

    return wflist, wfolist

# Calculates Wasserstein misfit for all waveforms 
def CalcWasserWaveform(wfsource,wftarget,wf,distfunc='W2',deriv=False,Nproj=10,returnmarg=False):
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
    # Calculate derivatives of Wasserstein wrt unormalized amplitudes of waveform
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

def arctan_trans(u,grids,deriv=False): #perform arctan transformation of one or more window
    nr,nc,nt = np.shape(u)
    un = np.zeros_like(u)
    und = np.zeros_like(u)
    for i in range(nr):
        for j in range(nc):
            u0,u1 = grids[i][j][2:4]
            up = ((u[i][j]-u0) + (u[i][j]-u1))/(u1-u0)
            un[i,j] = 0.5 + np.arctan(up)/np.pi
            und[i,j] = 2/((u1-u0)*np.pi*(1+up*up))
    if(deriv): return un,und
    return un

def plotmisfitsection(xlim,ylim,xgrid,ygrid,zg,ztrue,sol,misfitgrid,invopt,plotfile,returncontfunc=False):
    i=0
    # plot interpolated misfit function
    ninterp = 100 # fine grid to interpolate misfit function for plot
    X,Y = np.meshgrid(np.linspace(xlim[0],xlim[1],ninterp),np.linspace(ylim[0],ylim[1],ninterp))
    if(returncontfunc): Tplotout = np.zeros((4,ninterp,ninterp))
    for misplot in misfitgrid:
        xlims = xlim
        ylims = ylim
        plotfill = True
   
        plt.figure(figsize=(16,12))
        if(invopt['mistype']=='L2'): plt.suptitle('Contours of L2-norm misfit function between seismograms as a function of source position', fontsize=16)
        if(invopt['mistype']=='OT'): plt.suptitle('Contours of Wasserstein misfit function between seismograms as a function of source position', fontsize=16)
        plt.subplot(2,2,1)
        Ti = griddata((xgrid.flatten(),ygrid.flatten()), misplot[0].flatten(), (X, Y), method='cubic')
        Tplot = np.log(np.clip(Ti,1.0,np.inf)) # modify contours to ease plotting
        if(invopt['mistype']=='OT'): Tplot = Ti
        plt.contour(X,Y,Tplot,30,cmap=cm.cubehelix_r)
        if(plotfill): plt.contourf(X,Y,Tplot,30,cmap=cm.cubehelix_r)
        if(returncontfunc): Tplotout[0]=Tplot
        plt.title('Misfit at z='+str(zg[0])+' km with source at '+str(ztrue)+'km depth')
        plt.plot(0.,0.,'ko',markersize=4.0)
        
        plt.colorbar()

        plt.subplot(2,2,2)
        Ti = griddata((xgrid.flatten(),ygrid.flatten()), misplot[1].flatten(), (X, Y), method='cubic')
        Tplot = np.log(np.clip(Ti,1.0,np.inf)) # modify contours to ease plotting
        if(invopt['mistype']=='OT'): Tplot = Ti
        plt.contour(X,Y,Tplot,30,cmap=cm.cubehelix_r)
        if(plotfill): plt.contourf(X,Y,Tplot,30,cmap=cm.cubehelix_r)
        if(returncontfunc): Tplotout[1]=Tplot
        plt.title('Misfit at z='+str(zg[1])+' km with source at '+str(ztrue)+'km depth')
        plt.plot(0.,0.,'ko',markersize=4.0)
        plt.colorbar()

        plt.subplot(2,2,3)
        Ti = griddata((xgrid.flatten(),ygrid.flatten()), misplot[2].flatten(), (X, Y), method='cubic')
        Tplot = np.log(np.clip(Ti,1.0,np.inf)) # modify contours to ease plotting
        if(invopt['mistype']=='OT'): Tplot = Ti
        plt.contour(X,Y,Tplot,30,cmap=cm.cubehelix_r)
        if(plotfill): plt.contourf(X,Y,Tplot,30,cmap=cm.cubehelix_r)
        if(returncontfunc): Tplotout[2]=Tplot
        plt.title('Misfit at z='+str(zg[2])+' km with source at '+str(ztrue)+'km depth')
        plt.plot(0.,0.,'ko',markersize=4.0)
        plt.colorbar()

        Xs,Ys = np.meshgrid(np.linspace(xlims[0],xlims[1],ninterp),np.linspace(ylims[0],ylims[1],ninterp))
        plt.subplot(2,2,4)
        Ti = griddata((xgrid.flatten(),ygrid.flatten()), misplot[3].flatten(), (Xs, Ys), method='cubic')
        Tplot = np.log(np.clip(Ti,1.0,np.inf)) # modify contours to ease plotting
        if(invopt['mistype']=='OT'): Tplot = Ti
        plt.contour(Xs,Ys,Tplot,30,cmap=cm.cubehelix_r)
        if(plotfill): plt.contourf(Xs,Ys,Tplot,30,cmap=cm.cubehelix_r)
        if(returncontfunc): Tplotout[3]=Tplot
        #plt.title('Misfit at z='+str(z)+' km with source at {:10.4f} km depth'.format(x))
        plt.title('Misfit at z={:4.1f} km with source at {:4.1f} km depth'.format(zg[3],ztrue))
        plt.plot(0.,0.,'ko',markersize=4.0)
        plt.plot(sol[0],sol[1],'co',markersize=4.0)
        plt.colorbar()

        plt.savefig(plotfile[i])
        i+=1
                                            
    if(returncontfunc): return Tplotout
    return

def writepickle(filename,listOfStr,listOfdata):
    zipbObj = zip(listOfStr, listOfdata)
    # Create a dictionary from zip object
    dictdata = dict(zipbObj)
    pickle.dump( dictdata, open( filename, "wb" ) )
    return

def readpickle(filename):
    return pickle.load( open( filename, "rb" ) )

def printanalysis(sol,opt,mtrue,mstart,mis_start,mis_true,prop8data,sdata_nonoise,fit=False):
    print('\n Some analysis of solution: ')
    if(not opt.success):
        print('\n Optimisation Failed')
        return
    mis_final = opt.fun
    mfinal = sol[:3]
    Mxyz_true =buildMxyzfromupper(mtrue[3:])
    Mxyz_final = loc_cmt_util_opt.opt_history[-1][3]
    print('\n Model loc start :',mstart[:3],'\n Misfit start :',mis_start,'\n Misfit final :',mis_final,'\n Misfit true  :',mis_true)
    print('\n Final location = ',mfinal,'\n True location = ',mtrue[:3])
    print('\n Start CMT = \n',buildMxyzfromupper(mstart[3:]))
    print('\n Final CMT = \n',Mxyz_final)
    print('\n True CMT = \n',Mxyz_true)
    Mxyz_err = 100*(Mxyz_final - Mxyz_true)/Mxyz_true
    print('\n % Error in CMT:\n',Mxyz_err)
    #
    # Now fit Mxyz at True location to see error in Mxyz
    #
    if(fit):
        x,y,z = mtrue[:3]
        Mxyzfrom_mtrue = BuildMxyz(Moment_LS([x,y,z],prop8data))
        print('\n Best fit CMT using True location = \n',Mxyzfrom_mtrue)
        Mxyz_err2 = 100*(Mxyzfrom_mtrue - Mxyz_true)/Mxyz_true
        print('\n % Error :\n',Mxyz_err2)
        #
        # Now try without noise
        #
        prop8data_nonoise = copy.deepcopy(prop8data)
        prop8data_nonoise['obs_seis'] = sdata_nonoise # noiseless seismogram data
        Mxyzfrom_mtrue_nonoise = BuildMxyz(Moment_LS([x,y,z],prop8data_nonoise))
        print('\n Best fit CMT using True location and noiseless data = \n',Mxyzfrom_mtrue_nonoise)
        Mxyz_err3 = 100*(Mxyzfrom_mtrue_nonoise - Mxyz_true)/Mxyz_true
        print('\n % Error :\n',Mxyz_err3)
    
    return
