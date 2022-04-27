#
#   Calculate and plot 2D fingerprint function from receiver function waveform
#   
#   Multiple plots:
#   
#       Plot of input waveform (RF) as a function of amplitude, u, and time, t.
#       Plot of signed distance function with input waveform as zero level set.
#       Plot of level sets of travel time/distance function d(u,t)
#       Plot of level sets of the PDF function e^{-|d(u,t)|/lambda}=PDF of fingerprint
#       Plot of 3D perspective view of the PDF of the fingerprint surface (only if plot3Dfsurface = True)
#
# by M. Sambridge, 2020.
#
import numpy as np
import pylab as plt
import skfmm
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
import time
#from scipy.spatial.distance import cdist

class Error(Exception):
    """Base class for other exceptions"""
    pass

class WaveformPFderivError(Exception):
    """Raised when WaveformFP.wdistderiv is called without first calling WaveformFP.wdist """
    def __init__(self,msg=''):
        super().__init__('\n WaveformFP.wdistderiv may only be called after WaveformFP.wdist \n')

class FingerprintMethodError(Exception):
    """Raised when WaveformFP.calcpdf is called with invalid method string """
    def __init__(self,msg=''):
        super().__init__('\n Method not recognized by WaveformFP.calcpdf \n')

class waveformFP(object): # Waveform object for waveform fingerprint calculations
    '''
    Object to encapsulate a seismic waveform and its nearest distance field (Fingerprint)
        
    '''
    def __init__(self,t,w,grid,fpgrid=None):   # introduced a separate fingerprint box from non-dimensional transformation
        '''
        Create a non-dimensionalized time amplitude window.
        
        Inputs:
            t - 1d array: time values of time series.
            w - 1d array: amplitudes of time series
            grid - tuple of 4floats,2ints; (t0,t1,u0,u1,Nu,Nt)
                du = float, amplitude limits of waveform
                u0 = float, lower limit on amplitude box in un-normalized co-ordinates
                u1 = float, upper limit on amplitude box in un-normalized co-ordinates
                t0 = float, lower limit on time axis in un-normalized co-ordinates
                t1 = float, upper limit on time axis in un-normalized co-ordinates
                Nu = int, number of grid points along amplitude axis of window
                Nt = int, len(wave); number of grid points along time axis of window
            fpgrid - option to impose a non unit transformed time-amplitude window. 

        '''
        (t0,t1, u0,u1,nug,ntg) = grid # unit box in transformed co-ordinates
        
        self.ntg = int(ntg)
        self.nug = int(nug)
        self.ulim = (u0,u1)
        self.tlim = (t0,t1)
        self.tlimn = ((t[0]-t0)/(t1-t0),(t[-1]-t0)/(t1-t0))                # waveform time range in normalised co-ordinates
        self.ulimn = (0.,1.) # not correct if fpgrid is active and NEVER used here anyway
        self.nt = len(t)

        if(fpgrid is None):  
            (fp_t0,fp_t1, fp_u0,fp_u1) = grid[0:4]   # fingerprint box is equal to unit box in non-dimensional co-ordinates
            self.tlimfp  = self.tlim                                 # fingerprint box in dimensional co-ordinates
            self.ulimfp  = self.ulim                                 # fingerprint box in dimensional co-ordinates
            self.tlimnfp = self.tlimn                                # fingerprint box in non-dimensional co-ordinates
            self.ulimnfp = self.ulimn                                # fingerprint box in non-dimensional co-ordinates
        else:  
            (fp_t0,fp_t1, fp_u0,fp_u1) = fpgrid[0:4] # fingerprint box differs from unit box in non-dimensional co-ordinates
            self.tlimfp  = (fp_t0,fp_t1)                               # fingerprint box in dimensional co-ordinates
            self.ulimfp  = (fp_u0,fp_u1)                               # fingerprint box in dimensional co-ordinates
            self.tlimnfp = ((fp_t0-t0)/(t1-t0),(fp_t1-t0)/(t1-t0))     # fingerprint box in non-dimensional co-ordinates
            self.ulimnfp = ((fp_u0-u0)/(u1-u0),(fp_u1-u0)/(u1-u0))     # fingerprint box in non-dimensional co-ordinates

        self.delgrid = np.array([(self.ulimnfp[1]-self.ulimnfp[0])/self.nug,(self.tlimnfp[1]-self.tlimnfp[0])/self.ntg])
        self.p = np.array([t,w]).T                                         # co-ordinates of waveform points in un-normalised co-ordinates
        self.pn = np.array([(t-t0)/(t1-t0),(w-u0)/(u1-u0)]).T              # co-ordinates of waveform points in normalised co-ordinates
        self.x0 = self.pn[:-1].reshape(1,self.nt-1,2)                      # co-ordinates of waveform segment start points in normalised co-ordinates
        self.delta_n = np.subtract(self.pn[1:], self.pn[:-1])              # vectors along waveform segments in normalised co-ordinates
        self.lsq_n = np.sum(np.multiply(self.delta_n,self.delta_n),axis=1) # waveform segment squared lengths in normalised co-ordinates
        self.dcalc = False
        self.drcalc = False

    def calcpdf(self,q=None,lambdav=0.04,deriv=False,method='Enumerate',verbose=False,nsegs=0):
        '''
        Create a density field for input waveform and optinally 
        derivatives of density field with respect to waveform amplitudes

        Inputs: (All optional)
            lambdav - float: the distance scaling parameter used in the density field estimation.
            method - string; specifies method for calculating nearest distance field (Enumerate (Default), FMM=Fast Marching, NNsearch: Nearest Neighbour search)
            verbose - print out calculation times.
            nsegs - Number of segments used by NNsearch method (Obsolete unless method ='NNsearch')
            q - Wasserstein order (Default 2)
            deriv - logical; flag to indicate derivates are to be calculated.
            chainmatrix - used to facilitate application of chain rule for derivatives.
            
        Outputs: 
            .pdf - ndarray; shape(Nt,Nu); amplitudes of 2D density of nearest distance field from waveform.
            .dfield - ndarray; shape(Nt,Nu); nearest distance field from waveform.

        '''

        if(lambdav is not None): lambdav = 0.04
        self.lam = lambdav
        if(method =='FMM' or method=='fmm'): # calculate nearest distance from waveform to each point on the grid using FMM
           #Xn, Yn = np.meshgrid(np.linspace(self.tlimn[0],self.tlimn[1],self.ntg), np.linspace(0.,1.,self.nug))
         # this looks like it has an ERROR Y should be Yn ? 
            Xn, Yn = np.meshgrid(np.linspace(self.tlimfp[0],self.tlimfp[1],self.ntg), np.linspace(self.ulimfp[0],self.ulimfp[1],self.nug))   # 2D mesh in un-normalized co-ordinates
            t0 = time.time()
            phi = -np.ones((self.nug,self.ntg))
            RFi = np.interp(np.linspace(self.tlimfp[0],self.tlimfp[1],self.ntg), self.p.T[0], self.p.T[1])
            # shouldn't this by Yn rather than Y, or rather Xn,Yn above should be X,Y?
            phi[Yn > RFi] = 1  # create function with waveform as zero contour for input to FMM
            self.dfield = skfmm.distance(phi, dx=self.delgrid) # evaluate signed distance function
            self.dfield = np.abs(self.dfield) # Remove sign of distance field produced by FMM
            twd = time.time()-t0
            self.type = 'FMM'
            self.pos = np.dstack((Xn,Yn))
            
        elif (method =='Enumerate'): # calculate distance field by vectorised brute force and record where on waveform is closest point to each in the grid
            t0 = time.time()
            self.wdist(deriv=deriv) # calculate distance field and rays for all grid points in normailized co-ordinates
            twd = time.time()-t0
            self.type = 'Enu'
            
        elif (method =='NNsearch'): # calculate distance field using sklearn NN search plus linear segment algorithm of MS
            t0 = time.time()
            self.wdistNN(deriv=deriv,nsegs=nsegs) # calculate distance field and rays for all grid points in normailized co-ordinates
            twd = time.time()-t0
            self.type = 'NNs'
        else:
            print(' Method string provided = '+method)
            raise FingerprintMethodError
            
        self.tcalc_fp = twd

        t0 = time.time()
        self.q = q
        if(self.q is None):
            self.pdf = np.exp(-np.abs(self.dfield)/self.lam) # calculate PDF from distance function
        elif(self.q == 2):
            self.pdf = np.exp(-self.dfield**q/self.lam) # calculate PDF from distance function
        self.tcalc_pdf = time.time()-t0
        
        if(verbose): 
            print(' calcpdf:\n'+' Time taken for distance field:',twd,'\n Time taken for pdf field:',self.tcalc_pdf)
            
    def PDFderiv(self,chainmatrix=None): # calculate derivatives of PDF field with respect to unormalized waveform amplitude
        if(type(chainmatrix) == np.ndarray):
            pdfrow = self.pdf.flatten()*chainmatrix.flatten()
        else: 
            pdfrow = self.pdf.flatten()
            
        if(self.q is not None): 
            if(self.q == 2): pdfrow = 2*pdfrow*np.abs(self.dfield.flatten())
    
        s = np.zeros(self.nt)
        x = self.dddy.T[0]
        y = self.dddy.T[1]
        for i in range(self.nt):
            s[i] = np.sum(x[self.irays==i]*pdfrow[self.irays==i])      # sum over ray segments upper point
            s[i] += np.sum(y[self.irays==i-1]*pdfrow[self.irays==i-1]) # sum over ray segments lower point

        self.pdfd = -s/self.lam

    def PDFderivMarg(self,chainmatrix): # calculate derivatives of PDF field with respect to unormalized waveform amplitude
        pdfrowX = self.pdf.flatten()*chainmatrix[0].flatten()
        pdfrowY = self.pdf.flatten()*chainmatrix[1].flatten()

        if(self.q is not None): # 
            if(self.q == 2): 
                pdfrowX = 2*pdfrowX*np.abs(self.dfield.flatten())
                pdfrowY = 2*pdfrowY*np.abs(self.dfield.flatten())
    
        s = np.zeros((2,self.nt))
        x = self.dddy.T[0]
        y = self.dddy.T[1]
        for i in range(self.nt):
            s[0,i] = np.sum(x[self.irays==i]*pdfrowX[self.irays==i])
            s[0,i] += np.sum(y[self.irays==i-1]*pdfrowX[self.irays==i-1])
            s[1,i] = np.sum(x[self.irays==i]*pdfrowY[self.irays==i])
            s[1,i] += np.sum(y[self.irays==i-1]*pdfrowY[self.irays==i-1])

        self.pdfdMarg = [-s[0]/self.lam,-s[1]/self.lam]

    def wdist(self,deriv=False): # Linear segment nearest distance field with vectorised calculations 
        '''
        Does the work for calculating a density field for waveform and its derivative with respect to amplitude and time of the corresponding waveform point.

        Inputs: (All optional)
            deriv - logical; flag to indicate derivates are to be calculated.
            
        Outputs
            .dfield - ndarray; shape(Nt,Nu) containing the nearest distance field.
                                Each point in nearest distance filed corresponds to a single point on the waveform.
            .irays - integer array; shape()

        '''
 
# NOTE HERE THAT THE self.dfield  RETURNED TREATS AMPLITDUE AS X AND TIME AS Y, WHEN CONSISTENCY SUGGESTS THIS SHOULD BE THE OTHER WAY AROUND. 
# MAYBE THIS IS FOLLOWING A CONVENTION IMPOSED BY THIRD PARTY FMM CODE? 

        self.dcalc = True
        nump = self.ntg*self.nug
        x0 = self.x0     # waveform segment start points in normalised co-ordinates reshaped
        c = self.delta_n # waveform segment vectors in normalised co-ordinates
        Lsq = self.lsq_n # waveform segment squared lengths in normalised co-ordinates
        # waveforms have local time range but common amplitude range 

        #Xn, Yn = np.meshgrid(np.linspace(0.,1.,self.ntg), np.linspace(0.,1.,self.nug))
        #Xn, Yn = np.meshgrid(np.linspace(self.tlimn[0],self.tlimn[1],self.ntg), np.linspace(0.,1.,self.nug)) # grid in normalised co-ordinates
        Xn, Yn = np.meshgrid(np.linspace(self.tlimnfp[0],self.tlimnfp[1],self.ntg), np.linspace(self.ulimnfp[0],self.ulimnfp[1],self.nug))
        p = (np.vstack((Xn.flatten(),Yn.flatten())).T).reshape(-1,1,2)
        b = p - x0
        lam = np.clip(np.sum(np.multiply(b,c),axis=2)/Lsq,0.,1.).reshape(-1,len(c),1)
        ds = b-c*lam
        dsq = np.sum(np.multiply(ds,ds),axis=2)
        iclose = np.argmin(dsq,axis=1)    
        l = lam[np.arange(nump),iclose,:]
        xclose = self.x0[0,iclose,:] + l*c[iclose,:]  
        d = np.sqrt(dsq[np.arange(nump),iclose])
                                                     # all output variables in normalized grid co-ordinates
        self.dfield = d.reshape((self.nug,self.ntg)) # nearest distance field to waveform
        self.irays = iclose                          # index of waveform segment closest to each grid point
        self.xrays = xclose                          # co-ordinates of point along waveform segment closest to each grid point 
        self.lrays = l.flatten()                     # lambda parameter for waveform segment closest to each grid point 
        self.pos = np.dstack((Xn,Yn))
        
        if(deriv):
            self.wdistderiv()

    def wdistNN(self,deriv=False,nsegs=0): # Linear segment nearest distance field with vectorised calculations 
 
        self.dcalc = True
        self.nsegs = nsegs
        Lsq = self.lsq_n # waveform segment squared lengths in normalised co-ordinates
        # waveforms have local time range but common amplitude range 

        #Xn, Yn = np.meshgrid(np.linspace(self.tlimn[0],self.tlimn[1],self.ntg), np.linspace(0.,1.,self.nug)) # grid in normalised co-ordinates
        Xn, Yn = np.meshgrid(np.linspace(self.tlimnfp[0],self.tlimnfp[1],self.ntg), np.linspace(self.ulimnfp[0],self.ulimnfp[1],self.nug)) # grid in normalised co-ordinates
        p = np.vstack((Xn.flatten(),Yn.flatten())).T

        if(nsegs!=0):
            nt = self.nt*(nsegs+1) - nsegs
            t = np.linspace(self.tlimnfp[0],self.tlimnfp[1],nt)
            u = np.interp(t, self.pn.T[0], self.pn.T[1])  
            pline = np.vstack((t,u)).T
        else:
            pline = self.pn
            
        neigh = NearestNeighbors(n_neighbors=1) # set number of nearest neighbours to find along the waveform 
        neigh.fit(pline)
        npall = neigh.kneighbors(p,return_distance=False)
        if(nsegs!=0): 
            npoints = np.round(npall/(nsegs+1)).astype(int).flatten()
        else:
            npoints = npall

        segp = np.clip(npoints,0,np.max(npoints)-1).flatten()
        segm = np.clip(npoints-1,0,np.max(npoints)-1).flatten()
        bp = p-self.pn[segp]
        cp = self.delta_n[segp]
        lamp = np.clip(np.sum(np.multiply(bp,cp),axis=1)/Lsq[segp],0.,1.).reshape(-1,1) # lambda for postive segment
        bm = p-self.pn[segm]
        cm = self.delta_n[segm]
        lamm = np.clip(np.sum(np.multiply(bm,cm),axis=1)/Lsq[segm],0.,1.).reshape(-1,1) # lambda for negative segment

        dsp = bp-cp*lamp
        dp = np.sum(np.multiply(dsp,dsp),axis=1) # squared nearest distance of segment to each node
        dsm = bm-cm*lamm
        dm = np.sum(np.multiply(dsm,dsm),axis=1) # squared nearest distance of segment to each node
        xraysp = self.pn[segp] +cp*lamp # closest point on linear segment to node
        xraysm = self.pn[segm] +cm*lamm # closest point on linear segment to node

        self.dfield = dm
        self.irays = segm
        self.xrays = xraysm
        self.lrays = lamm
        mask = dp < dm
        
        self.dfield[mask] = dp[mask]                 # nearest distance field to waveform
        self.irays[mask] = segp[mask]                # index of waveform segment closest to each grid point
        self.xrays[mask] = xraysp[mask]              # co-ordinates of point along waveform segment closest to each grid point 
        self.lrays[mask] = lamp[mask]                # lambda parameter for waveform segment closest to each grid point
        self.dfield = np.sqrt(self.dfield).reshape(self.nug,self.ntg)
        self.lrays = self.lrays.flatten()
        self.pos = np.dstack((Xn,Yn))
        
        if(deriv):
            self.wdistderiv()    

    def wdistderiv(self,verbose=False): # calculate derivatives of linear segment distance field with respect to waveform amplitude
        '''
        Does the work for calculating derivatives of the density field for waveform.

        Inputs: (All optional)
            verbose - logical; flag to return additional derivatives (dd/dx,dd/dy,dl/dy) for error checking 
        
        Outputs:
            .dddy - ndarray, shape (); instance `.dddy` containing the derivative of the nearest distance field with respect to waveform amplitude.
            
        

        '''
        if(not self.dcalc):
            raise WaveformPFderivError
        
        dis = self.dfield.reshape(-1,1)
        #Xn, Yn = np.meshgrid(np.linspace(0.,1.,self.ntg), np.linspace(0.,1.,self.nug))
        #Xn, Yn = np.meshgrid(np.linspace(self.tlimn[0],self.tlimn[1],self.ntg), np.linspace(0.,1.,self.nug))
        Xn, Yn = np.meshgrid(np.linspace(self.tlimnfp[0],self.tlimnfp[1],self.ntg), np.linspace(self.ulimnfp[0],self.ulimnfp[1],self.nug))
        p = np.vstack((Xn.flatten(),Yn.flatten())).T
        dx0dy0 = np.array([0., 1.])
        dx1dy1 = np.array([0., 1.])

        dddx = (self.xrays-p)/dis # derivative of distance to grid point wrt co-ordinates of intersection point on closest segment.

        x0 = self.x0.reshape(-1,2)[self.irays]     # waveform segment start points in normalised co-ordinates reshaped
        c = self.delta_n[self.irays]               # waveform segment vectors in normalised co-ordinates
    
        # calculate derivatives of minimum distance to segment from each input point wrt to y co-ordinate on segment end points

        dlamdy0 = (2*c.T[1]*self.lrays + np.sum(((p-dx0dy0)*c - (p-x0)*dx0dy0),axis=1))/(self.lsq_n[self.irays]) # derivative of lambda wrt y0
        dlamdy0[self.lrays==0] = 0.
        dlamdy0[self.lrays==1] = 0.
        dxdy0 = dx0dy0 + dlamdy0.reshape(-1,1)*c - self.lrays.reshape(-1,1)*dx0dy0 # derivative of nearest point on segment wrt y0

        dlamdy1 = (-2*c.T[1]*self.lrays + np.sum((p*c + (p-x0)*dx1dy1),axis=1))/(self.lsq_n[self.irays]) # derivative of lambda wrt y1
        dlamdy1[self.lrays==0] = 0. # imposed clipping upon derivatives
        dlamdy1[self.lrays==1] = 0.

        dxdy1 = dlamdy1.reshape(-1,1)*c + self.lrays.reshape(-1,1)*dx1dy1 # derivative of nearest point on segment wrt y1

        dddy0 = np.sum(dddx*dxdy0,axis=1) # derivatives of minimum distance wrt waveform amplitude in normalized co-ordinates
        dddy1 = np.sum(dddx*dxdy1,axis=1)
    
        du = self.ulim[1]-self.ulim[0]
        dddy0 /= du # derivatives of minimum distance wrt unnormalized waveform amplitude
        dddy1 /= du
    
        if(verbose):
            self.dddy = np.vstack((dddy0,dddy1)).T
            self.dldy = (dlamdy0,dlamdy1)
            self.dxdy = (dxdy0,dxdy1)
        else:
            self.dddy = np.vstack((dddy0,dddy1)).T

def NNsearch(wf,ni=0): # calculates irays,lrays,xrays and dfield to segment using NN search plus vectorised calculations on segment pairs.
    #Xn, Yn = np.meshgrid(np.linspace(wf.tlimn[0],wf.tlimn[1],wf.ntg), np.linspace(wf.ulimn[0],wf.ulimn[1],wf.nug))   # 2D mesh in normalized co-ordinates
    Xn, Yn = np.meshgrid(np.linspace(wf.tlimnfp[0],wf.tlimnfp[1],wf.ntg), np.linspace(wf.ulimnfp[0],wf.ulimnfp[1],wf.nug))   # 2D mesh in normalized co-ordinates
    points = np.vstack((Xn.flatten(),Yn.flatten())).T # build grid points
    if(ni!=0):
        nt = wf.ntg*(ni+1) -ni
        t = np.linspace(wf.tlimnfp[0],wf.tlimnfp[1],nt)
        u = np.interp(t, wf.pn.T[0], wf.pn.T[1])  
        pline = np.vstack((t,u)).T
    else:
        pline = wf.pn
    neigh = NearestNeighbors(n_neighbors=1) # set number of nearest neighbours to find along the waveform 
    neigh.fit(pline)
    npall = neigh.kneighbors(points,return_distance=False)
    npoints = np.round(npall/(ni+1)).astype(int).flatten()
    
    segp = np.clip(npoints,0,np.max(npoints)-1).flatten()
    segm = np.clip(npoints-1,0,np.max(npoints)-1).flatten()
    Lsq = wf.lsq_n
    bp = points-wf.pn[segp]
    cp = wf.delta_n[segp]
    lamp = np.clip(np.sum(np.multiply(bp,cp),axis=1)/Lsq[segp],0.,1.).reshape(-1,1) # lambda for postive segment
    bm = points-wf.pn[segm]
    cm = wf.delta_n[segm]
    lamm = np.clip(np.sum(np.multiply(bm,cm),axis=1)/Lsq[segm],0.,1.).reshape(-1,1) # lambda for negative segment

    dsp = bp-cp*lamp
    dp = np.sum(np.multiply(dsp,dsp),axis=1) # squared nearest distance of segment to each node
    dsm = bm-cm*lamm
    dm = np.sum(np.multiply(dsm,dsm),axis=1) # squared nearest distance of segment to each node
    xraysp = wf.pn[segp] +cp*lamp # closest point on linear segment to node
    xraysm = wf.pn[segm] +cm*lamm # closest point on linear segment to node

    if(False):
        dfield = np.copy(dp)
        irays = np.copy(segp)
        xrays = np.copy(xraysp)
        lrays = np.copy(lamp)
        mask = dm < dp
        dfield[mask] = dm[mask]
        irays[mask] = segm[mask]
        xrays[mask] = xraysm[mask]
        lrays[mask] = lamm[mask]
        dfield = np.sqrt(dfield).reshape(wf.nug,wf.ntg)
        lrays = lrays.flatten()
    else:
        dfield = dm
        irays = segm
        xrays = xraysm
        lrays = lamm
        mask = dp < dm
        dfield[mask] = dp[mask]
        irays[mask] = segp[mask]
        xrays[mask] = xraysp[mask]
        lrays[mask] = lamp[mask]
        dfield = np.sqrt(dfield).reshape(wf.nug,wf.ntg)
        lrays = lrays.flatten()
    return dfield,irays,lrays,xrays

def wavedist(point,wf):
    c = wf.delta_n # waveform segment vectors in normalised co-ordinates
    Lsq = wf.lsq_n # Squared length of waveform segment vectors in normalised co-ordinates
    b = point-wf.pn[:-1]
    lam = np.clip(np.sum(np.multiply(b,c),axis=1)/Lsq,0.,1.).reshape(len(c),1)
    ds = b-c*lam
    dsq = np.sum(np.multiply(ds,ds),axis=1)
    iclose = np.argmin(dsq)
    xclose = wf.pn[:-1][iclose] + lam[iclose]*c[iclose]
    return np.sqrt(dsq[iclose]),iclose,xclose

def wavedistv(point,wf): # Linear segment nearest distance field with vectorised calculations 
    nump = len(point)
    if(point.ndim==1): nump = 1
    x0 = wf.x0     # waveform segment start points in normalised co-ordinates reshaped
    c = wf.delta_n # waveform segment vectors in normalised co-ordinates
    Lsq = wf.lsq_n


    p = point.reshape(-1,1,2)
    b = p - x0
    lam = np.clip(np.sum(np.multiply(b,c),axis=2)/Lsq,0.,1.).reshape(-1,len(c),1)
    ds = b-c*lam
    dsq = np.sum(np.multiply(ds,ds),axis=2)
    iclose = np.argmin(dsq,axis=1)    
    l = lam[np.arange(nump),iclose,:]
    xclose = x0[0,iclose,:] + l*c[iclose,:]  
    dsqout = np.sqrt(dsq[np.arange(nump),iclose])
    
    return dsqout,iclose,xclose,l.flatten()

    #DOES THIS ACCOUNT FOR CASE WHERE irays = i-1, lam=1, xrays = x_i-1 is same as irays=i, lam=0, xrays=x_i ? TESTS SUGGEST It produces different results fr derivative in these cases.

def wavederiv(d,irays,xrays,lrays,points,wf,verbose=False): # calculate derivatives of linear segment distance field with respect to waveform amplitude
    #p = points.reshape(np.shape(xrays))
    dis = d.reshape(-1,1)
    p = points
    dx0dy0 = np.array([0., 1.])
    dx1dy1 = np.array([0., 1.])

    dddx = (xrays-p)/dis

    x0 = wf.x0.reshape(-1,2)[irays]     # waveform segment start points in normalised co-ordinates reshaped
    c = wf.delta_n[irays]               # waveform segment vectors in normalised co-ordinates
    #l = np.sqrt(wf.lsq_n[irays])       # distance along segment
    
    # calculate derivatives of minimum distance to segment from each input point wrt to y co-ordinate on segment end points

    dlamdy0 = (2*c.T[1]*lrays + np.sum(((p-dx0dy0)*c - (p-x0)*dx0dy0),axis=1))/(wf.lsq_n[irays]) # derivative of lambda wrt y0
    dlamdy0[lrays==0] = 0. # imposed clipping upon derivatives
    dlamdy0[lrays==1] = 0.
    dxdy0 = dx0dy0 + dlamdy0.reshape(-1,1)*c - lrays.reshape(-1,1)*dx0dy0 # derivative of nearest point on segment wrt y0

    dlamdy1 = (-2*c.T[1]*lrays + np.sum((p*c + (p-x0)*dx1dy1),axis=1))/(wf.lsq_n[irays]) # derivative of lambda wrt y1
    dlamdy1[lrays==0] = 0. # imposed clipping upon derivatives
    dlamdy1[lrays==1] = 0.

    dxdy1 = dlamdy1.reshape(-1,1)*c + lrays.reshape(-1,1)*dx1dy1 # derivative of nearest point on segment wrt y1

    dddy0 = np.sum(dddx*dxdy0,axis=1) # derivatives of minimum distance wrt waveform amplitude in normalized co-ordinates
    dddy1 = np.sum(dddx*dxdy1,axis=1)
    
    du = wf.ulim[1]-wf.ulim[0]
    dddy0 /= du # derivatives of minimum distance wrt unnormalized waveform amplitude
    dddy1 /= du
    
    if(verbose):
        return np.vstack((dddy0,dddy1)).T,dlamdy0,dlamdy1,dxdy0,dxdy1
    else:
        return np.vstack((dddy0,dddy1)).T

def check_FDderiv(wf,k,du=0.001,verbose=False): # finite difference test routine for derivatives of distance filed wrt waveform amplitudes.
 # accuracy problems can occur with this finite difference routine if pertubations of waveform amplitudes change which segement is the closest to grid point   
    t = wf.p.T[0]
    RF = wf.p.T[1]
    u0,u1 = wf.ulim
    t0,t1 = wf.tlim
    nug = wf.nug
    #du = u1-u0
    irays = wf.irays
    i = irays[k] # segment
    dups = du*np.abs(RF[i])

    Xn, Yn = np.meshgrid(np.linspace(wf.tlimnfp[0],wf.tlimnfp[1],wf.ntg), np.linspace(wf.ulimnfp[0],wf.ulimnfp[1],wf.nug))
    points = np.vstack((Xn.flatten(),Yn.flatten())).T

    RFp = np.copy(RF)
    RFp[i]+=dups # pertub waveform +ve
    wfp = waveformFP(t,RFp,(t0,t1,u0,u1,nug,wf.ntg))
        
    dp,iraysp0,xraysp,lraysp = wavedistv(points,wfp) # calculate distance field and rays for all grid points in normailized co-ordinates
    
    RFm = np.copy(RF)
    RFm[i]-=dups # pertub waveform +ve
    wfm = waveformFP(t,RFm,(t0,t1,u0,u1,nug,wf.ntg))
    
    dm,iraysm0,xraysm,lraysm = wavedistv(points,wfm) # calculate distance field and rays for all grid points in normailized co-ordinates
    
    dddy0fd = (dp[k]-dm[k])/(2*dups)
    #dldy0fd = (lraysp[k]-lraysm[k])/(2*dups)
    #dxdy0fd = (xraysp[k]-xraysm[k])/(2*dups)
    
    RFp = np.copy(RF)
    RFp[i+1]+=dups # pertub waveform +ve
    wfp = waveformFP(t,RFp,(t0,t1,u0,u1,nug,wf.ntg))
        
    dp,iraysp1,xraysp,lraysp = wavedistv(points,wfp) # calculate distance field and rays for all grid points in normailized co-ordinates
    
    RFm = np.copy(RF)
    RFm[i+1]-=dups # pertub waveform +ve
    wfm = waveformFP(t,RFm,(t0,t1,u0,u1,nug,wf.ntg))
    
    dm,iraysm1,xraysm,lraysm = wavedistv(points,wfm) # calculate distance field and rays for all grid points in normailized co-ordinates
    
    dddy1fd = (dp[k]-dm[k])/(2*dups)
    #dldy1fd = (lraysp[k]-lraysm[k])/(2*dups)
    #dxdy1fd = (xraysp[k]-xraysm[k])/(2*dups)
        
    #print('grid point:',k,'ray segment: ',i,' FD derivatives : ',dddy0fd,dddy1fd,' : dddy',dddy[k])
    #print(k,i,dddy0fd,dddy1fd,dddy[k])
    if(verbose):
        print('\n segments after FD perturbation : ',' pos 0 ',iraysp0[k],' minus 0',iraysm0[k],  'pos 1 ',iraysp1[k],' minus 1',iraysm1[k])
    #if(verbose):
    #    print(' lambda: ',i,' : ',dldy0fd,' : ',dlamdy0[k],dlamdy1[k])
    #    print(' x: ',i,' : ',dxdy0fd,' : ',dxdy0[k],dxdy1[k])
    #return dddy0fd,dddy1fd,iraysp0[k],iraysm0[k],iraysp1[k],iraysm1[k]
    return i,dddy0fd,dddy1fd

def check_FDchain_orig(wf,deriv,lambdav,dufd=0.0001): # Finite difference derivatives of sum of PDFs wrt waveform amplitude
    
    t = wf.p.T[0]
    RF = wf.p.T[1]
    u0 = wf.ulim[0]
    u1 = wf.ulim[1]
    t0 = wf.tlim[0]
    t1 = wf.tlim[1]
    nug = wf.nug

    Xn, Yn = np.meshgrid(np.linspace(wf.tlimnfp[0],wf.tlimnfp[1],wf.ntg), np.linspace(wf.ulimnfp[0],wf.ulimnfp[1],wf.nug))
    points = np.vstack((Xn.flatten(),Yn.flatten())).T

    print('Calculating finite difference derivatives of sum of PDF amplitude')
    for i in range(wf.nt):  # compare derivatives to finite difference. 

        RFp = np.copy(RF)
        RFp[i]+=dufd # pertub waveform +ve
        wfp = waveformFP(t,RFp,(t0,t1,u0,u1,nug,wf.ntg))
        
        dp,iraysp0,xraysp,lraysp = wavedistv(points,wfp) # calculate distance field and rays for all grid points in normailized co-ordinates

        PDFdp = np.exp(-np.abs(dp)/lambdav) # calculate PDF from distance function
        sp = np.sum(PDFdp)
    
        RFm = np.copy(RF)
        RFm[i]-=dufd # pertub waveform +ve
        wfm = waveformFP(t,RFm,(t0,t1,u0,u1,nug,wf.ntg))
    
        dm,iraysm0,xraysm,lraysm = wavedistv(points,wfm) # calculate distance field and rays for all grid points in normailized co-ordinates

        PDFdm = np.exp(-np.abs(dm)/lambdav) # calculate PDF from distance function
        sm = np.sum(PDFdm)
   
        dsdyfd = (sp-sm)/(2*dufd)
        print(i,' : ',dsdyfd,deriv[i])
    return

def check_FDchain(wf,lambdav,dufd=0.0001): # Finite difference derivatives of sum of PDFs wrt waveform amplitude
    
    t = wf.p.T[0]
    RF = wf.p.T[1]
    u0 = wf.ulim[0]
    u1 = wf.ulim[1]
    t0 = wf.tlim[0]
    t1 = wf.tlim[1]
    nug = wf.nug

    Xn, Yn = np.meshgrid(np.linspace(wf.tlimnfp[0],wf.tlimnfp[1],wf.ntg), np.linspace(wf.ulimnfp[0],wf.ulimnfp[1],wf.nug))
    points = np.vstack((Xn.flatten(),Yn.flatten())).T

    #print('Calculating finite difference derivatives of sum of PDF amplitude')
    for i in range(wf.nt):  # compare derivatives to finite difference. 

        RFp = np.copy(RF)
        RFp[i]+=dufd # pertub waveform +ve
        wfp = waveformFP(t,RFp,(t0,t1,u0,u1,nug,wf.ntg))
        
        dp,iraysp0,xraysp,lraysp = wavedistv(points,wfp) # calculate distance field and rays for all grid points in normailized co-ordinates

        PDFdp = np.exp(-np.abs(dp)/lambdav) # calculate PDF from distance function
        sp = np.sum(PDFdp)
    
        RFm = np.copy(RF)
        RFm[i]-=dufd # pertub waveform +ve
        wfm = waveformFP(t,RFm,(t0,t1,u0,u1,nug,wf.ntg))
    
        dm,iraysm0,xraysm,lraysm = wavedistv(points,wfm) # calculate distance field and rays for all grid points in normailized co-ordinates

        PDFdm = np.exp(-np.abs(dm)/lambdav) # calculate PDF from distance function
        sm = np.sum(PDFdm)
   
        dsdyfd = (sp-sm)/(2*dufd)
        #print(i,' : ',dsdyfd,deriv[i])
    return dsdyfd

def wPDFderiv(pdf,dddy,lambdav,irays,wf,chainmatrix): # calculate derivatives of PDF field with respect to unormalized waveform amplitude
    pdfrow = pdf.flatten()*chainmatrix.flatten()
    
    s = np.zeros(wf.nt)
    x = dddy.T[0]
    y = dddy.T[1]
    for i in range(wf.nt):
        s[i] = np.sum(x[irays==i]*pdfrow[irays==i])
        s[i] += np.sum(y[irays==i-1]*pdfrow[irays==i-1])

    return -s/lambdav

#-----------------------------------------------------------------
# plot Receiver function with contoured signed distance function

def plot_RF_SDF(t,RFo,ltype='b-',string="Predicted receiver function",grid=False,legend=False):
    fig, ax = plt.subplots(figsize=(8,4))
    plt.title(string)
    plt.xlabel("Time, t (s)")
    plt.ylabel("Amplitude, u")
    plt.grid(grid)
    if(len(RFo)!=0):plt.plot(t, RFo, '-', color='grey',label='Noisy Receiver Function')
    plt.plot(t, np.zeros(np.shape(RFo)), '--', linewidth=0.5,color='grey')
    if(legend): plt.legend()
    #print(plt.xlim(),plt.ylim())
    xl,yl = plt.xlim(),plt.ylim()
    plt.show()
    return xl,yl

def plotPDFsurface(pdf,t,ridge,mycmap,elev=75,azim=-134):
    ax = Axes3D(plt.figure(figsize=(16,16)))

    #mymap = 'viridis_r'
    #mymap = cm.copper_r
    #mymap = cm.cubehelix_r

    #ls = LightSource(azdeg=-142, altdeg=66)
    ls = LightSource(azdeg=40, altdeg=45)
    illuminated_surface = ls.shade(PDFd, cmap=mycmap)

    ax.set_xlabel('Time, t')
    ax.set_ylabel('Waveform amplitude, u')
    ax.set_zlabel('PDF amplitude');
    #ax.view_init(75,-134)
    ax.view_init(elev=elev,azim=azim)
    plt.title('PDF = $e^{-|d(u,t)|/\lambda}$')

    ax.plot_surface(X, Y, pdf, antialiased=False, rstride=2, cstride=2,cmap=mycmap,facecolors=illuminated_surface)
    ax.plot(t, ridge, np.ones(np.shape(ridge)),lw=1.,color='w',zorder=99) # plot waveform as white line
    plt.show()

def plot_phi(X,Y,phi,t,waveform,xl,yl):
    plt.figure(figsize=(8,4))
    plt.xlim(xl[0],xl[1])
    plt.ylim(yl[0],yl[1])
    plt.xlabel("t")
    plt.ylabel("u")
    plt.contour(X, Y, phi,[0], linewidths=(1), colors='grey')
    plt.contourf(X, Y, phi,[-1, 0,1], colors=['lightgray','powderblue'])
    plt.plot(t, waveform, '-', color='green',lw=0.5,label='Noisy Receiver Function')
    plt.title('Zero contour of $d(u,t)$')
    plt.show()
    return

def plot_rays_discrete(X,Y,f,phi,t,waveform,xl,yl,title,col1,col2,darg,q,points):

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    (Nu,Nt) = np.shape(X)
    Xn, Yn = np.meshgrid(np.linspace(0,1,Nt), np.linspace(0,1,Nu))
    plt.contour(Xn, Yn, phi,[0], linewidths=(1), colors=col1)
    plt.contour(Xn, Yn, f, 30,linewidths=0.5,colors=col2)
    u0=Y[0,0]
    du = Y[-1,0] -u0
    wp = np.where(q>=1)
    #wn = np.where(q<=-1)
    for k in range(len(points)):
        i = points[k][0]
        j = points[k][1]
        x0,y0 = Xn[i,j],Yn[i,j]
        ii = wp[1][darg[i,j]] # t index of ray path origin from grid point i,j
        jj = wp[0][darg[i,j]] # u index of ray path origin from grid point i,j
        #if(phi[i,j] == 1.):     
        #    ii = wp[1][darg[i,j]] # t index of ray path origin from grid point i,j
        #    jj = wp[0][darg[i,j]] # u index of ray path origin from grid point i,j
        #if(phi[i,j] == -1.):     
        #    ii = wn[1][darg[i,j]] # t index of ray path origin from grid point i,j
        #    jj = wn[0][darg[i,j]] # u index of ray path origin from grid point i,j
        y1 = Yn[jj][0]
        x1 = Xn[0][ii]
        plt.plot([x0,x1],[y0,y1],'b-',lw=0.5)
        plt.plot(np.linspace(0,1,Nt), (waveform-u0)/du, '-', color='green',lw=0.5,label='Noisy Receiver Function')
    if(True):
        plt.plot(Xn[wp],Yn[wp],'o',lw=0.5)
        plt.plot(Xn[np.where(q==2)],Yn[np.where(q==2)],'ro')
        plt.plot(Xn[np.where(q==-2)],Yn[np.where(q==-2)],'go')
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("u")
    plt.show()
    return

def plot_rays(plotind,wf,title,col1,col2,filename='no',fxsize=None,fysize=None):

    if(fxsize is None): fxsize = 9
    if(fysize is None): fysize = 9
    fig = plt.figure(figsize=(fxsize,fysize))
    #Nu = np.int(np.shape(xrays)[0]/wf.nt)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    Xn, Yn = np.meshgrid(np.linspace(wf.tlimnfp[0],wf.tlimnfp[1],wf.ntg), np.linspace(wf.ulimnfp[0],wf.ulimnfp[1],wf.nug))
    Xn = Xn.flatten()
    Yn = Yn.flatten()
    
    for k in range(len(plotind)):
        kk = plotind[k]
        x0,y0 = Xn[kk],Yn[kk]
        y1 = wf.xrays[kk][1]
        x1 = wf.xrays[kk][0]
        plt.plot([x0,x1],[y0,y1],'b-',lw=0.5)
        plt.plot(x1,y1,'ro',markersize=2.0)
        plt.plot(wf.pn.T[0], wf.pn.T[1], '-', color='green',lw=0.5,label='Noisy Receiver Function')
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("u")
    if(filename !='no'): plt.savefig(filename)
    plt.show()
    return

def plot_LS(f,wf,xl,yl,title,col1,col2,aspect=False,filename='no',pdf=False,ncon=10,fxsize=None,fysize=None):
    
    if(aspect):
        if(fxsize is not None and fysize is not None):
            fig = plt.figure(figsize=(fxsize,fysize))
        else:
            fig = plt.figure(figsize=(9,9))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        #Xn, Yn = np.meshgrid(np.linspace(0,1,wf.ntg), np.linspace(0,1,wf.nug))
        Xn, Yn = np.meshgrid(np.linspace(wf.tlimnfp[0],wf.tlimnfp[1],wf.ntg), np.linspace(wf.ulimnfp[0],wf.ulimnfp[1],wf.nug))
        plt.plot(wf.pn.T[0],wf.pn.T[1],'-',color=col1)
        plt.contour(Xn, Yn, f, 3*ncon,linewidths=0.5,colors=col2)
    else:
        if(fxsize is not None and fysize is not None):
            fig = plt.figure(figsize=(fxsize,fysize))
        else:
            fig = plt.figure(figsize=(8,4))

        plt.xlim(xl[0],xl[1])
        plt.ylim(yl[0],yl[1])
        plt.ylim(u0,u1)
        #plt.contour(X, Y, phi,[0], linewidths=(1), colors=col1)
        Xs, Ys = np.meshgrid(np.linspace(wf.tlimfp[0],wf.tlimfp[1],Nt), np.linspace(wf.ulimfp[0],wf.ulimfp[1],Nu))   # 2D mesh in un-normalized co-ordinates
        plt.plot(wf.p.T[0],wf.p.T[1],'-',color=col1)
        plt.contour(Xs, Ys, f, 2*ncon,linewidths=0.5,colors=col2)
        #plt.plot(t, waveform, '-', color='green',lw=0.5,label='Noisy Receiver Function')
    #plt.title('Level sets of the PDF field $e^{-|d(u,t)|/\lambda}$')
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("u")
    if(filename !='no'): plt.savefig(filename)
    plt.show()
    return 

def plot_2LS(wf1,wf2,title1,title2,col1,col2,filename='no',pdf=False,ncon=10,fxsize=None,fysize=None,aspect=False):
    
    if(fxsize is not None and fysize is not None):
        fig = plt.figure(figsize=(fxsize,fysize))
    else:
        fig = plt.figure(figsize=(18,18))
    ax = fig.add_subplot(121)
    if(aspect): ax.set_aspect('equal')
    Xn, Yn = np.meshgrid(np.linspace(wf1.tlimnfp[0],wf1.tlimnfp[1],wf1.ntg), np.linspace(wf1.ulimnfp[0],wf1.ulimnfp[1],wf1.nug))
    plt.plot(wf1.pn.T[0],wf1.pn.T[1],'-',color=col1)
    if(pdf):
        plt.contour(Xn, Yn, wf1.pdf, 3*ncon,linewidths=0.5,colors=col2)
    else:
        plt.contour(Xn, Yn, wf1.dfield, 3*ncon,linewidths=0.5,colors=col2)
    plt.title(title1)
    plt.xlabel("t")
    plt.ylabel("u")
    ax = fig.add_subplot(122)
    if(aspect): ax.set_aspect('equal')
    Xn, Yn = np.meshgrid(np.linspace(wf2.tlimnfp[0],wf2.tlimnfp[1],wf2.ntg), np.linspace(wf2.ulimnfp[0],wf2.ulimnfp[1],wf2.nug))
    plt.plot(wf2.pn.T[0],wf2.pn.T[1],'-',color=col1)
    if(pdf):
        plt.contour(Xn, Yn, wf2.pdf, 3*ncon,linewidths=0.5,colors=col2)
    else:
        plt.contour(Xn, Yn, wf2.dfield, 3*ncon,linewidths=0.5,colors=col2)
    plt.title(title2)
 
    plt.xlabel("t")
    plt.ylabel("u")
    if(filename !='no'): plt.savefig(filename)
    plt.show()
    return 

def plotMarginals(wfwave,wf,tag='_'):
    wf.setMarginals()  # Calculate marginals
    
    # plot time marginal
    fig = plt.figure(figsize=(9,1))
    plt.plot(wf.marg[1].x,wf.marg[1].pdf)
    plt.fill_between(wf.marg[1].x, 0, wf.marg[1].pdf)
    plt.xlim(wf.marg[1].x[0],wf.marg[1].x[-1])
    plt.tick_params(left=False,bottom=True,labelleft=False,labelbottom=False)
    if(tag != '-'): 
        plt.savefig('Marginal_u'+tag+'.png',dpi=300)
    else:
        plt.savefig('Marginal_u.png',dpi=300)
        
    plt.show
        
    fig = plt.figure(figsize=(9,1))
    plt.plot(wf.marg[0].x,wf.marg[0].pdf)
    plt.fill_between(wf.marg[0].x, 0, wf.marg[0].pdf)
    plt.xlim(wf.marg[0].x[0],wf.marg[0].x[-1])
    plt.tick_params(left=False,bottom=True,labelleft=False,labelbottom=False)
    if(tag != '-'): 
        plt.savefig('Marginal_t'+tag+'.png',dpi=300)
    else:
        plt.savefig('Marginal_t.png',dpi=300)
    plt.show

    if(tag != '-'): 
        filename = 'Marginals_and_fingerprint'+tag+'.pdf'
    else:
        filename = 'Marginals_and_fingerprint.pdf'

    plot_LS(wfwave.dfield,wfwave,None,None," ",'black','grey',aspect=True,filename=filename,pdf=False)
    return

def calcFMM_dist_deriv(d,deltax):
    dy, dx = np.gradient(d,deltax[0],deltax[1])
    (Nu, Nt) = np.shape(d)
    Xn, Yn = np.meshgrid(np.linspace(0,1,Nt), np.linspace(0,1,Nu))
    A = np.sqrt(np.multiply(dx,dx)+np.multiply(dy,dy))
    dy = np.divide(dy, A, out=np.zeros_like(dy), where=A!=0)
    dx = np.divide(dx, A, out=np.zeros_like(dx), where=A!=0)
    Xw = Xn-np.multiply(d,dx) # Time co-ordinate of end point of rays
    Yw = Yn-np.multiply(d,dy) # Amplitude co-ordinate of end point of rays
    Xw[np.where(A==0)] = 0.
    Yw[np.where(A==0)] = 0.
    
    return Xw,Yw

def find_raystart_point_with_gradient(d,deltax): # failed attempt to locate ray start points from fmm grid using np.gradient and distance
    Xw,Yw = calcFMM_dist_deriv(d,deltax)
    A = Xw/deltax[1]
    A = A.astype(int)
    B = Yw/deltax[0]
    B = B.astype(int)
    A = np.clip(A,0,Nt-1)
    B = np.clip(B,0,Nu-1)
    dl = np.abs(d[B.flatten(),A.flatten()])
    dlev = np.max(dl)/200.
    px = Xw.flatten()
    py = Yw.flatten()
    cx = px[dl < dlev]
    cy = py[dl < dlev]
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    plt.plot(px,py,'k.')
    plt.plot(cx,cy,'c.')
    plt.title('Test of ray end points')
    plt.xlabel("t")
    plt.ylabel("u")
    plt.show()

#-----------------------------------------------------------------

if __name__ == "__main__":
    import rf

    # Set up Vs seismic model
    vtype = 2 # format for velocitry model
    velmod = np.zeros([13,3]) # Set up a velocity reference model in Interface depth format
    velmod[0] = [20.0, 3.4600, 1.73]
    velmod[1] = [35.0, 3.8500, 1.73]
    velmod[2] = [77.5, 4.4850, 1.73]
    velmod[3] = [120.0, 4.4950, 1.73]
    velmod[4] = [165.0, 4.5045, 1.73]
    velmod[5] = [210.0, 4.5145, 1.73]
    velmod[6] = [260.0, 4.5660, 1.73]
    velmod[7] = [310.0, 4.6525, 1.73]
    velmod[8] = [360.0, 4.7395, 1.73]
    velmod[9] = [410.0, 4.8265, 1.73]
    velmod[10] = [460.0, 5.1330, 1.73]
    velmod[11] = [510.0, 5.2390, 1.73]
    velmod[12] = [510.0, 5.3450, 1.73]

#----------------------------OPTIONS-------------------------------------

    plot3Dfsurface =True # option for 3D perspective plot of fingerprint PDF
    plot3Dfsurface =False # option for 3D perspective plot of fingerprint PDF
    fmm = False # option to use Fast Marching or scipy distance function
    fmm = True # option to use Fast Marching or scipy distance function

#------------------------------------------------------------------------

# calculate RF waveforms
#                                                   Note after restart of kernel the default observed is called `obsx' in Figures

    time1, RFo = rf.rfcalc(velmod,sn=0.25,mtype=vtype,seed=61254557) # calculate noisy `observed' RF waveform
    time2, RFp = rf.rfcalc(velmod,mtype=vtype)         # calculate noiseless `predicted' RF waveform

    #RF,t = RFo[::100],  time1[::100] # smallest test problem
    RF,t = RFo,time1                 # Full problem
    RF,t = RFo[::10],  time1[::10]   # smaller test problem

    xl,yl = plot_RF_SDF(t,RF)        # Plot single noisy RF and return limits in time and amplitude

# scaling

    du = yl[1]-yl[0]
    u0 = yl[0]-0.15*du               # lower limit on amplitude box in un-normalized co-ordinates
    u1 = yl[1]+0.15*du               # upper limit on amplitude box in un-normalized co-ordinates
    Nu,Nt = int(1.3*len(RF)),len(RF) # number of grid points in each window
    Nu,Nt = 800,600
    Du,Dt = u1-u0,t[-1]-t[0]         # window lengths in un-normalized co-ordinates
    deltax = np.array([0.,0.])       # grid spacing 
    
# unit box system
    sigma_u = Du # reference distance along u axis
    sigma_t = Dt # reference distance along t axis
    lambdav = 0.04

    deltax[1] = Dt/(sigma_t*Nt) # spacing in t grid points
    deltax[0] = Du/(sigma_u*Nu) # spacing in x grid points
    
# create waveform object

    wf = waveformFP(t,RF,(t[0],t[-1],u0,u1,Nu,Nt))

# create waveform fingerprint PDF
    
    wf.calcpdf(verbose=True,deriv=False,method='fmm') # calculate Fingerprint
    
# create 2D mesh
    X, Y = np.meshgrid(np.linspace(t[0],t[-1],Nt), np.linspace(u0,u1,Nu))   # 2D mesh in un-normalized co-ordinates
    #Xn, Yn = np.meshgrid(np.linspace(0.,1.,Nt), np.linspace(0.,1.,Nu)) # 2D mesh in normalized co-ordinates
    Xn, Yn = np.meshgrid(np.linspace(wf.tlimnfp[0],wf.tlimnfp[1],wf.ntg), np.linspace(wf.ulimnfp[0],wf.ulimnfp[1],wf.nug))

    time0 = time.time()

    if(fmm): # calculate nearest distance from waveform to each point on the grid using FMM
        phi = -np.ones_like(X) # create function with waveform as zero contour for input to FMM
        RFi = np.interp(np.linspace(t[0],t[-1],Nt), t, RF)
        phi[Y > RFi] = 1  # Set all points above waveform to 1. This only works if waveform has same dimension as grid.
        plot_phi(X,Y,phi,t,RF,xl,yl) #Plot level sets of phi 
        dfmm = skfmm.distance(phi, dx=deltax) # evaluate signed distance function
        dplot = dfmm
    else: # calculate distance field by vectorised brute force and record where on waveform is closest point to each in the grid

        points = np.vstack((Xn.flatten(),Yn.flatten())).T
        d,irays,xrays,lrays = wavedistv(points,wf) # calculate distance field and rays for all grid points in normailized co-ordinates
        dplot = d.reshape((Nu,Nt))
    
    tskfmm = time.time() - time0

    PDFd = np.exp(-np.abs(dplot)/lambdav) # calculate PDF from distance function

    # plot level sets of distance function

    # plot_LS(dplot,wf,xl,yl,'Level sets of the distance field $d(u,t)$','grey','grey',aspect=True) 
    plot_LS(wf.dfield,wf,xl,yl,'Level sets of the distance field $d(u,t)$','grey','grey',aspect=True,ncon=50) 

    # plot level sets of PDF
    #plot_LS(PDFd,wf,xl,yl,

    plot_LS(wf.pdf,wf,xl,yl,
        'Level sets of the PDF field $e^{-|d(u,t)|/\lambda}$',
        'black','green') 

    if(plot3Dfsurface): # Plot PDF of fingerprint surface using lightsource
        plotPDFsurface(wf.pdf,t,RF,cm.cubehelix_r)
        plotPDFsurface(wf.pdf,t,RF,cm.copper_r)

    print(' Lambda ',lambdav,'\n Nt \t\t',Nt,'\n Nu \t\t',Nu,'\n Dt \t\t',Dt,'\n Du \t\t',Du,'\n Dmin \t\t',np.min(wf.dfield),'\n Dmax \t\t',np.max(wf.dfield))
    print(' Tmin \t\t',t[0],'\n Tmax \t\t',t[-1])
    print(' Umin \t\t',u0,'\n Umax \t\t',u1)
    print(' PDFmin \t',np.min(PDFd),'\n PDFmax \t',np.max(PDFd))

    print('\n Time for skfmm : ',tskfmm)

    # example plot showing raypaths traced from distance field back to waveform

    if(not fmm): # test plot results of raypaths calculated with brute force linear segment algorithm
    
    # plot random selection of rays from grid points back to waveform
    
        plotind = np.random.randint(0, Nu*Nt,1000)
        plot_rays(plotind,wf,
                  'Level sets of the distance field $d(u,t)$ with example raypaths','cyan','grey')
    
    # NB: Only run finite difference checks below when number of points in waveform is down sampled otherwise takes too long

    if(False): # test derivatives of brute force linear segment algorithm with finite differences
               # NB: Only run this when number of points in waveform is down sampled otherwise takes too long
        wf.calcpdf(verbose=True,deriv=True,fmm=fmm)
        #dddy = wavederiv(wf.deriv,wf.irays,wf.xrays,wf.lrays,points,wf) # calculate derivatives of distance field with respect to normailized waveform co-ordinates
        print(' Time taken to calculate derivatives of d wrt to waveform amplitude',wf.tcalc_fp)
    
        # compare derivatives of brute force linear segment algorithm with finite difference
        kv = np.arange(wf.dfield.size) # cycling over all derivatives would take ~30s x 2 x NtxNu
        kv = np.random.choice(wf.dfield.size, 100) # random selection of 100 derivatives
        print('Calculating finite difference derivatives')
        for k in kv:  # compare derivatives to finite difference. This will take ~39s x 2 x NtxNu
            check_FDderiv(wf,wf.irays,wf.dddy,k,verbose=False)
    
    if(False): # test combining derivatives matrices using chain rule across grid

        chainmatrix = np.ones_like(wf.pdf) # This choice represents taking derivative of the sum of the pdf amplitudes.
        deriv = wPDFderiv(wf.pdf,wf.dddy,lambdav,wf.irays,wf,chainmatrix) # calculate derivatives of function of PDF field wrt waveform amplitude
    
        wf.PDFderiv(chainmatrix=chainmatrix)

        check_FDchain(wf,wf.pdfd,lambdav)