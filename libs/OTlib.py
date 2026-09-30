# -*- coding: utf-8 -*-
#
# OTlib.py
#
# Library for solution of Optimal Transport problems using various algorithms.
#
# Can perform calculation of Wasserstein W1 and W2 metrics for arbitrary 1D PDFs
# as well as approximate Wasserstein distances for 2D PDFs using either a Marginal 
# or Sliced Wasserstein algorithm.
#
# Makes use of third party Linear programming and Monge-Ampere solvers and compares to analytically derived results
#
# M. Sambridge, 
# ANU June 2020.
#
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.optimize import lsq_linear
import bisect
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance_matrix 
from scipy.optimize import minimize

try:
    import ot as ot # import POT library
    noPOTlibrary = False
except ImportError:
    noPOTlibrary = True

class Error(Exception):
    """Base class for other exceptions"""
    pass

class PDFShapeError(Exception):
    """Raised when input PDF has inconsistent set of amplitdues and locations"""
    def __init__(self,msg=''):
        super().__init__('\n PDF amplitude and point location have different shapes \n')
        
class DistfuncShapeError(Exception):
    """Raised when input distance array has inconsistent shape compared to source and target PDFs"""
    def __init__(self,msg=''):
        super().__init__('\n Input distfunc is an array with wrong shape. First index should be source PDF dimension and second index target PDF dimension\n')

class PDFSignError(Exception):
    """Raised when input PDF has a component with a negative amplitude"""
    def __init__(self,msg=''):
        super().__init__('\n OTpdf Error: Input PDF has a negative amplitude\n')
    
class UnknownOTDistanceTypeError(Exception):
    """Raised when the unknown Wasserstein p value given"""
    def __init__(self,msg=''):
        super().__init__('\n Error in wasserPOT: Do not recognize parameter distfunc\n')
    
class TargetSourceCDFError(Exception):
    """Raised when the target and source CDFs have common entries"""
    def __init__(self,cset=[]):
        msg='\n Identical values in CDF of source and target detected \n\n Common set :'+str(cset) \
        +'\n\n For this case derivatives of Wasserstein distance with respect to corrresponding PDF amplitudes become discontinuous, and will not be comparable to Finite difference estimates\n'
        super().__init__(msg)

class TargetSource2DShapeError(Exception):
    """Raised when the target or source PDFs are not 2D"""
    def __init__(self,msg=''):
        super().__init__('\n  Input PDF is not 2D when it should be.\n')

class TargetSourceShapeError(Exception):
    """Raised when the target and source PDFs have different dimension"""
    def __init__(self,msg=''):
        super().__init__('\n  Input PDFs do not have same dimension, i.e. both 1D or both 2D. This is inconsistent.\n')

class Bary2Dcontinuousnotimpemented(Exception):
    """Raised when the Barycentral paths between 2D continuous PDFs requested"""
    def __init__(self,msg=''):
        super().__init__('\n  Barycentral paths between 2D continuous PDFs not yet implemented.\n'+
                              '  Only discrete pointmasses implemented (Try pointmass=True)\n')

class Bary2Dpointmassnotimpemented(Exception):
    """Raised when the Barycentral paths between 2D discrete PDFs requested and masses are not equal"""
    def __init__(self,msg=''):
        super().__init__('\n  Barycentral paths between 2D discrete PDFs with unequal PDF amplitudes not yet implemented.\n'+
                              '  (Try pointmass=False)\n')

class SlicedWassersteinError(Exception):
    """Raised when input parameters not valid in SlicedWasserstein routine"""
     
class MarginalWassersteinError(Exception):
    """Raised when input parameters not valid in MarginalWasserstein routine"""
    def __init__(self,mset=[]):
        msg='\n Marginal Wasserstein routine not set up to recognize distfunc:'+mset \
        +'\n \n'
        super().__init__(msg)

class SlicedWassersteinPointCloudError(Exception):
    """Raised when distance matrix has been supplied to SlicedWassersteinPointCloud with p value"""
    def __init__(self,cset=[]):
        msg='\n Distance matrix has been supplied to Sliced Wasserstein Point Cloud algorithm without p value :' \
        +'\n\n A p value must be supplied because derivatives of W with respect to source locations depend on p\n' \
        +'\n Remedy: replace ndarray for distance matrix A with tuple (p,A) as distfunc keyword \n'
        super().__init__(msg)

class POTlibraryError(Exception):
    """Raised when POT library is not installed"""
    def __init__(self,msg=''):
        super().__init__('\n POT library is not installed \n')

class OTpdf(object): # PDF object for OT library
    """
    Optimal transport PDF object. 
    
    A Class that is used by various routines to calculate Wasserstein distances 
    and transport plans between pairs of OTpdf objects.
    
    """
    def __init__(self,
                 pdf):
        if(np.min(pdf[0])< 0.0): raise PDFSignError()

        x = pdf[1]
        p = pdf[0]
        self.pointcloud = False
        self.upointcloud = False

        if(len(np.shape(pdf[1]))==2 and len(np.shape(pdf[0]))==1): # input is 2D point cloud
            self.type='2D'
            self.pointcloud = True
            self.ndim = 2
            self.n = len(p)
            self.mean = np.mean(x,axis=0)
            if(np.all(p == p[0])): # uniform point cloud with all equal amplitudes
               self.upointcloud = True
		
        elif(len(np.shape(pdf[0]))==2): # input PDF is 2D
            self.type='2D'
            self.ndim = 2
            if(np.shape(p) != np.shape(x)[:2]):
                raise PDFShapeError
            self.nx=np.shape(x)[0]
            self.ny=np.shape(p)[1]
            self.n = self.nx*self.ny
        else:                           # input PDF is 1D

            self.n = len(p)
            self.ndim = 1
            self.type = '1D'
            if(len(x) != len(p)):
                raise PDFShapeError
                    
        self.amp = np.sum(p)
        self.pdf = p/np.sum(p)
        self.x = x.copy()
        self.nproj = 0

        cdf = np.cumsum(self.pdf).copy()
        cdf /=cdf[-1]                    # avoid some special case rounding errors
        self.cdf = cdf
        self.calcproj = True # set to True if projection data not yet calculated 
        self.calcmarg = True # set to True if Marginals have not yet calculated 
        self.ProjNum = -1
        
    def setSliced(self,
                  Nproj,
                  org,
                  theta = None,
                  phi=45.0): # build projected PDF for Nproj angles and store within class
        """
            OTpdf class function. 
    
            Calculates sliced distributions from 2D point masses
            Input:
                Nproj - integer; Number of slices to be used (regularly spaced in angle about org)
                org - ndarray, shape(2); (x,y) origin about which to form slices.
                
        """
        if(self.type !='2D'): raise TargetSource2DShapeError
        self.nproj = Nproj
        self.origin = org
        self.slicedangle = phi
        wx,wy = getxyweights(phi)
        f = self.pdf.reshape((self.n)) # make into 1D PDFs
        if(theta is None):
            theta = np.linspace(0.1745,np.pi,Nproj+1) # set angles with some offset
            theta = theta[:-1]
        else:
            Nproj = len(theta)
        r = np.array([wx*np.cos(theta),wy*np.sin(theta)])
        a = self.x-org
        a = a.reshape((self.n,2))
        fxp = np.dot(a,r).T
        fxpargsort = np.argsort(fxp)
        s = [OTpdf((f[fxpargsort[i]],fxp[i][fxpargsort[i]])) for i in range(Nproj)] # set up source object for each projection
        self.proj = s
        self.angles = theta
        self.psorted = fxpargsort
        self.calcproj = False 

    def setMarginals(self): # build projected PDF for Nproj angles and store within class
        """
            OTpdf class function. 
    
            Calculates Marginal distributions from 2D PDF by integrating over x and y.
                
        """
        if(self.type !='2D'): raise TargetSource2DShapeError
        self.nproj = 2
        factor = 1.
        f0 = np.sum(factor*self.pdf,axis=0) # marginal PDF over first axis
        f1 = np.sum(factor*self.pdf,axis=1) # marginal PDF over second axis
        fx0 = self.x[0,:,0]
        fx1 = self.x[:,0,1]
        theta = np.array([0.0,np.pi/2.])
        m0 = OTpdf((f0,fx0)) # set up source object for first marginal
        m1 = OTpdf((f1,fx1)) # set up source object for second marginal
        s = [m0,m1] # set up source object for each projection
        self.marg = s
        self.angles = theta
        self.calcmarg = False 

def _checkdistfunc(distfunc): # checks type of distance function to be used for errors
    calcW1 = False
    calcW2 = False
    dfunc = False
    distfunction_args = False
    if(isinstance(distfunc, str)):
        if(distfunc=='W2' or distfunc=='W12'): 
            calcW2 = True
        if (distfunc=='W1' or distfunc=='W12'):  
            calcW1 = True
    elif(type(distfunc) is tuple):
       dfunc=True 
       distfunction_args = distfunc
       
    elif(type(distfunc) is np.ndarray): 
        dfunc=True 
        distfunction_args = distfunc
    else:
        raise UnknownOTDistanceTypeError
    
    return calcW1,calcW2,dfunc,distfunction_args

def _calc_distArray(source,
                    target,
                    distfunc=None,
                    args=None,
                    verbose=False):

    if(verbose): print(' inside _calc_distArray: distfunc=',distfunc)
    if(type(distfunc) is np.ndarray):     
        return distfunc.flatten(), None
    elif( type(distfunc) is tuple):
        p,A = distfunc
        return A.flatten(), None
    n = source.n
    if(source.type=='2D'):
        fx = source.x.reshape((n,2))
        gx = target.x.reshape((n,2))
    else:
        fx = source.x
        gx = target.x

    A_eq = np.zeros((2*n,n*n))
    d = np.ndarray((n,n))
    for j in range(0,n):   
        for i in range(0,n):
            l = fx[j]-gx[i] # or arbitrary located input data
            if(distfunc == 'W1' or distfunc == None ): # calculate W1 by default
                if(source.type=='2D'):
                    d[j,i] = abs(l[0])+abs(l[1]) # W-1 distance
                else:
                    d[j,i] = abs(l) # W-1 distance
            elif(distfunc == 'W2'):
                if(source.type=='2D'):
                    d[j,i] = l[0]**2+ l[1]**2 # W-2 distance
                else:
                    d[j,i] = (l)**2 # W-2 distance
            else:
                d[j,i] = distfunc(j,i,args)
            A_eq[j,j*n+i] = 1.0
            A_eq[n+j,i*n+j] = 1.0
    return d,A_eq

def _checkderiv(source,
                target,
                df,
                mapout=None,
                verbose=False,
                percent=False,
                ignoreCommonCDFerror=False,
                reflect=False): # compare 1D wasserstein analytical results to Finite Difference

    f = source.pdf*source.amp
    #g = target.pdf
    fx = source.x
    #gx = target.x
    if(mapout is None):
        mapped = np.arange(0,len(f))
    else:
        mapped = mapout

    w2,H = wasser(source,target,distfunc='W2',returnplan=True,reflect=reflect,ignoreCommonCDFerror=ignoreCommonCDFerror)
    #s,H = wasser_find_optplan(source,target,w2,distfunc='W2')
    #w20 = w2 * np.ones(len(f))
    w2m, w2p = np.zeros(len(f)), np.zeros(len(f))

    W1, dW1, dW1_pos, W2, dW2, dW2_pos, H, derivH = wasser(source,target,returnplan=True,derivatives=True,ignoreCommonCDFerror=ignoreCommonCDFerror,reflect=reflect)

    # Finite differences
    print('\n Compare analytical and finite difference Derivatives : \n')
    print('I           d(W2)/df            Finite Diff         Average difference in plan derivatives\n')
    dfused = df # use fixed perturbation
    dffloor = 0.0001*np.max(f)
    for i in range(len(dW2)):
        j = mapped[i]
        if(percent): dfused = np.abs(f[j])*df/100. # use relative perturbation
        if(np.abs(f[j]) > dffloor): # do not do FD if amplitude too small
            fmin = np.copy(f)
            fmin[j] = f[j] - dfused
            sm = OTpdf((fmin,fx)) # set up source object
            w2m[j],Hm = wasser(sm, target, distfunc='W2',returnplan=True,ignoreCommonCDFerror=ignoreCommonCDFerror,reflect=reflect)
            fplu = np.copy(f)
            fplu[j] = f[j] + dfused
            sp = OTpdf((fplu,fx)) # set up source object
            w2p[j],Hp = wasser(sp, target, distfunc='W2',returnplan=True,ignoreCommonCDFerror=ignoreCommonCDFerror,reflect=reflect)
            wfd = (w2p[j]-w2m[j])/(2*dfused)
            fd = (Hp-Hm)/(2*dfused)
            print(j, ' :      ', dW2[j], ' ', wfd, np.mean(fd-derivH[j]))
            if(verbose):
                print('\n Parameter :',j,'\n')
                print(' Finite Difference:' )
                print(fd)
                print(' Analytics:' )
                print(derivH[j])

    print('\nI           d(W1)/df            Finite Diff         Average difference in plan derivatives\n')
    for i in range(len(dW1)):
        j = mapped[i]
        if(percent): dfused = np.abs(f[j])*df/100. # use relative perturbation
        #if(i == ival):
        if(np.abs(f[j]) > dffloor): # do not do FD if amplitude too small
            fmin = np.copy(f)
            fmin[j] = f[j] - dfused
            sm = OTpdf((fmin,fx)) # set up source object
            w2m[j],Hm = wasser(sm, target, distfunc='W1',returnplan=True,ignoreCommonCDFerror=ignoreCommonCDFerror,reflect=reflect)
            fplu = np.copy(f)
            fplu[j] = f[j] + dfused
            sp = OTpdf((fplu,fx)) # set up source object
            w2p[j],Hp = wasser(sp, target, distfunc='W1',returnplan=True,ignoreCommonCDFerror=ignoreCommonCDFerror,reflect=reflect)
            wfd = (w2p[j]-w2m[j])/(2*dfused)
            fd = (Hp-Hm)/(2*dfused)
            print(j, ' :      ', dW1[j], ' ', wfd, np.mean(fd-derivH[j]))
            if(verbose):
                print('\n Parameter :',j,'\n')
                print(' Finite Difference:' )
                print(fd)
                print(' Analytics:' )
                print(derivH[j])

    print('\n Compare analytical and finite difference Derivatives for translation of source positions: \n')
    print('d(W2)/dx               Finite Diff           Average change in plan entries\n')
    dx = (np.max(fx)-np.min(fx))/100. # set perturbation to source PDF locations
    fminx = np.copy(fx) - dx
    sm = OTpdf((f,fminx)) # set up source object
    w2mp,Hmp = wasser(sm, target, distfunc='W2',returnplan=True,reflect=reflect)
    fplux = np.copy(fx) + dx
    sp = OTpdf((f,fplux)) # set up source object
    w2pp,Hpp = wasser(sp, target, distfunc='W2',returnplan=True,reflect=reflect)
    wfd = (w2pp-w2mp)/(2*dx)
    fd = (Hpp-Hmp)
    print(dW2_pos, ' ', wfd,' :',np.mean(fd))
            
def _checkderivSliced(source,
                      target,
                      df,
                      Nproj=10,
                      distfunc='W2',
                      verbose=False,
                      origin = [0.5,0.5],
                      memory=False,
                      phi=45.0,
                      reflect=False):

    if(source.upointcloud):
       p=2
       if(type(distfunc) is np.ndarray):
          print('\n Inconsistent inputs into checkderivSliced for Point Clouds:')
          print('               Checking derivatives for point cloud and distance matrix has been provided')
          print('               Legal uses are distfunc = W1, W2 of (p,A)')
          print('               Using p=2')
       elif(type(distfunc) is tuple):
          p,A = distfunc
       elif(distfunc == 'W1'):p=1
       return _checkderivSlicedPointCloud(source,target,Nproj,p=p,origin=origin,phi=phi) # finite difference estimator of dwdx,dwdy for pointclouds
        
    f = source.pdf.reshape(source.n)*source.amp # get unnormalized PDF amplitudes
    fx = source.x

    Wplan, dWplan = SlicedWasserstein(source,target,Nproj,derivatives=True,distfunc=distfunc,memory=memory,reflect=reflect,phi=phi)

    # Finite differences
    print('\n W2 from average plan: ',np.sqrt(Wplan))
    print('\n Compare analytical and finite difference derivatives from Sliced Wasserstein: \n')
    print('I           d(W2)/df            Finite Diff \n')
    for i in range(len(dWplan)):
        fmin = np.copy(f)
        fmin[i] = f[i] - df
        sm = OTpdf((fmin.reshape((source.nx,source.ny)),fx)) # set up source object
        w2m = SlicedWasserstein(sm, target, Nproj, distfunc=distfunc,memory=memory,reflect=reflect,phi=phi)[0]
        fplu = np.copy(f)
        fplu[i] = f[i] + df
        sp = OTpdf((fplu.reshape((source.nx,source.ny)),fx)) # set up source object
        w2p = SlicedWasserstein(sp, target, Nproj, distfunc=distfunc,memory=memory,reflect=reflect,phi=phi)[0]
        #print(w2p,w2m)
        wfd = (w2p-w2m)/(2*df)
        #wavgfd = (w2avgp-w2avgm)/(2*df)
        #fd = (Hp-Hm)/(2*df)
        #print(i, ' :    avg   ', dWavg.flatten()[i], ' ', wavgfd, np.mean(fd-derivH[i]))
        print(i, ' :    plan  ', dWplan.flatten()[i], ' ', wfd)
    return 

def getxyweights(phi):
    wx, wy = 1.,1.
    if(phi != 45.0):
        wx = np.sqrt(2)*np.cos(phi*np.pi/180.)
        wy = np.sqrt(2)*np.sin(phi*np.pi/180.)
    return wx,wy

def _checkderivSlicedPointCloud(s,t,Nproj,p=2,origin=[0.5,0.5],theta=None,phi=45.0): # finite difference estimator of dwdx,dwdy for pointclouds
            
    metric = 'sqeuclidean'
    if(p==1): metric = 'cityblock'
    #A = pairwise_distances(s.x, t.x, metric=metric) # use Euclidean distance matrix to power p
    wx,wy = getxyweights(phi)
    A = pairwise_distances(s.x*np.array([wx,wy]), t.x*np.array([wx,wy]), metric=metric) # use Euclidean distance matrix to power p

    Wplan,dWx,dWy,dWpos,H = SlicedWassersteinPointCloud(s,t,Nproj,A,p=p,derivatives=True,returnplan=True,returnderivwindow=True,phi=phi)

    dAdxFDs = np.zeros(len(s.x))
    dAdyFDs = np.zeros(len(s.x))
    
    # get minimum projected separation of point cloud so that FD estimnates do not change order of projected points
    xs = s.x-origin
    xt = t.x-origin
    if(theta is None):
        theta = np.linspace(0.1745,np.pi,Nproj+1) # set angles with some offset
        theta = theta[:-1]
    else:
        Nproj = len(theta)
    r = np.array([wx*np.cos(theta),wy*np.sin(theta)])
    sproj_sorted = np.sort(np.dot(xs,r),axis=0).T
    dm=np.zeros(Nproj)
    for i in range(Nproj):
        dm[i] = np.min(np.abs(sproj_sorted[i][1:]-sproj_sorted[i][:-1])) # minimum projected separations
    
    dx = np.min(dm)
    if(s.ndim==2):
        print('\nFinite Difference Derivatives with respect to point mass locations:\n'   )
        print('        Analytical           FD (dW2dx)        Delta H mean')
        for i in range(len(s.x)):
            fp = np.copy(s.x)
            #dx=np.abs(fp[i,0])*dxp
            fp[i,0]+=dx
            sp = OTpdf((s.pdf,fp))
            #Ap = pairwise_distances(sp.x, t.x, metric=metric) # use Euclidean distance matrix to power p
            Ap = pairwise_distances(sp.x*np.array([wx,wy]), t.x*np.array([wx,wy]), metric=metric) # use Euclidean distance matrix to power p
            outp = SlicedWassersteinPointCloud(sp,t,Nproj,Ap,returnplan=True,phi=phi)
            Hp,Wp2s = outp[1],outp[0]
            #Wp2 = np.sum(np.multiply(H,Ap))
            fm = np.copy(s.x)
            fm[i,0]-=dx
            sm = OTpdf((s.pdf,fm))
            #Am = pairwise_distances(sm.x, t.x, metric=metric) # use Euclidean distance matrix to power p
            Am = pairwise_distances(sm.x*np.array([wx,wy]), t.x*np.array([wx,wy]), metric=metric) # use Euclidean distance matrix to power p
            outm = SlicedWassersteinPointCloud(sm,t,Nproj,Am,returnplan=True,phi=phi)
            Hm,Wm2s = outm[1],outm[0]
            #Wm2 = np.sum(np.multiply(H,Am))
            #dAdxFD[i] = (Wp2-Wm2)/(2*dx)
            dAdxFDs[i] = (Wp2s-Wm2s)/(2*dx)
            print(i,dWx[i],dAdxFDs[i], np.mean((np.abs(Hp-H),np.abs(Hm-H))))
        print('\n Mean absolute difference in FD and analtical derivatives for x:',np.mean(np.abs(dWx-dAdxFDs)))  
        print('\n        Analytical           FD (dW2dy)        Delta H mean')

        for i in range(len(s.x)):
            fp = np.copy(s.x)
            #dx=np.abs(fp[i,1])*dxp
            fp[i,1]+=dx
            sp = OTpdf((s.pdf,fp))
            #Ap = pairwise_distances(sp.x, t.x, metric=metric) # use Euclidean distance matrix to power p
            Ap = pairwise_distances(sp.x*np.array([wx,wy]), t.x*np.array([wx,wy]), metric=metric) # use Euclidean distance matrix to power p
            outp = SlicedWassersteinPointCloud(sp,t,Nproj,Ap,returnplan=True,phi=phi)
            Hp,Wp2s = outp[1],outp[0]
            #Wp2 = np.sum(np.multiply(H,Ap))
            fm = np.copy(s.x)
            fm[i,1]-=dx
            sm = OTpdf((s.pdf,fm))
            #Am = pairwise_distances(sm.x, t.x, metric=metric) # use Euclidean distance matrix to power p
            Am = pairwise_distances(sm.x*np.array([wx,wy]), t.x*np.array([wx,wy]), metric=metric) # use Euclidean distance matrix to power p
            outm = SlicedWassersteinPointCloud(sm,t,Nproj,Am,returnplan=True,phi=phi)
            Hm,Wm2s = outm[1],outm[0]
            #Wm2 = np.sum(np.multiply(H,Am))
            #dAdyFD[i] = (Wp2-Wm2)/(2*dx)
            dAdyFDs[i] = (Wp2s-Wm2s)/(2*dx)
            print(i,dWy[i],dAdyFDs[i], np.mean((np.abs(Hp-H),np.abs(Hm-H))))
        print('\n Mean absolute difference in FD and analtical derivatives for y:',np.mean(np.abs(dWy-dAdyFDs)))  

        print('\n Compare analytical and finite difference Derivatives for translation of source positions: \n')
        print('d(W2)/dx               Finite Diff           Average change in plan entries\n')
        fminx = np.copy(s.x) 
        fminx[:,0] -= dx
        sm = OTpdf((s.pdf,fminx)) # set up source object
        #Am = pairwise_distances(sm.x, t.x, metric=metric) # use Euclidean distance matrix to power p
        Am = pairwise_distances(sm.x*np.array([wx,wy]), t.x*np.array([wx,wy]), metric=metric) # use Euclidean distance matrix to power p
        #w2mp,Hmp = wasser(sm, t, distfunc='W2',returnplan=True,reflect=reflect)
        outm = SlicedWassersteinPointCloud(sp,t,Nproj,Am,returnplan=True,phi=phi)
        Hmp,w2mp = outm[1],outm[0]
        fplux = np.copy(s.x)
        fplux[:,0] += dx
        sp = OTpdf((s.pdf,fplux)) # set up source object
        #Ap = pairwise_distances(sp.x, t.x, metric=metric) # use Euclidean distance matrix to power p
        Ap = pairwise_distances(sp.x*np.array([wx,wy]), t.x*np.array([wx,wy]), metric=metric) # use Euclidean distance matrix to power p
        #w2pp,Hpp = wasser(sp, t, distfunc='W2',returnplan=True,reflect=reflect)
        outp = SlicedWassersteinPointCloud(sp,t,Nproj,Ap,returnplan=True,phi=phi)
        Hpp,w2pp = outp[1],outp[0]
        wfd = (w2pp-w2mp)/(2*dx)
        fd = (Hpp-Hmp)
        dW2_pos = 0.
        print(dWpos, ' ', wfd,' :',np.mean(fd))

    return 

def _checkderivMarg(source,
                    target,
                    df,
                    distfunc='W2',
                    verbose=False,
                    memory=False,
                    percent=False,
                    ind=None,
                    returnmargW=False,
                    reflectX=False,
                    reflectY=False,
                    dffloor=None):

    f = source.pdf.reshape(source.n)*source.amp # get unnormalized PDF amplitudes
    fx = source.x

    # This routine compares finite difference derivatives to analytical for derivative  
    # of Wasserstein distance wrt to unormalised amplitude of 2D PDF

    Wpm, dWm, dWm_window = MargWasserstein(source,target,derivatives=True,distfunc=distfunc,memory=memory,returnmargW=returnmargW,reflectX=reflectX,reflectY=reflectY)

    # Finite differences
    if(verbose):
        print('\n W from average marginal : ',np.sqrt(Wpm))
        print('\n Compare analytical and finite difference derivatives from Marginal Wasserstein: \n')
        print('I                     d(W)/df            Finite Diff \n')
    dfused = df
    if(dffloor is None): dffloor = 0.0001*np.max(f)
    if (ind is None):
        setofindices = range(source.n)
        #setofindices = range(len(dWm))
    else:
        setofindices = ind
        
    #print(' setofindices ',setofindices,range(source.n))
    donenowork = True
    if(returnmargW):  # perform finite difference comparison separately for each marginal
        for i in setofindices:
            if(percent): dfused = np.abs(f[i])*df/100.
            if(np.abs(f[i]) > dffloor): # do not do FD if amplitude too small
                donenowork = False
                fmin = np.copy(f)
                fmin[i] = f[i] - dfused
                sm = OTpdf((fmin.reshape((source.nx,source.ny)),fx)) # set up source object
                w2m = MargWasserstein(sm, target, distfunc=distfunc,memory=memory,returnmargW=True,reflectX=reflectX,reflectY=reflectY)[0]
                fplu = np.copy(f)
                fplu[i] = f[i] + dfused
                sp = OTpdf((fplu.reshape((source.nx,source.ny)),fx)) # set up source object
                w2p = MargWasserstein(sp, target, distfunc=distfunc,memory=memory,returnmargW=True,reflectX=reflectX,reflectY=reflectY)[0]
                #print(w2p,w2m)
                wfd0 = (w2p[0]-w2m[0])/(2*dfused)
                wfd1 = (w2p[1]-w2m[1])/(2*dfused)
                if(verbose):
                    print(i, ' :     Marg t   ', dWm[0].flatten()[i], ' ', wfd0)
                    print(i, ' :     Marg u   ', dWm[1].flatten()[i], ' ', wfd1)
        return wfd0,wfd1

        
    else: # perform finite difference comparison for sum of Wp distances over marginals

        for i in setofindices:
            if(percent): dfused = np.abs(f[i])*df/100.
            if(np.abs(f[i]) > dffloor): # do not do FD if amplitude too small
                donenowork = False
                fmin = np.copy(f)
                fmin[i] = f[i] - dfused
                sm = OTpdf((fmin.reshape((source.nx,source.ny)),fx)) # set up source object
                w2m = MargWasserstein(sm, target, distfunc=distfunc,memory=memory,reflectX=reflectX,reflectY=reflectY)[0]
                fplu = np.copy(f)
                fplu[i] = f[i] + dfused
                sp = OTpdf((fplu.reshape((source.nx,source.ny)),fx)) # set up source object
                w2p = MargWasserstein(sp, target, distfunc=distfunc,memory=memory,reflectX=reflectX,reflectY=reflectY)[0]
                #print(w2p,w2m)
                wfd = (w2p-w2m)/(2*dfused)
                if(verbose):
                    print(i, ' :     avg   ', dWm.flatten()[i], ' ', wfd)
                    #print(i, ' :     avg   ', dWm[i], ' ', wfd)
        return wfd
    if(donenowork): return None,None
            
def _normalise(
               source, 
               target): # Mike Snow's OT routines for discrete L2

    return np.divide(source, np.sum(source))
    
def _optimaltransport(sourcein, 
                      targetin): # Mike Snow's OT routines for discrete L2

    """
    Computes the 1D optimal transport distance (W_2^2) between the source and target in domain x in [0, 1]. 
    Gives a metric to compare signals/distributions. Extends to compare points clouds in bins of interest in the signal.

    Assumes signals as numpy arrays. by Mike Snow
    
    https://github.com/mike-snow/1D-optimal-transport
    
    """    

    # normalise densities to have equal sum. Integers for ease.
    
    source = sourcein.pdf
    target = targetin.pdf
    
    if len(source) == 0:
        mapping, cost = np.zeros((1,1)), 0.000001
        return mapping, cost
    
    source, target = np.array(source), np.array(target)
    f_x, g_y = _normalise(source, target), _normalise(target, source)
    
    if len(f_x) == 1:
        m, n = 100000000, len(g_y)
    else:        
        m, n = len(f_x), len(g_y)
       
    c, i, j = 0, 0, 0
    
    mapping = np.zeros((m, n)) # Can create heatmap to visualise mapping. Only for small m, n! Or use sparse matrix

    while i < m and j < n:
        if g_y[j] == 0: 
            j += 1
        elif f_x[i] == 0: # if supply/demand if empty, skip. 
            i += 1
        else:
            if f_x[i] - g_y[j] > 0:
                f_x[i] -= g_y[j]
                c += (i/(m-1) - j/(n-1)) ** 2 * g_y[j] # density * cost to transport
                mapping[i,j] = g_y[j]
                j += 1
            elif f_x[i] - g_y[j] < 0:
                g_y[j] -= f_x[i]
                c += (i/(m-1) - j/(n-1)) ** 2 * f_x[i] # density * cost to transport
                mapping[i,j] = f_x[i]
                i += 1
            else: 
                c += (i/(m-1) - j/(n-1)) ** 2 * f_x[i] # density * cost to transport
                mapping[i,j] = f_x[i]
                i += 1                
                j += 1
    
    return mapping, c

def BuildLinProg(source,
                 target,
                 distfunc=None,
                 args=None):
#
# Build linear system for LP problem
#
# General version that calls a user supplied distance function
#
    if( distfunc == None ): raise UnknownOTDistanceTypeError
    b_eq_ab = np.array(list(source.pdf)+list(target.pdf))
    d,A_eq = _calc_distArray(source,target,distfunc=distfunc,args=args)
    return d,A_eq,b_eq_ab

def Wasser_LinProg(source,
                   target,
                   distfunc=None,
                   args=None,
                   silent=False,
                   maxiter=555,
                   mth="interior-point"):
#
# Computing the 1-Wasserstein or 2-Wasserstein distances from f to g using linear programming
#
# General version that calls a user supplied distance function
#
# This code is based on that by Andreas Bærentzen available at http://www2.compute.dtu.dk/~janba/w2.html
#
    if( distfunc is None ):
        raise UnknownOTDistanceTypeError
    if(source.type=='2D'):
        f = source.pdf
        f = f.reshape((source.n))
        g = target.pdf
        g = g.reshape((source.n))
    else:
        f = source.pdf
        g = target.pdf  
    fn = list(f)
    gn = list(g)
    n = len(fn)
    a = fn
    b = gn
    b_eq_ab = np.array(a+b)
    WLin = None
    #Tf2g = None        
    H = None
    d,A_eq = _calc_distArray(source,target,distfunc=distfunc,args=args)
    
    c = np.reshape(d,n*n) 
    output = linprog(c,A_eq=A_eq[:2*n-1], b_eq=b_eq_ab[:2*n-1],options={'maxiter':maxiter},method=mth)
    if(output.success):
        WLin = float(c.dot(output.x))
        H = (output.x).reshape(n,n)
        #Tf2g = np.dot(H.T,np.diag(1/f)) # transform matrix from source to target
    else:
        if(not silent):
            print('\n Problem with Linear solver for W \n')
            print(output)

    #return WLin,Tf2g,H # return W^p, transform matrix from source to target, Transport plan for source to target 
    return WLin,H # return W^p, Transport plan for source to target 
    
def plotWasser(xp,
               Fp,
               Gp,
               t,
               IF,
               IG,
               x,
               IGF,
               xmIFGsq,
               iFGdiff,
               filename='Null'):

    plt.figure(figsize=(9,10))

    plt.subplot(3, 2, 1)
    plt.plot(xp,Fp,'r',label='$F(x)$')
    plt.plot(xp,Gp,'g',label='$G(x)$')
    plt.title('CDFs')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend(loc=2)
    plt.ylim(-0.1,1.1)
    x0,x1 = plt.xlim()

    plt.subplot(3, 2, 2)
    #plt.plot(Fp,xp,'r',label='$F^{-1}(x)$') # Inverse CDFs from switching plots axes
    #plt.plot(Gp,xp,'g',label='$G^{-1}(x)$') # Inverse CDFs from switching plots axes
    plt.plot(t,IF,'r',label='$F^{-1}(t)$')   # Inverse CDFs from formulae
    plt.plot(t,IG,'g',label='$G^{-1}(t)$')
    plt.title('Inverse CDFs')
    #plt.title('Inverse CDFs from formula')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.xlim(-0.1,1.1)
    plt.ylim(x0,x1)
    plt.legend(loc=2)

    plt.subplot(3, 2, 5)
    plt.plot(x,IGF,'b',label='$G^{-1}(F(x))$')
    plt.plot(x,x,'k:')
    #plt.title('x and $G^{-1}(F(x))$')
    plt.xlabel('x')
    #plt.ylabel('$x^{\prime}$')
    plt.ylabel('T(x)')
    plt.xlim(x0,x1)
    plt.ylim(x0,x1)
    plt.legend(loc=4)

    plt.subplot(3, 2, 6)
    plt.plot(x,xmIFGsq,label='$(x - G^{-1}(F(x)))^2$')
    #plt.title('Squared difference $(x - G^{-1}(F(x)))^2$')
    plt.xlabel('x')
    plt.ylabel('$(x - G^{-1}(F(x)))^2$')
    plt.xlim(x0,x1)
    #plt.legend(loc=4)

    plt.subplot(3, 2, 3)
    plt.plot(t,np.abs(iFGdiff),'k',label='$(G^{-1} - F^{-1})$')   # Inverse CDFs from formulae
    plt.xlabel('t')
    plt.ylabel('$(G^{-1} - F^{-1})$')
    plt.xlim(-0.1,1.1)

    plt.subplot(3, 2, 4)
    iFGdiffsq = np.multiply(np.abs(iFGdiff),np.abs(iFGdiff))
    plt.plot(t,iFGdiffsq,'m',label='$(G^{-1} - F^{-1})^2$')   # Inverse CDFs from formulae
    plt.xlabel('t')
    plt.ylabel('$(G^{-1} - F^{-1})^2$')
    plt.xlim(-0.1,1.1)
    
    if(filename!='Null'):
        plt.savefig(filename)

    plt.tight_layout()
    plt.show()
    return

def distfunction(iarr,
                 jarr,
                 distfunction_args,
                 proj=-1,
                 deriv=False): # distance between all PDF source and target points is precomputed in A

    if(type(distfunction_args)is np.ndarray):
        A = distfunction_args
    else:
        source,target,p,A = distfunction_args
        
    if(proj==-1): # option suitable for 1D input source and target PDFs
        dxft = A[iarr,jarr] # use this for 1D distance**p between points
        
    else: # option suitable for use by sliced Wasserstein algorithm
    
        fxpargsort = source.psorted[proj]
        gxpargsort = target.psorted[proj]
        lf = fxpargsort[iarr]
        lg = gxpargsort[jarr]
        dxft = A[lf,lg] # use this for 2D distance**p between points
            
    if(deriv): # derivatives not implemented and can't be unless we know the p value used to calculate A. 
        return dxft
    else:
        return dxft # looks like this is not completely implemented

def wasser(source,
           target,
           distfunc='W12',
           proj=-1,
           returnplan=False,
           derivatives=False,
           memory=False,
           checkCommonCDF=False,
           ignoreCommonCDFerror=False,
           reflect=False):
    """

    Calculates pth power of Wasserstein metric W_p^p(f,g) 
    and optionally its derivatives with respect to unormalised source amplitudes.

    Works for p=1 and 2 for 1-D PDFs of arbitrary length and location

    Derivatives of (p-Wasserstein distance)**p are calculated with respect to the UNORMALISED elements of f.

    Note: the Wasserstein distance is defined as 1/p pth power of the output, i.e. for p=2 we return the square of the Waserstein distance.
    (Note they are equivalent for the p=1 case.)

    Derivatives are only accurate when there are no identical values in source and target CDFs. 
    Typically this is achieved by adding a perturbation to either source or target PDF values. 
    This condition can be checked with checkCommonCDF=True.

    Parameters: 
        
        source  (PDF object): 1D or 2D PDF object used as source for Optimal Transport (created with OT.PDF)
        target  (PDF object): 1D or 2D PDF object used as target for Optimal Transport (created with OT.PDF)
    
    Optional parameters:
    
        distfunc                : 'W1','W2' or 'W12' (default) or user supplied array of distances.
        returnplan              : Calculate optimal plan (default is False).
        derivatives             : Calculate derivatives of W and plan (default False).
        memory                  : set to True to save large memory matrices and replace python matrix sums with for loops. (default False)
        checkCommonCDF          : Checks input PDFs for condition to cause derivatives to fail (default False).
        ignoreCommonCDFerror    : Does not raise an error in derivative failure condition is found (default False).
    
    Returns:
        
        W1, dW1/ds, dW1/dt W2, dW2/ds, dW2/dt, H, dH/ds : Arrangement depends upon input options.
        
        W1     = Wasserstein distance (p=1)
        dW1/ds = Derivative of W_1 w.r.t. unormalised amplitudes of input source PDF (source.pdf)
        dW1/dt = Derivative of W_1 w.r.t. translation of source PDF (i.e. source.x = x_i + t) 
        W2     = Wasserstein distance (p=2)
        dW2/ds = Derivative of W_2 w.r.t. unormalised amplitudes of input source PDF (source.pdf)
        dW2/dt = Derivative of W_2 w.r.t. translation of source PDF (i.e. source.x = x_i + t) 
        H      = Transport plan
        dH/ds  = Derivative of H w.r.t. unormalised amplitudes of input source PDF (source.pdf)
    
    """

    calcW1,calcW2,dfunc,distfunction_args = _checkdistfunc(distfunc)

    #print(calcW1,calcW2,dfunc,returnplan,derivatives,checkCommonCDF)

    if(reflect): # create symmetric PDFs by reflection about upper limit 
    	cf,cfamp,nf,xf  = reflectPDF(source)
    	cg,cgamp,ng,xg  = reflectPDF(target)
    	#print('reflectT',nf,ng)
    else:
    	cf = source.cdf
    	cg = target.cdf
    	cfamp = source.amp
    	cgamp = target.amp
    	nf = source.n    
    	ng = target.n
    	xf = source.x
    	xg = target.x
    
    if(dfunc):
        if(type(distfunction_args) is tuple):
            dum0,dum1,p,Am=distfunction_args 
            #print('Am',Am)
        else:
            Am = distfunction_args
        #print('Am',type(Am),Am)
        if(nf != np.shape(Am)[0] or ng != np.shape(Am)[1]):
            print('source.n',nf,np.shape(Am)[0])
            print('target.n',ng,np.shape(Am)[1])
            if(reflect):
               print('Wasser routine reflect mode is on and a distance matrix has been supplied in call sequence: ',
                     '\nThis combination of features requires input distance matrix consistent with relfected PDFs')
            raise DistfuncShapeError
            
    if(derivatives or checkCommonCDF):
        cset = np.intersect1d(cg[:-1],cf[:-1])
        if(reflect):
            if(len(cset) >1):  # reflect mode always creates a single common CDF value at 0.5 so we ignore this
               if( not ignoreCommonCDFerror): raise TargetSourceCDFError(cset)
        else:
            if(len(cset) !=0): # If there are common CDF values report this unless we are told not to
               if( not ignoreCommonCDFerror): raise TargetSourceCDFError(cset)

    a = np.append(cf[:-1],cg)
    tkarg = np.argsort(a)
    tk = a[tkarg]
    indf = list(map(lambda x:bisect.bisect_left(cf,x) ,tk))
    indg = list(map(lambda x:bisect.bisect_left(cg,x) ,tk))
    dtk = np.insert(tk[1:] - tk[:-1],0,tk[0])  # A*tk_deriv
   
    if(calcW1 or calcW2):
        xft = xf[indf]
        xgt = xg[indg]
        dxft = np.abs(xft-xgt)
        #print(' shape of indf:',np.shape(indf))
        #print(' shape of indg:',np.shape(indg))
        #print(' shape of xf:',np.shape(xf))
        #print(' shape of xft:',np.shape(xft))
        #print(' shape of xg:',np.shape(xg))
        #print(' shape of xgt:',np.shape(xgt))

# Calculations required for derivatives
    if(derivatives):
        B = np.triu(np.ones((nf,ng)))
        C = (B-cf)/cfamp
        D = np.hstack((C[:,:-1],np.zeros((nf,ng))))
        Difftk = D[:,tkarg]
        Diffdtk = np.hstack((Difftk[:,0:1],Difftk[:,1:]-Difftk[:,:-1]))

    out = []
    if(calcW1): # Calculate W1 and its derivative with respect to unnornalised source amplitudes
        #print(' shape of dxft:',np.shape(dxft))
        #print(' shape of dtk:',np.shape(dtk))
        W1 = np.dot(dxft,dtk)
        out += [W1]
        if(derivatives): 
            dxftdg = np.sign(xft-xgt) # derivative of distances with respect to translation of input PDF position.
            dW1 = np.dot(Diffdtk,dxft)
            if(reflect):
                ii = int(len(dW1)/2)
                dW1 = (dW1[:ii]+dW1[-1:ii-1:-1])
            out += [dW1]
            out += [np.dot(dxftdg,dtk)] # add in derivative of W1 wrt to time window position
        
    if(calcW2): # Calculate W2^2 and its derivative with respect to unnornalised source amplitudes
        dsqxft = np.multiply(dxft,dxft)
        W2 = np.dot(dsqxft,dtk)
        out += [W2]
        if(derivatives):
           dsqxftdg = 2.0*(xft-xgt) # derivative of distances with respect to translation of input PDF position.
           dW2 = np.dot(Diffdtk,dsqxft)
           if(reflect):
               ii = int(len(dW2)/2)
               dW2 = (dW2[:ii]+dW2[-1:ii-1:-1])
           out += [dW2]
           out += [np.dot(dsqxftdg,dtk)] # add in derivative of W2 wrt to time window position

    if(dfunc): # user supplied function provides distances between elements of source and target
    # NB derivatives with respect to grid position not currently implemented for dfunc option hence return zero derivative for now
        dxft = distfunction(indf,indg,distfunction_args,proj=proj)
        Wf = np.dot(dxft,dtk)
        out += [Wf]
        if(derivatives): 
            dWf = np.dot(Diffdtk,dxft) 
            out += [dWf]
            out +=[ 0.0] # distfunction would need to be updated to get this to work because derivative form depends on value of p, which is unspecified for this option.

    if(returnplan):  #Calculate Optimal plan and its derivative with respect to unnornalised source amplitudes
        m = len(dtk)
        if(memory): # This option saves allocation of large memory array at increase in execution cost. But can also save computation as experience shows.
            H = np.zeros((nf,ng))
            for i, v in enumerate(dtk):
                H[indf[i],indg[i]] +=v
        else:
            H = np.zeros((nf,ng,m))
            H[indf,indg,np.arange(m)] = dtk
            H = H.sum(2)  # This statement is the bottle neck and very time consuming
        if(reflect): # if we are in reflect mode then only return upper (source.n X target.n) matrix of H
           H = H[:source.n,:target.n]
        out += [H]

        if(derivatives):
            if(memory): # This snippet saves allocation of large memory array at increase in execution cost
                dH = np.zeros((nf,nf,ng))
                for j in range(m):
                    dH[:,indf[j],indg[j]] += Diffdtk[:,j]
            else:
                DerivH = np.zeros((nf,nf,ng,m))
                DerivH[:,indf,indg,np.arange(m)] = Diffdtk
                dH = DerivH.sum(3)
            if(reflect): # if we are in reflect mode then only return upper (source.n X target.n) matrix of H
                dH = dH[:source.n,:source.n,:target.n]
            out += [dH]
    return out

def reflectPDF(source): # makes PDF symmetric by reflection in upper limit and updates various quantities 
    x = source.x
    pdf = source.pdf
    ddx = x[1]-x[0]
    dx = x[-1]-x[0]
    #xf = np.linspace(x[0],x[-1]+dx+ddx,2*len(x)) 
    xf = np.concatenate((x,x+dx+ddx))
    pdf = source.amp*np.concatenate((pdf,pdf[::-1]))
    cfamp = np.sum(pdf)
    pdf = pdf/np.sum(pdf)
    cf = np.cumsum(pdf).copy()
    cf /=cf[-1]                    # avoid some special case rounding errors
    nf = len(xf)
    return cf,cfamp,nf,xf  

def barypath(source,
             target,
             weights,
             npoints=50000,
             returntaxis=False,
             pointmass=False,
             sliced = False,
             Nproj=10,
             verbose=False,
             sinkhorn=False,
             sinkhornreg = 0.004,
             origin=[0.5,0.5],
             phi=45.0):

    """

    Calculates Barycentral path for the Wasserstein metric W_p^p(f,g) for 1D or 2D PDFs

    Works for p=1 and 2 for 1-D PDFs of arbitrary length and location
    Works for cases where PDFs are expressed as a discretization of a continuous function (1D only), 
    or series of 1D or 2D discrete point masses.

    Parameters: 
        
        source  (PDF object): PDF object used as source for Optimal Transport (created with OT.PDF)
        target  (PDF object): PDF object used as target for Optimal Transport (created with OT.PDF)
        weights  list: set of weights (0,1) defining each point along the Barycentral path between f and g.
        pointmass bool: True if input PDF is a sum of dirac delta functions.
        sliced    bool: True is Sliced Wasserstein calculation if barycentral paths is to be used (only if PDFs are 2D). (Option under development)
        Nproj    scalar: Number of projections used in SlicedWasserstein. Input value only used if `.setSliced()' class function not already called during SlicedWasserstein calculation.
        origin   ndarray or tuple; origin of Sliced Wasserstein axes used for rotation. 
        phi      scalar; angle in degrees of weighting defining distance metric (w_x = np.sqrt(2)*cost(phi), w_y = np.sqrt(2)*sin(phi))
        sinkhornreg float: Entropic regularization weight used by sinkhown algorithm (For 2D continuous PDFs only)
    
    Returns:
        
        x : nd-array (nweights,2,npoints) (x,y) locations of Barycentral path for each weight and each pointmass.

    Notes:
        For 1D source and target PDFs:
            If pointmass is true then the special case of transport of discrete point masses is used (i.e. no interpolation of CDFs). 
            In this case npoints has maximum value = source.n + target.n - 1, which is the number of discrete stair elements of the union of the CDFs.
            (Zero amplitude intermediate point masses are removed.)
            Note that if PDF ampitudes are all equal then no mass splitting occurs, otherwise it does occur and the routine works for both cases.  
            Splitting of discrete masses results in at most npoints intermediate point masses, but likely fewer.
        
            If pointmass is False PDFs are assumed continuous. In this case PDFs at intermediate points along
            the baycentral paths between source and target are found by interpolating the CDFs at equal spacings along the cummulative 
            index t (0 <= t <= 1). This is exact to the level of discretization.

        For 2D source and target PDFs:
            If pointmass is True,
            
            If pointmass is False, then they same approach as for continuous 1D PDFs is used for each slice projection.
            In this case inverse PDFs are calculated by an inverse Radon transform of projections. Under development.
            
    """

    if(source.ndim != target.ndim): 
        raise TargetSourceShapeError
    elif (source.ndim == 1): # input is 1D so call 1D code
        return barypath1D(source,target,weights,npoints=npoints,returntaxis=returntaxis,pointmass=pointmass) 

    # input is 2D
    
    cf = source.cdf
    cg = target.cdf
    fx = source.x
    gx = target.x
    
    if(source.calcproj or source.nproj !=Nproj): # projections have not previously been calculated and so we do so here
        source.setSliced(Nproj,origin,phi=phi)   # calculate source projection PDFs
    if(target.calcproj or target.nproj !=Nproj):
        target.setSliced(Nproj,origin,phi=phi)   # calculate target projection PDFs
        
    Nproj = source.nproj

    if(pointmass): # treat input as a set of 2D point masses and use Sliced Wasserstein to transport masses
        
        if(False): # This is development code and not functional
        
            # calculate barycentral points along each of the slices for each weight
            intpdf = []
            for j in range(Nproj):
                intpdf.append(barypath(source.proj[j],target.proj[j],weights,pointmass=True)) # use OT library to calculate barycentral distributions
            intpdf = np.transpose(np.array(intpdf), (1, 0, 2, 3))    

            #  Definition of `intpdf`:
            #  First index is the array of barycentral positions, or weights, between 0 and 1. There are `len(weights)` of these.
            #  Second index correspond to slice projections
            #  Third index are the `(x,pdf)` pair of each intermediate point masses, where x is the location on the projected axis.
            #  Fourth index is the number of intermediate point masses along each slice axis. These have max length `source.n+target.n-1`. But some may have zero amplitude and so are removed.
    
            uS,uT = np.zeros((Nproj,source.n)),np.zeros((Nproj,target.n))
            invsortS = np.argsort(source.psorted) # Each set of slice points are ordered by local location along projection axis, so we need to undo to mix points correctly
            invsortT = np.argsort(target.psorted) # Same for target slice projections
            xs,xt,x = [],[],[]
            for j in range(len(weights)): # loop over weights (=barycentral path indices)
                for i in range(Nproj):    # loop over slices
                    uS[i] = intpdf[j,i,0][invsortS[i]]
                    uT[i] = intpdf[j,i,0][invsortT[i]]
                xs.append(reconstruct_2Dpointmasses_from_SlicedProjections(uS,source.angles,source.origin)) # find least squares best fot 2D reconstructed points from slice projections
                xt.append(reconstruct_2Dpointmasses_from_SlicedProjections(uT,target.angles,target.origin))
            for j in range(len(weights)):
                x.append(weights[j]*xt[-j] + (1-weights[j])*xs[j]) # take weighted average of source reconstructions and target reconstructions to achive consistent 2D path.
                
        else: # 
            
            if(len(set(source.pdf))==1): # all pointmasses have equal weight and so we can calculate the full transport plan 
                                         # to identify the mapping between masses and determine linear barycentral paths
                w2pot,G0 = wasserPOT(source,target,returnplan=True,maxiters=300000)
                Tmap = np.argsort(np.argmax(G0,axis=0))
                x = []
                for i,w in enumerate(weights):
                    x.append(w * target.x[Tmap] + (1-w)*source.x)
                    
            else: # Now all pointmasses are not equal and we have no valid approach in 2D so raise a flag
                
                raise Bary2Dpointmassnotimpemented
                
        return x # return barycentral co-ordinates for 2D points and given weights

    else: # treat input as a continuous distribution in 2D (not implemented)

        if(sinkhorn): # use Sinkhorn's bregmann iteration algorithm. Depends on choice of regularization parameter which convolves the PDFs with a Gaussian and smoothes them.
            reg = 0.005 
            x =  []
            if(noPOTlibrary): raise POTlibraryError

            for i,w in enumerate(weights):
                A = [source.pdf,target.pdf] # give two PDFs
                weightspair = np.array([w,1-w]) # give a pair of weights
                x.append(ot.bregman.convolutional_barycenter2d(A, sinkhornreg, weightspair,numItermax=20000))
                
        else: # use direct optimisation of sliced Wasserstein for weights pair of source and target
            
            return baryoptSliced(source,target,Nproj,weights,verbose=verbose)
    return x

def obj(p,PDFlist,weights,Nproj,verbose=False):
    objective = 0.
    for i,ps in enumerate(PDFlist):
        if(weights[i] == 0.): continue
        target = ps
        source = OTpdf((p.reshape((target.nx,target.ny)),target.x)) # set up source object
        [Ws] = SlicedWasserstein(source,target,Nproj,ignoreCommonCDFerror=True)
        objective += Ws*weights[i]
        Wslast = Ws
        if(verbose): 
            if(i==len(PDFlist)-1): print(i,Wslast*weights[i-1],Ws*weights[i],objective)
    return objective
        
def dobj(p,PDFlist,weights,Nproj,verbose=False):
    dobjective = np.zeros(PDFlist[0].nx*PDFlist[0].ny)
    for i,ps in enumerate(PDFlist):
        if(weights[i] == 0.): continue
        target = ps
        source = OTpdf((p.reshape((target.nx,target.ny)),target.x)) # set up source object
        Ws,dWs = SlicedWasserstein(source,target,Nproj,derivatives=True,ignoreCommonCDFerror=True)
        #if(verbose): print(i,Ws,weights[i])
        dobjective += dWs.flatten()*weights[i]
    return dobjective

def baryoptSliced(source,target,Nproj,weights,returnoptres=False,verbose=False): # perform minimization of Sliced Wasserstein distance for barycentral PDF between source and target PDFs.
    x0 = None
    A = [source,target] # give two PDFs
    bpaths = []
    results = []
    x0 = source.pdf.flatten()
    bnds = [(0,None) for i in range(source.n)]
    Ac = np.ones((1,len(x0)))
    bc = np.array([1.0])
    cons = [{"type": "eq", "fun": lambda x: Ac @ x - bc}]
    a = source.x.reshape((source.n, source.ndim))
    b = target.x.reshape((target.n, target.ndim))
    C = ot.dist(a,b,metric='sqeuclidean') # loss matrix

    for i,w in enumerate(weights):
        wpair = np.array([1-w,w]) # give a pair of weights
        if(verbose): print('\n Optimizing for weights ',wpair)
        args = (A,wpair,Nproj)
        #res = minimize(obj, x0, args, jac=dobj,bounds=bnds,constraints=cons) 
        if(verbose):
            bary,res = baryoptSliced_multi(A,Nproj,wpair,returnoptres=True,verbose=False,x0=x0,sliced=False,dist=None)
            if(returnoptres): results.append(res)
        else:
            bary = baryoptSliced_multi(A,Nproj,wpair,verbose=False,x0=x0,sliced=False,dist=None)
            
        if(verbose): print(res)
        bpaths.append(res.x.reshape((source.nx,source.ny)))
        x0 = res.x
    if(returnoptres): return bpaths, results
    return bpaths

def bary_opt_obj(p,PDFlist,weights,C,verbose=False): # objective function for optimization of Full Wasserstein distances for barycentres
    objective = 0.
    p/= np.sum(p)
    for i,ps in enumerate(PDFlist):
        if(weights[i] == 0.): continue
        target = ps
        source = OTpdf((p.reshape((target.nx,target.ny)),target.x)) # set up source object
        #print(' source ',np.sum(source.pdf))
        #print(' target ',np.sum(target.pdf))
        [Ws] = wasserPOT(source,target,maxiters=300000,distfunc=C)
        #Ws = OT.Wasser_LinProg(source,target,distfunc=C)
        objective += Ws*weights[i]
        Wslast = Ws
        if(verbose): 
            if(i==len(PDFlist)-1): print(i,Wslast*weights[i-1],Ws*weights[i],objective)
    return objective

def bary_opt_obj_sliced(p,PDFlist,weights,Nproj,verbose=False): # objective function for optimization of Sliced Wasserstein distances for barycentres
    objective = 0.
    for i,ps in enumerate(PDFlist):
        if(weights[i] == 0.): continue
        target = ps
        source = OTpdf((p.reshape((target.nx,target.ny)),target.x)) # set up source object
        [Ws] = SlicedWasserstein(source,target,Nproj,ignoreCommonCDFerror=True)
        #Ws = np.sqrt(Ws) # the routine returns the square of the Siced Wasserstein distance (for p=2)
        objective += Ws*weights[i]
        Wslast = Ws
        if(verbose): 
            if(i==len(PDFlist)-1): print(i,Wslast*weights[i-1],Ws*weights[i],objective)
    return objective
        
def bary_opt_dobj_sliced(p,PDFlist,weights,Nproj,verbose=False): # derivative of objective function for optimization of Sliced Wasserstein distances for barycentres
    dobjective = np.zeros(PDFlist[0].nx*PDFlist[0].ny)
    for i,ps in enumerate(PDFlist):
        if(weights[i] == 0.): continue
        target = ps
        source = OTpdf((p.reshape((target.nx,target.ny)),target.x)) # set up source object
        Ws,dWs = SlicedWasserstein(source,target,Nproj,derivatives=True,ignoreCommonCDFerror=True)
        #Ws = np.sqrt(Ws) # the routine returns the square of the Sliced Wasserstein distance (for p=2). So convert to Wasserstein distance.
        #dWs/= 2*Ws # convert derivative for Ws**2 to derivative of Ws
        #if(verbose): print(i,Ws,weights[i])
        dobjective += dWs.flatten()*weights[i]
    return dobjective

def baryoptSliced_multi(A,Nproj,weights,returnoptres=False,verbose=False,x0=None,sliced=True,dist=None):
    results = []
    if(x0 is None):
        x0 = A[0].pdf.flatten()
    bnds = [(0,None) for i in range(A[0].n)]
    Ac = np.ones((1,len(x0)))
    bc = np.array([1.0])
    cons = [{"type": "eq", "fun": lambda x: Ac @ x - bc}]
    if(verbose): print('\n Optimizing for weights ',weights)
    if(sliced):
        args = (A,weights,Nproj)
        #res = minimize(OT.obj, x0, args, jac=OT.dobj,bounds=bnds)        
        res = minimize(bary_opt_obj_sliced, x0, args, jac=bary_opt_dobj_sliced,bounds=bnds,constraints=cons)        
    else:
        if(dist is None):
            source,target = A[0],A[1]
            a = source.x.reshape((source.n, source.ndim))
            b = target.x.reshape((target.n, target.ndim))
            C = ot.dist(a,b,metric='sqeuclidean') # loss matrix
        else:
            C = dist
        args = (A,weights,C)
        res = minimize(bary_opt_obj, x0, args, bounds=bnds,constraints=cons)        
    if(verbose): print(res)
    bpaths = res.x.reshape((A[0].nx,A[0].ny))
    if(returnoptres): return bpaths, res
    return bpaths

def barypath_pointmass1D(source,
                       target,
                       weights):
    """

    Calculates Barycentral path for the Wasserstein metric W_p^p(f,g) for 1D point masses

    Works for p=1 and 2 for 1-D PDFs of arbitrary length and location


    Parameters: 
        
        source  (PDF object): 1D PDF object used as source for Optimal Transport (created with OT.PDF)
        target  (PDF object): 1D PDF object used as target for Optimal Transport (created with OT.PDF)
        weights  list: set of weights (0,1) defining each point along the Barycentral path between f and g.
    
    
    Returns:
        
        pdf_int : List with interpolated (f,fx) pairs for each weight value
            
    """

    cf = source.cdf
    cg = target.cdf
            
    a = np.append(cf[:-1],cg)
    tkarg = np.argsort(a)
    tk = a[tkarg]
    indf = list(map(lambda x:bisect.bisect_left(cf,x) ,tk))
    indg = list(map(lambda x:bisect.bisect_left(cg,x) ,tk))
   
    xft = source.x[indf]
    xgt = target.x[indg]

    pdf_int_amp = [] # create list
    pdf_int_x = [] # create list
    a = np.insert(tk[1:]-tk[:-1],0,tk[0])
    for i,w in enumerate(weights):
        pdf_int_x += [weights[i]*xgt + (1.-weights[i])*xft]
        pdf_int_amp += [a]
    pdf_int_amp[0] = source.pdf
    pdf_int_amp[-1] = target.pdf
    pdf_int_x[0] = source.x
    pdf_int_x[-1] = target.x
    return pdf_int_amp,pdf_int_x

def barypath1D(source,
             target,
             weights,
             npoints=50000,
             returntaxis=False,
             pointmass=False):

    """

    Calculates Barycentral path for the Wasserstein metric W_p^p(f,g) for 1D PDFs

    Works for p=1 and 2 for 1-D PDFs of arbitrary length and location


    Parameters: 
        
        source  (PDF object): 1D PDF object used as source for Optimal Transport (created with OT.PDF)
        target  (PDF object): 1D PDF object used as target for Optimal Transport (created with OT.PDF)
        weights  list: set of weights (0,1) defining each point along the Barycentral path between f and g.
    
    Returns:
        
        pdf_int : Array (nweights,2,npoints) with interpolated (fx,f) pairs for each intermediate location in weights

    Notes:
        If pointmass is true then the special case of transport of discrete point masses is used (i.e. no interpolation of CDFs). 
        In this case npoints = source.n + target.n - 1, which is the number of discrete stair elements of the union of the CDFs.
        Note that this case creates splitting of discrete masses in at most npoints intermediate point masses.
        
        If pointmass is false then input PDFs are assumed effectively continuous. In this case PDFs at intermediate points along
        the baycentral paths between source and target are found by interpolating the CDFs at equal spacings along the cummulative 
        index t (0 <= t <= 1). 
            
    """

    cf = source.cdf
    cg = target.cdf
    fx = source.x
    gx = target.x

    if(pointmass): # treat input is a set of pointmasses and use discrete approach to transport masses
        a = np.append(cf[:-1],cg)
        tkarg = np.argsort(a)
        tk = a[tkarg]
        indf = list(map(lambda x:bisect.bisect_left(cf,x) ,tk))
        indg = list(map(lambda x:bisect.bisect_left(cg,x) ,tk))
        xft = source.x[indf]
        xgt = target.x[indg]
        pdf_int_amp = [] # create list
        pdf_int_x = [] # create list
        a = np.insert(tk[1:]-tk[:-1],0,tk[0])
        # MS 29-10-24 removed the zero amplitude intermediate points caused by common CDF values between f and g.
        # pdf_int = np.zeros((len(weights),2,len(tk))) 
        pdf_int = np.zeros((len(weights),2,len(a[a!=0])))
        for i,w in enumerate(weights):
            pdf_int[i,0] = weights[i]*xgt[a!=0] + (1.-weights[i])*xft[a!=0]
            pdf_int[i,1] = a[a!=0]
        return pdf_int
    
    else: # treat input as a continuous distribution 
        t = np.linspace(0.0, 1.0, npoints)
        cfint = np.interp(t, cf, fx) # interpolate inverse CDFs onto regular `t` grid.
        cgint = np.interp(t, cg, gx) # interpolate inverse CDFs onto regular `t` grid.
        pdf_int = np.zeros((len(weights),2,npoints)) 

        for i,w in enumerate(weights):
            cfi = cgint*w +(1-w)*cfint # calculate intermediate barycentral distribution for each weight
            pdf_int[i,0] = cfi # x values of PDF
            pdf_int[i,1] = np.gradient(t,cfi) # amplitude of PDF
    
        if(returntaxis): return pdf_int, t
        return pdf_int

def wasserNumInt(source,
                 target):
    cf = source.cdf
    fx = source.x
    cg = target.cdf
    gx = target.x

    nx=10000  # number of points to integrate over                       
    t = np.linspace(0,1,nx)          # t-axis evaluation points for plot

    dfx = fx[1:]-fx[:-1] 
    IF = fx[0] + np.dot(dfx,np.transpose(list(map(lambda x: np.heaviside(x-cf[:-1],0.0), t))))
    dgx = gx[1:]-gx[:-1]
    IG = gx[0] + np.dot(dgx,np.transpose(list(map(lambda x: np.heaviside(x-cg[:-1],0.0), t))))
    iFGdiff = IF-IG # Difference in inverse Cummulative functions
    
    delt = 1.0/(len(t)-1)
    W1est = np.sum(delt*np.abs(iFGdiff))

    W2est = delt*np.dot(iFGdiff,iFGdiff)

    return W1est,W2est # return W1, W2^2

def wasser_find_optplan(source,
                        target,
                        W,
                        distfunc=None,
                        args=None): 

    """
          Find optimal transport plan with knowledge of Wasserstein^p optimal value (note W is to the power of p).
          This method assumes data separation is constant and equal to dx and domain is (0,(n-1)*dx).
 
    """
    f = source.pdf
    g = target.pdf
    fn = list(f)
    gn = list(g)
    n = len(fn)
    b_eq_ab = np.array(fn+gn)
    d,A = _calc_distArray(source,target,distfunc=distfunc,args=args)
            
    c = np.reshape(d,n*n) # W-metric
    A_eq = np.append(A,c).reshape(2*n+1,n*n)
    b_eq = np.append(b_eq_ab,W*np.sum(f))
    output = lsq_linear(A_eq,b_eq,bounds=(0.0, np.inf),method='bvls')
    s = output.success
    if(output.success):
        H = (output.x).reshape(n,n) # optimal plan
        #Tf2g = np.dot(H.T,np.diag(1/f)) # Transform matrix from source to target 
    else:
        print('\n Problem with Linear solver for finding optimal plan \n')
        print(output)
        H = None
        #Tf2g = None
            
    #return s, Tf2g, H # return W^p, Transport plan for source to target, Transform matrix from source to target  
    return s, H # return W^p, Transport plan for source to target 

def wasserPOT(source,
              target,
              distfunc='W2',
              returnplan=False,
              returndist=False,
              maxiters=100000): # only implemented for W1, W2 not W12

    if(noPOTlibrary):
        #print('POT library not installed')
        raise POTlibraryError
    if(isinstance(distfunc, str)):
        if(distfunc=='W2'): metric = 'sqeuclidean'
        elif (distfunc=='W1'):  metric = 'cityblock'
        a = source.x.reshape((source.n, source.ndim))
        b = target.x.reshape((target.n, target.ndim))
        A = ot.dist(a,b,metric=metric) # loss matrix
    elif (distfunc.ndim==2):
        A = distfunc
    else:
        raise UnknownOTDistanceTypeError
    M = A/A.max()
        
    G0 = ot.emd(source.pdf.flatten(), target.pdf.flatten(), M,numItermax=maxiters)
    w = np.sum(G0*A)
    out = [w]
    if(returnplan): out +=[G0]
    if(returndist): out +=[A]
        
    return out

#
# Andreas Bærentzen's implementation of the Sinkhorn algorithm for
# Entropically Smoothed Wasserstein-2
#
# M. Sambridge, Canberra, 2020
#

def filter(image,
           sigma):

    return gaussian_filter(image,sigma,mode='constant',truncate=32)

powv = np.vectorize(pow)
maxv = np.vectorize(max)
logv = np.vectorize(lambda x: np.log(max(1e-300,x)))
    
def SinkhornAB(mu,
               sigma,
               verbose=False): # multi-dimensional mu[0] = f, mu[1] = g, assumes unit spacing on regular grid.

    """
      Andreas Bærentzen's implementation of the Sinkhorn algorithm for
      Entropically Smoothed Wasserstein-2

    """

    iter = 5001
    v = np.ones(mu[0].shape)
    w = np.ones(mu[0].shape)
    for i in range(0,iter):
        v = mu[0] / maxv(filter(w,sigma), 1e-300)
        w = mu[1] / maxv(filter(v,sigma), 1e-300)
        wasserstein_dist = (mu[0]*(logv(v))+mu[1]*(logv(w))).sum()*sigma
        if (i % 1000 == 0 and verbose): 
            print('Sinkhorn distance: ' + str(wasserstein_dist))
    return (wasserstein_dist,v,w)

def Sinkhorn(source,
             target,
             gamma=0.005,
             verbose=False,
             iter=250): # multi-dimensional mu[0] = f, mu[1] = g, assumes unit spacing on regular grid.

    """
      Andreas Bærentzen's implementation of the Sinkhorn algorithm for
      Entropically Smoothed Wasserstein-2 interfacing to local paramter class for PDFs.

    """

    s = source.pdf
    t = target.pdf
    #iter = 5001
    v = np.ones(s.shape)
    w = np.ones(s.shape)
    for i in range(0,iter):
        v = s / maxv(filter(w,gamma), 1e-300)
        w = t / maxv(filter(v,gamma), 1e-300)
        wasserstein_dist = (s*(logv(v))+t*(logv(w))).sum()*gamma
        if (i % 1000 == 0 and verbose): print('Sinkhorn distance: ' + str(wasserstein_dist))
    return (wasserstein_dist,v,w) # return W_pe

def Sinkhorn_MS(sou,
                tar,
                gamma=0.0005,
                maxiters = 5001,
                verbose=False): 
    
    """
    Sinkhorn iterative algorithm for calculation of W_2 entropically smoothed Wasserstein distance from sinkhorn.ipynb

    Generalized to work with different length f and g. Original assumed symmetric M matrix.
    
    Only seems to work if fx and gx are linearly spaced between 0 and 1 ?
    
    # Modified by M. Sambridge from Mike Snow's code to handle 1D or 2D input PDFs
    """
    
    f = sou.pdf
    g = tar.pdf
    fx = sou.x
    gx = tar.x
    source2 = (f / np.sum(f)).reshape((sou.n,1))
    target2 = (g / np.sum(g)).reshape((tar.n,1))
    #n1 = sou.n
    n2 = tar.n
    cost = pairwise_distances(fx.reshape(sou.n,sou.ndim), 
                        gx.reshape(tar.n,tar.ndim),
                        metric='sqeuclidean')
        
    M = np.exp(-cost/gamma)
    amp = M.max()
    M/=amp

    #mu = np.ones((n1,1))
    nu = np.ones((n2,1))

    for k in range(maxiters):
        mu = np.divide(source2, np.dot(M, nu))
        nu = np.divide(target2, np.dot(M.T, mu))
        #if (k % 100): print(k,'mu',mu,'nu',nu)
        #if (k <= 1): print(k,'mu',mu,'nu',nu)
    
        if (k % 1000 == 0 and verbose): 
            pi = np.dot(np.dot(np.diag(nu.reshape(-1)),M.T),np.diag(mu.reshape(-1)))
            sinkhorn_distance = np.sqrt(amp*sum(sum(np.multiply(pi.T, cost))))
            print('Iter:',k,' Sinkhorn distance: ' + str(sinkhorn_distance))
    pi = np.dot(np.dot(np.diag(nu.reshape(-1)),M.T),np.diag(mu.reshape(-1)))
    sinkhorn_distance = np.sqrt(amp*sum(sum(np.multiply(pi.T, cost))))

    return sinkhorn_distance**2,pi # return W_pe^p, Transport plan

# calculate W2 using POT sinkhorn algorithm

def sinkhornPOT(source,
                target,
                distfunc='W2',
                returnplan=False,
                gamma=0.0005,
                returndist=False): # only implemented for W1, W2 not W12

    """
        Interface local parameter classes for PDFs to POT library sinkhorn solver.
    """
    #lambd = 2e-3
    if(noPOTlibrary):
        #print('POT library not installed')
        raise POTlibraryError
    a = np.copy(source.pdf)
    b = np.copy(target.pdf)

    # This code does not like zeros in the amplitude

    if(len(np.where(a == 0.)[0]>=1)):
       print(' Warning in sinkhornPOT: input PDF has a zero amplitude. Replacing with non zeros')
       a[np.where(a == 0.)] = np.min(a[np.where(a != 0.)]) # replace zeros with minimum non zero amplitude
    elif (len(np.where(b == 0.)[0]>=1)):
       print(' Warning in sinkhornPOT: input PDF has a zero amplitude. Replacing with non zeros')
       b[np.where(b == 0.)] = np.min(b[np.where(b != 0.)]) # replace zeros with minimum non zero amplitude
       
    if(isinstance(distfunc, str)):
        if(distfunc=='W2'): metric = 'sqeuclidean'
        elif (distfunc=='W1'):  metric = 'manhattan'
        else:
            print('distfunc parameter not recognized: ',distfunc)
        a = source.x.reshape((source.n, source.ndim))
        b = target.x.reshape((target.n, target.ndim))
        A = ot.dist(a,b,metric=metric) # loss matrix
    elif (distfunc.ndim==2):
        A = distfunc
    else:
        print(' Error in wasserPOT: Do not recognize distfunc =',distfunc)
    M = A/A.max()

    Gs = ot.sinkhorn(source.pdf.flatten(), target.pdf.flatten(), M, gamma)

    w = np.sum(Gs*A)
    out = [w]
    if(returnplan): out +=[Gs]
    if(returndist): out +=[A]
        
    return out
    
def MargWasserstein(source,
			target,
			distfunc='W2',
			derivatives=False,
			verbose=False,
			memory=False,
			returnmargW=False,
                        ignoreCommonCDFerror=False,
                        reflectX=False,
                        reflectY=False,
                        resetmeans=False):
    """
       Calculates the Marginal Wasserstein distance for x and y marginals of source and target 2D PDFs. 
    
            Input:
                source - OTpdf object; contains 2D source PDF [as ndarray with shape(source.nx,source.ny)]
                target - OTpdf object; contains 2D target PDF [as ndarray with shape(target.nx,target.ny)]
                distfunc - string or ndarray; determines p value for Wasserstein or array of supplied 
                           distances for each pair of elements in source and target discretized PDF                
                derivatives - logical; True to return vectors of derivatives of Wasserstein with 
                                       respect to density amplitudes of 2D PDF
                returnmargW - logical; True to perform calculations separately on each marginal; 
                                       False then return average Wassersteins.
                                      
            Output:
                out - List of Wassersetin distances and optionally derivatives, where:
                    
                0.5*(Wx+Wy) = out                                  : if derivatives==False; returnmargW==False
                [0.5*(Wx+Wy),0.5*(dwxdu+dwydu),0.5*(dwxdt0)] = out : if derivatives==True; returnmargW==False
                [Wx,Wy] = out                                      : if derivatives==False; returnmargW==True
                [Wx,Wy,dwxdu,dwydu,dwxdx0] = out                   : if derivatives==True; returnmargW==True
                
            where,
                Wx = Wasserstein distance between X marginals
                Wy = Wasserstein distance between Y marginals
                dwxdu = Derivatives of Wasserstein distance between X marginals w.r.t. 2D density amplitudes
                dwydu = Derivatives of Wasserstein distance between Y marginals w.r.t. 2D density amplitudes
                dwxdx0 = Derivative of Wasserstein distance between X marginals w.r.t. x co-ordinate of axis origin 
                                        
                Note that dwydx0 = dwxdy0 = 0.                       
    
    """

    if(source.type !='2D'): raise TargetSource2DShapeError
    if(target.type !='2D'): raise TargetSource2DShapeError
    if(type(distfunc) == str):
        if(distfunc=='W12'): raise MarginalWassersteinError(mset='W12')

    if(source.calcmarg): source.setMarginals() # calculate source marginal PDFs
    if(target.calcmarg): target.setMarginals() # calculate target marginal PDFs
                            
    if(derivatives): 
        dwp = np.zeros((source.nx,source.ny)) # define derivatives array
        dwpmargX  = np.zeros_like(dwp)
        dwpmargY  = np.zeros_like(dwp)
    wp = 0.
    Nproj = 2
    
    wpmarg  = np.zeros(2) #  W_p for each marginal
    dwgmarg = [0.]*2 #  deriv wrt t grid for each marginal
    
    r = [reflectX,reflectY]
    for i in range(2): # loop over marginals
        
        s = source.marg[i]
        t = target.marg[i]
        
        wout = wasser(s,t,distfunc=distfunc,
                      derivatives=derivatives,
                      checkCommonCDF=True,
		      ignoreCommonCDFerror=ignoreCommonCDFerror,
                      memory=memory,reflect=r[i])
        
        wsqpd = wout[0] # place output of wasser in local variables

        if(derivatives): 
            wsqpd, dw = wout[0:2]
            if(i==0):
                dwp[:] += dw
                dwg = wout[2]  
                dwgmarg[i] = dwg
                if(returnmargW): dwpmargX = np.copy(dwp)

            else:
                dwp.T[:] += dw
                if(returnmargW): dwpmargY.T[:] += dw
        else:   
            wsqpd = wout[0]
            
        wpmarg[i] = wsqpd
        wp += wsqpd # sum W^p over marginals
            
        if(verbose): print('Projection ',i,' completed w =',np.sqrt(wsqpd),' theta ',source.angles[i]*180/np.pi)        #

    out=[wp/Nproj] # return average W**p of marginals
    
    outMarg = [[wpmarg[0],wpmarg[1]]] # return W**p separately for each marginal
 
    if(derivatives): 
        dwp -= np.dot(dwp.reshape(source.n),source.pdf.reshape(source.n)) # calculate derivatives w.r.t. unormalised source PDF amplitudes
        dwp /=source.amp
        if(returnmargW):
            dwpmargX -= np.dot(dwpmargX.reshape(source.n),source.pdf.reshape(source.n)) # calculate derivatives w.r.t. unormalised source PDF amplitudes first marginal
            dwpmargY -= np.dot(dwpmargY.reshape(source.n),source.pdf.reshape(source.n)) # calculate derivatives w.r.t. unormalised source PDF amplitudes second marginal
            dwpmargX /= source.amp
            dwpmargY /= source.amp
            outMarg +=[[dwpmargX,dwpmargY]] 
            outMarg +=[dwgmarg]
        out +=[dwp/Nproj] # return derivatives of average W**p
        out +=[dwg/Nproj] # return derivatives w.r.t time window position
        #out +=[dwp.reshape((source.nx,source.ny))/Nproj] # return derivatives of average W**p
    if(returnmargW): return outMarg
    return out
    
def SlicedWasserstein(source,
                      target,
                      Nproj,
                      distfunc='W2',
                      derivatives=False,
                      returnplan=False,
                      verbose=False,
                      returnProjpoints=False,
                      calcWplan=False,
                      calcAvgW=True,
                      origin=[0.5,0.5],
                      memory=False,
                      reflect=False,
                      returnderivwindow=False,
                      ignorepointcloud=False,
                      ignoreCommonCDFerror=False,
                      phi=45.0,
                      theta = None): # Calculate sliced Wasserstein and transport plan from 2D OT from projection of Nproj 1-D solutions
    """
       Calculates the Sliced Wasserstein distance for projections of source and target 2D PDFs. 
    
            Input:
                source - OTpdf object; contains 2D source PDF [as ndarray with shape(source.nx,source.ny)]
                target - OTpdf object; contains 2D target PDF [as ndarray with shape(target.nx,target.ny)]
                Nproj - Number of angular projections about middle of axes
                distfunc - string, ndarray or tuple; determines p value for Wasserstein or array of supplied 
                           distances for each pair of elements in source and target discretized PDF                
                derivatives - logical; True to return vectors of derivatives of Wasserstein with 
                                       respect to density amplitudes of 2D PDF
                returnplan - logical; True to return average Transport plan over slices.
                origin - ndarray or tuple; origin of Sliced Wasserstein axes used for rotation. 
                phi -    scalar; angle in degrees of weighting defining distance metric (w_x = np.sqrt(2)*cost(phi), w_y = np.sqrt(2)*sin(phi))
                theta -  array; angle in radians of slice axes. If None then default of np.linspace(0.1745,np.pi,Nproj+1[:-1]) are used. 
                                                                If set, then Nproj is re-set to len(theta).
                                      
            Output:
                out - List of Wasserstein distances, Transport plan and derivatives, where:
                    
                [wsliced] = out                     : if derivatives==False; returnplan==False
                [wsliced,H] = out                   : if derivatives==False; returnplan==True
                [wsliced,dwsliced] = out            : if derivatives==True; returnplan==False, returnderivwindow==False
                [wsliced,dwsliced,dpos] = out       : if derivatives==True; returnplan==False, returnderivwindow==True
                [wsliced,dwsliced,dpos,H,dH] = out  : if derivatives==True; returnplan==True, returnderivwindow==True
                
            where,
                wsliced  = float; Sliced Wasserstein distance [=sum(w_i)/Nproj, (i=1,...,Nproj)]
                wi       = Wasserstein distance between i-th projection of source and target (i=1,...,Nproj)
                H        = ndarray, shape(source.nx*source.ny,target.nx*target.ny); Transport plan Matrix
                dwsliced = ndarray; shape(source.nx,source.ny); Derivatives of Sliced Wasserstein 
                           distance w.r.t. 2D source density amplitudes 
                dpos     = derivative of Wasserstein distance with respect to pdf x origin.
                dH       = ndarray, shape(source.nx*source.ny,target.nx*target.ny,target.nx*target.ny); 
                           Derivative of Wasserstein distance between X marginals w.r.t. x co-ordinate of axis origin 
    """
    
    # Calculation  of Wplan can be achieved two ways: 
    #   1) with distfunc = 'W2' or `W1' and calcWplan = True; This is very slow as it requires transport plan to be calculated for each projection
    #   2) with distfunc = (A) containing precalculated distances between all points (W1 or W2), and calcAvgW = True; Result is fast Wplan
    
    if(source.type !='2D'): raise TargetSource2DShapeError
    if(target.type !='2D'): raise TargetSource2DShapeError
        
    wx,wy = getxyweights(phi)

    # added 14/10/2024 to allow the user to set slice angles.
    if(source.upointcloud and not ignorepointcloud):
        if(type(distfunc) is tuple):
            p,A = distfunc # if distance matrix A is supplied it assumed consistent with the value of phi provided.
        elif(type(distfunc) is np.ndarray):
            A = distfunc   # if distance matrix A is supplied it assumed consistent with the value of phi provided.
            if(derivatives):
                raise SlicedWassersteinPointCloudError()
            else:
                p=0 # dummy value of p which is not used because derivatives are not being calculated
        else:
            metric,p = 'sqeuclidean',2
            if(distfunc == 'W1'):metric,p = 'cityblock',1
            A = pairwise_distances(source.x*np.array([wx,wy]), target.x*np.array([wx,wy]), metric=metric) # use Euclidean distance matrix to power p

        return SlicedWassersteinPointCloud( source, target, Nproj, A, p=p,
                                           derivatives=derivatives, 
                                           returnplan=returnplan, 
                                           returnderivwindow=returnderivwindow, 
                                           origin=origin,
                                           phi=phi)

    if(source.calcproj or source.nproj !=Nproj): # projections have not previously been calculated and so we do so here
        source.setSliced(Nproj,origin,phi=phi) # calculate source projection PDFs
    if(target.calcproj or target.nproj !=Nproj):
        target.setSliced(Nproj,origin,phi=phi) # calculate target projection PDFs
            
    sizesource = source.n
    sizetarget = target.n
    #if(reflect):  # all plan arrays are twice as big along each dmension if we are in reflect mode
    #    sizesource = 2*source.n
    #    sizetarget = 2*target.n

    if(type(distfunc) is np.ndarray): 
        distfunction_args = (source,target,2,distfunc)
    elif(type(distfunc) is tuple):
        p,A = distfunc
        distfunction_args = (source,target,p,A)
    else:
        distfunction_args = distfunc
    
    if(returnProjpoints): # return the projected points along each slice axis
        fproj = np.zeros((Nproj,2,sizesource))
        gproj = np.zeros((Nproj,2,sizetarget))
        theta = source.angles
        for i in range(Nproj):
            fxp = source.proj[i].x
            gxp = target.proj[i].x
            fproj[i,0] = origin[0] + fxp*np.cos(theta) # calculate locations of projected masses for plotting
            fproj[i,1] = origin[1] + fxp*np.sin(theta)
            gproj[i,0] = origin[0] + gxp*np.cos(theta)
            gproj[i,1] = origin[1] + gxp*np.sin(theta)
            
    if(calcWplan or returnplan): 
        Hgp = np.zeros((sizesource,sizetarget)) # define optimal plan array
    if(derivatives): 
        if(calcWplan or returnplan):
            dHgp = np.zeros((sizesource,sizesource,sizetarget)) # define optimal plan array
            dHgpdummy = np.zeros((sizesource,sizesource,sizetarget)) # define optimal plan array
        dwp = np.zeros((sizesource)) # define derivative
    wp = 0.
    dw_winp = 0.
    for i in range(Nproj): # loop over projections
        
        s = source.proj[i]
        t = target.proj[i]

	# if reflect mode is on for 2D PDFs then Wasserstein is calculated on symmetric PDFs. We must turn this on for projected PDFs

        #
        # Rabin et al. (2012) shows that the optimal map for point clouds is the identity in the sorted ordering of fx and gx.
        # So we don't need to do this calculation for equal weight masses.
        #
        fxpargsort = source.psorted[i]
        gxpargsort = target.psorted[i]

        plan = False                # do not calculate and average of 1D Transport Plans by default
        if(returnplan): plan = True # calculate and average of 1D Transport Plans if we are returning it (Slow!)
        if(calcWplan and type(distfunc) is not np.ndarray and type(distfunc) is not np.ndarray): 
            plan = True # calculate and average the 1D Transport Plans 
                                                                        # if we need it for Wplan calculation.
                                                                        # Note it is much faster to provide 
                                                                        # distfunc=distance matrix and set calcWplan=True 
        #print(' plan = ',plan)
        #print(' calcWplan = ',calcWplan)
        #print(' distfunc type = ',type(distfunc))
        
        if(plan):
           a = np.repeat(fxpargsort,sizesource).reshape(s.n,s.n)
           b = np.tile(gxpargsort,sizetarget).reshape(t.n,t.n)
        #print(i,' :',distfunction_args,' s',s.x,' t ',t.x)
        
        wout = wasser(s,t,distfunc=distfunction_args,proj=i,
                      derivatives=derivatives,
                      returnplan=plan,
                      checkCommonCDF=True,
                      ignoreCommonCDFerror = ignoreCommonCDFerror,
                      reflect=reflect,
                      memory=memory)
        
        wsqpd = wout[0] # place output of wasser in local variables

        if(derivatives and plan ):
            #wsqpd, dw, H, dH = wout[0:4] removed 17/12/20 because of updated functionality of wasser returning additional derivative.
            wsqpd, dw, dw_win, H, dH = wout[0:5] # allow for window derivatives returned from wasser but for now ignore
            dHgpdummy[:,a,b] = dH
            dHgp[fxpargsort] += dHgpdummy
            #print(np.shape(dwp),np.shape(fxpargsort),np.shape(dw))
            dwp[fxpargsort] += dw
            dw_winp += dw_win*np.cos(source.angles[i]) # transform derivative of project x-axis
            Hgp[a,b]+=H
        elif(not derivatives and plan ):
            wsqpd, H = wout[0:2]
            Hgp[a,b]+=H
        elif(derivatives and not plan): 
            #wsqpd, dw = wout[0:2] removed 17/12/20 because of updated functionality of wasser returning additional derivative.
            #wsqpd, dw = wout[0:2]
            wsqpd, dw, dw_win = wout[0:3] # updated 15/06/23 to return derivatives with respect to x axis position of input PDF
            dwp[fxpargsort] += dw  
            dw_winp += dw_win*np.cos(source.angles[i]) # transform derivative of project x-axis
        else:   
            wsqpd = wout[0]
            
            #for k in range(source.n):
            #    for kk in range(sizetarget):
            #        Hgp[fxpargsort[k],gxpargsort[kk]] += H[k,kk]
        wp += wsqpd # sum W^p from projections
            
        if(verbose): print('Projection ',i,' completed w =',np.sqrt(wsqpd),' theta ',source.angles[i]*180/np.pi)        #

    out = []
    if(calcWplan):
        if(plan):  # calculate Wplan from Transport plan (slow)
            Hgp = Hgp/Nproj # calculate average optimal plan
            d = _calc_distArray(source,target,distfunc=distfunc)[0]
            c = np.reshape(d,source.n*target.n) 
            wplan = float(c.dot(Hgp[:source.n,:target.n].reshape(source.n*target.n))) # calculate W^p from average optimal plan
            out += [wplan] # return W**p calculated from the average optimal plan over projections
            if(derivatives): 
                dwplan = np.dot(dHgp[:source.n,:target.n].reshape(source.n,source.n*target.n),c)/Nproj # calculate W^p from average optimal plan
                dwplan -= np.dot(dwplan,source.pdf.reshape(sizesource)) # calculate derivatives w.r.t. unormalised source PDF amplitudes
                dwplan /=source.amp
                if(source.pointcloud):
                    out +=[dwplan] # return plan
                else:
                    out += [dwplan.reshape((source.nx,source.ny))] # 
        else:
            out+=[wp/Nproj] # return average W**p from projections
            if(derivatives): 
                dwp -= np.dot(dwp,source.pdf.reshape(sizesource)) # calculate derivatives w.r.t. unormalised source PDF amplitudes
                dwp /=source.amp
                if(source.pointcloud):
                    out +=[dwp/Nproj] # return derivatives of average W**p
                else:
                    out +=[dwp.reshape((source.nx,source.ny))/Nproj] # return derivatives of average W**p
                if(returnderivwindow): out +=[dw_winp/Nproj] # return derivatives of average W**p with respect to x window/frame of PDF points
    if(calcAvgW): 
        out+=[wp/Nproj] # return average W**p from projections
        if(derivatives): 
            dwp -= np.dot(dwp,source.pdf.reshape(sizesource)) # calculate derivatives w.r.t. unormalised source PDF amplitudes
            dwp /=source.amp
            if(source.pointcloud):
               out +=[dwp/Nproj] # return derivatives of average W**p
            else:
               out +=[dwp.reshape((source.nx,source.ny))/Nproj] # return derivatives of average W**p
            if(returnderivwindow): out +=[dw_winp/Nproj] # return derivatives of average W**p with respect to x window/frame of PDF points
    if(returnplan): 
        out+=[Hgp] # return average optimal plan from projections
        if(derivatives): 
            dHgp -= np.dot(np.transpose(dHgp),source.pdf.reshape(sizesource))
            dHgp /= source.amp                
            out +=[dHgp/Nproj] # return derivatives of average optimal plan
    if(returnProjpoints): out+=[fproj]+[gproj] # return locations of projected points for each projection
    return out

def SlicedWassersteinPointCloud(
                      source,
                      target,
                      Nproj,
                      A,
                      p=2,
                      derivatives=False,
                      returnderivwindow=False,
                      returnplan=False,
                      origin=[0.5,0.5],
                      phi=45.0,
                      theta = None):

    """
       Calculates the Sliced Wasserstein distance for projections of source and target point clouds.
            Input:
                source - OTpdf object; contains 2D source PDF [as ndarray with shape(source.nx,source.ny)]
                target - OTpdf object; contains 2D target PDF [as ndarray with shape(target.nx,target.ny)]
                Nproj  - Number of angular projections about middle of axes
                A      - ndarray; distance matrix(source.n,target.n)
                p      - scalar; p-value of norm consistent with A (needed for derivative calculation).
                derivatives - logical; True to return vectors of derivatives of Wasserstein with
                                       respect to locations of pointcloud (x,y). These depend on what A represents.
                returnplan - logical; True to return average Transport plan over slices.
                origin - ndarray or tuple; origin of Sliced Wasserstein axes used for rotation. 
                phi -    scalar; angle in degrees of weighting defining distance metric (w_x = np.sqrt(2)*cost(phi), w_y = np.sqrt(2)*sin(phi))
                theta -  array; angle in radians of slice axes. If None then default of np.linspace(0.1745,np.pi,Nproj+1[:-1]) are used. 
                                                                If set, then Nproj is re-set to len(theta).

            Output:
                out - List of Wasserstein distances, Transport plan and derivatives, where:

                [wsliced] = out                       : if derivatives==False; returnplan==False
                [wsliced,H] = out                     : if derivatives==False; returnplan==True
                [wsliced,dwsdx,dwsdy] = out           : if derivatives==True; returnplan==False; returnderivwindow==False
                [wsliced,dwsdx,dwsdy,dwsdpos] = out   : if derivatives==True; returnplan==False; returnderivwindow==True
                [wsliced,dwsdx,dwsdy,dwsdpos,H] = out : if derivatives==True; returnplan==True; returnderivwindow==True

            where,
                wsliced  = float; Sliced Wasserstein distance [=A_{ij)mean over k(H{ij}_k), (k=1,...,Nproj)]
                H        = ndarray, shape(source.n,target.n); Average Transport plan Matrix over projections
                dwsdx,dwsdy = 2xndarray; shape(source.n); Derivatives of Sliced Wasserstein w.r.t. source locations (x,y)
                dwsdpos = float; Derivatives of Sliced Wasserstein w.r.t. source window, i.e. all x locations together
                                 which is sum(dwsdx) over point cloud.

                Note: The derivatives of the Transport plan H with respect to source locations (x,y) 
                      are zero, because projection ordering only changes with finite movement of source locations.
    """
    s = source
    t = target
    wx,wy = getxyweights(phi)
    xs = s.x-origin
    xt = t.x-origin
    if(theta is None):
        theta = np.linspace(0.1745,np.pi,Nproj+1) # set angles with some offset
        theta = theta[:-1]
    else:
        Nproj = len(theta)

    r = np.array([wx*np.cos(theta),wy*np.sin(theta)])
    sarg = np.argsort(np.dot(xs,r),axis=0).T
    targ = np.argsort(np.dot(xt,r),axis=0).T
    out = [np.mean(A[sarg,targ])] # Assumes that wx and wy are consistent with supplied distance matrix, A.
    
    if(derivatives): # calculate derivatives of W wrt pointcloud locations
        dAdx = np.subtract.outer(s.x.T[0],t.x.T[0]) # these derivatives for W**p wrt to source locations
        dAdy = np.subtract.outer(s.x.T[1],t.x.T[1]) # these derivatives for W**p wrt to source locations
        if(p==2):
            dAdx = 2.*(wx**2)*dAdx # these derivatives for W**2 wrt to source locations
            dAdy = 2.*(wy**2)*dAdy # these derivatives for W**2 wrt to source locations
        elif(p==1):
            dAdx = wx*np.sign(dAdx) # these derivatives for W**p wrt to source locations
            dAdy = wy*np.sign(dAdy) # these derivatives for W**p wrt to source locations

        #dAdx = np.divide(np.subtract.outer(s.x.T[0],t.x.T[0]),A) # these derivatives for W wrt to source locations
        #dAdy = np.divide(np.subtract.outer(s.x.T[1],t.x.T[1]),A) # these derivatives for W wrt to source locations
        Dx = np.zeros((s.n))
        Dy = np.zeros((s.n))
        for i in range(Nproj):
            Dx[sarg[i,:]] += dAdx[sarg[i,:],targ[i,:]]
            Dy[sarg[i,:]] += dAdy[sarg[i,:],targ[i,:]]
        Dx /= (Nproj*s.n)
        Dy /= (Nproj*s.n)
        out +=[Dx,Dy]
        if(returnderivwindow): out +=[np.sum(Dx)]
        
    if(returnplan):
        H = np.zeros((s.n,t.n))
        for i in range(Nproj):
            H[sarg[i,:],targ[i,:]] += 1.
        out +=[H/(Nproj*s.n)]
        
    return out

def getSlicedProjection_points(source,
                               target,
                               Nproj,
                               verbose=False,
                               origin=[0.5,0.5],
                               phi=45.0):
    """
       Calculates the projections of source and target points used in Sliced Wasserstein algorithm. 
       These are returned by SlicedWasserstein(). This is a utility routine that returns the projected points without calculating the 
       SlicedWasserstein distance.
    
            Input:
                source - OTpdf object; contains 2D source PDF [as ndarray with shape(source.nx,source.ny)]
                target - OTpdf object; contains 2D target PDF [as ndarray with shape(target.nx,target.ny)]
                Nproj - Number of angular projections about middle of axes
                origin - ndarray or tuple; origin of Sliced Wasserstein axes used for rotation. 
                phi -    scalar; angle in degrees of weighting defining distance metric (w_x = np.sqrt(2)*cost(phi), w_y = np.sqrt(2)*sin(phi))
                                      
            Output:
                out - ndarray(Nproj,2) - Array of projected points along each of Nproj axes through origin.
                                    
    """
    
    if(source.type !='2D'): raise TargetSource2DShapeError
    if(target.type !='2D'): raise TargetSource2DShapeError
        
    if(source.calcproj or source.nproj !=Nproj): # projections have not previously been calculated and so we do so here
        source.setSliced(Nproj,origin,phi=phi) # calculate source projection PDFs
    if(target.calcproj or target.nproj !=Nproj):
        target.setSliced(Nproj,origin,phi=phi) # calculate target projection PDFs
            
    sizesource = source.n
    sizetarget = target.n

    
    fproj = np.zeros((Nproj,2,sizesource))
    gproj = np.zeros((Nproj,2,sizetarget))
    theta = source.angles
    for i,t in enumerate(theta):
        fxp = source.proj[i].x
        gxp = target.proj[i].x
        fproj[i,0] = origin[0] + fxp*np.cos(t) # calculate locations of projected masses for plotting
        fproj[i,1] = origin[1] + fxp*np.sin(t)
        gproj[i,0] = origin[0] + gxp*np.cos(t)
        gproj[i,1] = origin[1] + gxp*np.sin(t)
            
    return fproj,gproj # projected points

def reconstruct_2Dpointmasses_from_SlicedProjections(u,theta,origin):
                              
    
    """
       Calculates the original pointmass locations (or a best fit) from projections of source and target points used in Sliced Wasserstein algorithm. 
       For a consistent set of points, this reverses the operation of `getSlicedProjection_points`. 
       Can be used to get Barycentral locations of 2D point masses calcualted by the SlicedWasserstein algorithm. 
    
            Input:
                u - ndarray(Nproj,k); contains distances along each slice of k points
                origin - ndarray or tuple; origin of Sliced Wasserstein axes used for rotation. 
                theta -  array; angle in radians of slice axes. 
                                      
            Output:
                out - ndarray(Nproj,2) - Array of projected points along each of Nproj axes through origin.
                                    
    """
    A = np.array([np.cos(theta),np.sin(theta)]).T
    return origin + np.linalg.lstsq(A,u,rcond=None)[0].T

def optimize_source_loc(source,
           target,
           distfunc='W12',
           proj=-1,
           memory=False,
           checkCommonCDF=False,
           ignoreCommonCDFerror=False,
           reflect=False):
    """

    Finds the optimal location of source point mass positions in 1D to minimize Wasserstein distance w.r.t. target PDF.

    Works for p=1 and 2 for 1-D PDFs of arbitrary length and location

    Note: the Wasserstein distance is defined as 1/p pth power of the output, which only affects the W2 value

    Derivatives are only accurate when there are no identical values in source and target CDFs. 
    Typically this is achieved by adding a perturbation to either source or target PDF values. 
    This condition can be checked with checkCommonCDF=True.

    Parameters: 
        
        source  (PDF object): 1D or 2D PDF object used as source for Optimal Transport (created with OT.PDF)
        target  (PDF object): 1D or 2D PDF object used as target for Optimal Transport (created with OT.PDF)
    
    Optional parameters:
    
        returnplan              : Calculate optimal plan (default is False).
        memory                  : set to True to save large memory matrices and replace python matrix sums with for loops. (default False)
        checkCommonCDF          : Checks input PDFs for condition to cause derivatives to fail (default False).
        ignoreCommonCDFerror    : Does not raise an error in derivative failure condition is found (default False).
    
    Returns:
        
        x_opt A : Arrangement depends upon input options.
        
        x_opt  = New point mass locations which minimize pth power of Wasserstein distance to target PDF.
        A      = Assignment of input source mass locations to output source locations
    
    """

    calcW1,calcW2,dfunc,distfunction_args = _checkdistfunc(distfunc)

    cf = source.cdf
    cg = target.cdf
    cfamp = source.amp
    cgamp = target.amp
    nf = source.n    
    ng = target.n
    xf = source.x
    xg = target.x

    if(dfunc):
        if(type(distfunction_args) is tuple):
            dum0,dum1,p,Am=distfunction_args 
            #print('Am',Am)
        else:
            Am = distfunction_args
        #print('Am',type(Am),Am)
        if(nf != np.shape(Am)[0] or ng != np.shape(Am)[1]):
            print('source.n',nf,np.shape(Am)[0])
            print('target.n',ng,np.shape(Am)[1])
            if(reflect):
               print('Wasser routine reflect mode is on and a distance matrix has been supplied in call sequence: ',
                     '\nThis combination of features requires input distance matrix consistent with relfected PDFs')
            raise DistfuncShapeError
            
    if(derivatives or checkCommonCDF):
        cset = np.intersect1d(cg[:-1],cf[:-1])
        if(reflect):
            if(len(cset) >1):  # reflect mode always creates a single common CDF value at 0.5 so we ignore this
               if( not ignoreCommonCDFerror): raise TargetSourceCDFError(cset)
        else:
            if(len(cset) !=0): # If there are common CDF values report this unless we are told not to
               if( not ignoreCommonCDFerror): raise TargetSourceCDFError(cset)

    a = np.append(cf[:-1],cg)
    tkarg = np.argsort(a)
    tk = a[tkarg]
    indf = list(map(lambda x:bisect.bisect_left(cf,x) ,tk))
    indg = list(map(lambda x:bisect.bisect_left(cg,x) ,tk))
    dtk = np.insert(tk[1:] - tk[:-1],0,tk[0])  # A*tk_deriv
   
    if(calcW1 or calcW2):
        xft = xf[indf]
        xgt = xg[indg]
        dxft = np.abs(xft-xgt)

# Calculations required for derivatives

    out = []
    if(calcW1): # Calculate W1 and its derivative with respect to unnornalised source amplitudes
        #print(' shape of dxft:',np.shape(dxft))
        #print(' shape of dtk:',np.shape(dtk))
        W1 = np.dot(dxft,dtk)
        out += [W1]
        
    if(calcW2): # Calculate W2^2 and its derivative with respect to unnornalised source amplitudes
        dsqxft = np.multiply(dxft,dxft)
        W2 = np.dot(dsqxft,dtk)
        out += [W2]

    if(dfunc): # user supplied function provides distances between elements of source and target
    # NB derivatives with respect to grid position not currently implemented for dfunc option hence return zero derivative for now
        dxft = distfunction(indf,indg,distfunction_args,proj=proj)
        Wf = np.dot(dxft,dtk)
        out += [Wf]

    return out

## Utility plot routines

def trim_axs(axs, 
             N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def plot_optimal_transform_frames(source,
                                  target,
                                  frames,
                                  plotsum=False): # plots frames of transform from f to g (both positive PDFs)
    f = source.pdf
    fx = source.x
    g = target.pdf
    gx = target.x
    
    w1,w2 = wasser(source,target,'W12') # calculate Wasserstein 1 and 2 distances
    s1,H1,T1f2g = wasser_find_optplan(source,target,w1,distfunc='W1') # calculate optimal plan from Wassertein distance
    s2,H2,T2f2g = wasser_find_optplan(source,target,w2,distfunc='W2') # calculate optimal plan from Wassertein distance
    tf = np.zeros((len(f)+1,len(f)))
    tg = np.zeros((len(f)+1,len(g)))
    tf[0] = f
    for i in range(len(f)):
        tg[i+1] = tg[i] +  H1[i]
        tf[i+1] = tf[i]
        tf[i+1][:i+1] = 0.
    
    if(frames<=0 or frames >=len(f)+1): frames =len(f)
    snapshots = np.linspace(0,len(f),frames,dtype=int)
    
    if(frames <=4):
        cols = frames
        xl,yl = 10,4
    else:
        cols = 4
        xl,yl = 16,16
    rows = frames // cols
    if(rows==1):yl=4
    dxf = np.mean(np.abs(fx[1:]-fx[:-1]))
    dxg = np.mean(np.abs(gx[1:]-gx[:-1]))
    dxg = dxf

    fig1, axs = plt.subplots(rows, cols, figsize=(xl,yl), sharex=True, sharey=True)
    axs = trim_axs(axs, frames)
    #plt.title('Ricker wavelets optimal transport')
    if(plotsum):
        for ax, case in zip(axs, snapshots):
            ax.bar(fx, tf[case]+tg[case],width=dxf,color='b')
    else:
        for ax, case in zip(axs, snapshots):
            ax.bar(fx, tf[case],width=dxf,color='b')
            ax.bar(gx, tg[case],width=dxg,color='g',edgecolor='darkgreen',linewidth=0.2)
            ax.bar(fx, tf[case],width=dxf,edgecolor='b',linewidth=0.2)

    plt.plot(t,np.zeros(len(t)),'k-',lw=0.5)
    plt.suptitle('Optimal transport between initial and final PDFs')
    plt.show()

    fig1, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)
    ax1.bar(fx,f,width=dxf,edgecolor='b',linewidth=0.2)
    ax1.set_title(' f')
    ax2.bar(gx,g,width=dxg,color='g',edgecolor='darkgreen',linewidth=0.2)
    ax2.set_title(' g')
    plt.suptitle('Initial and final PDFs')
    plt.show()
    
    return snapshots
  
def plotOT1D(source,
             target,
             filename='Null',
             returnplan=False):

    f = source.pdf
    g = target.pdf
    fx  = source.x
    gx = target.x
    cf = source.cdf
    cg = target.cdf
    
    nx=10000                         
    start = np.min((fx[0],gx[0]))+0.5*(np.max((fx[-1],gx[-1])-np.min((fx[0],gx[0]))))/nx 
    stop = np.max((fx[-1],gx[-1]))-0.5*(np.max((fx[-1],gx[-1])-np.min((fx[0],gx[0]))))/nx 
    x = np.linspace(start,stop,nx)   # x-axis evaluation points for plot
    t = np.linspace(0,1,nx)          # t-axis evaluation points for plot

    F = np.dot(f,np.transpose(list(map(lambda x: np.heaviside(x-fx,0.0), x)))) #cdf of f
    G = np.dot(g,np.transpose(list(map(lambda x: np.heaviside(x-gx,0.0), x)))) #cdf of g

    Fp = np.append(F,1.0) # tidy up upper end point for neatness of plot
    Gp = np.append(G,1.0) # tidy up upper end point for neatness of plot
    xp = np.append(x,np.max((fx[-1],gx[-1])))
    Fp = F
    Gp = G
    xp = x

    dfx = fx[1:]-fx[:-1] 
    IF = fx[0] + np.dot(dfx,np.transpose(list(map(lambda x: np.heaviside(x-cf[:-1],0.0), t))))
    dgx = gx[1:]-gx[:-1]
    IG = gx[0] + np.dot(dgx,np.transpose(list(map(lambda x: np.heaviside(x-cg[:-1],0.0), t))))
    iFGdiff = IF-IG # Difference in inverse Cummulative functions
    IGF = gx[0] + np.dot(dgx,np.transpose(list(map(lambda x: np.heaviside(x-cg[:-1],0.0), F))))
    xmIFGsq = np.multiply(x - IGF,x - IGF)

    plotWasser(xp,Fp,Gp,t,IF,IG,x,IGF,xmIFGsq,iFGdiff,filename=filename)

    if(returnplan):
        return x,IGF
    return 

# default testing
    
if __name__ == "__main__":
    plot = False # perform plots or not
    plot = True # perform plots or not

    f = np.array([0.0,0.0,0.7,0.1,0.2,0.0]) # fixed choice masses
    g = np.array([0.0,0.0,0.2,0.2,0.6,0.0])
    n = len(f) # number of points

    f = np.array([0.0,0.0,0.7,0.1,0.2,0.0]) # fixed choice masses
    g = np.array([0.0,0.0,0.2,0.2,0.3,0.3,0.0]) # different number of g masses
    n = len(f) # number of points
    fx = np.linspace(3,2*n+2,n)   # regular mass locations with non unit spacings and different numbers of data
    gx = np.linspace(3,2*n+2,n+1) # regular mass locations

    f = np.array([0.2,0.5,0.1,0.2]) # fixed choice masses
    g = np.array([0.1,0.2,0.2,0.5])
    g = np.copy(f)
    n = len(f) # number of points
    n = 6 # number of points
    f = np.random.rand(n)  # random masses
    g = np.random.rand(n)  # random masses

    fx = np.linspace(3,2+n,n) # regular mass locations with unit spacings 
    gx = np.linspace(3,2+n,n) # regular mass locations

    fx = np.linspace(3,2*n+2,n) # regular mass locations with non unit spacings 
    gx = np.linspace(3,2*n+2,n) # regular mass locations
    gx = np.linspace(3,3*n+2,n) # regular mass locations with different spacing from fx

    gx = gx+4 # displace second mass locations

    fg = np.random.rand(n)     # random masses    
    
    source = OTpdf((f,fx)) # set up source object
    target = OTpdf((g,gx)) # set up target object
                   
    f/= np.sum(f)           # normalize PDFs and calculate cummulative sums
    cf = np.cumsum(f)
    g/= np.sum(g)
    cg = np.cumsum(g)

    # write out summary of test problem
    regloc = True
    print(' Test problem: \n','   Equal numbers f and g data     :',len(f)==len(g)
                             ,'\n    Regular f and g data locations :',regloc)
    if(fx[0]==gx[0]):
        print('    f and g distributions start at the same point\n')
    else:
        print('    f and g distributions start at different points\n')
    nx=10000                         
    start = np.min((fx[0],gx[0]))+0.5*(np.max((fx[-1],gx[-1])-np.min((fx[0],gx[0]))))/nx 
    stop = np.max((fx[-1],gx[-1]))-0.5*(np.max((fx[-1],gx[-1])-np.min((fx[0],gx[0]))))/nx 
    x = np.linspace(start,stop,nx)   # x-axis evaluation points for plot
    t = np.linspace(0,1,nx)          # t-axis evaluation points for plot

    F = np.dot(f,np.transpose(list(map(lambda x: np.heaviside(x-fx,0.0), x)))) #cdf of f
    G = np.dot(g,np.transpose(list(map(lambda x: np.heaviside(x-gx,0.0), x)))) #cdf of g

    Fp = np.append(F,1.0) # tidy up upper end point for neatness of plot
    Gp = np.append(G,1.0) # tidy up upper end point for neatness of plot
    xp = np.append(x,np.max((fx[-1],gx[-1])))
    Fp = F
    Gp = G
    xp = x

    dfx = fx[1:]-fx[:-1] 
    IF = fx[0] + np.dot(dfx,np.transpose(list(map(lambda x: np.heaviside(x-cf[:-1],0.0), t))))
    dgx = gx[1:]-gx[:-1]
    IG = gx[0] + np.dot(dgx,np.transpose(list(map(lambda x: np.heaviside(x-cg[:-1],0.0), t))))
    iFGdiff = IF-IG # Difference in inverse Cummulative functions
    IGF = gx[0] + np.dot(dgx,np.transpose(list(map(lambda x: np.heaviside(x-cg[:-1],0.0), F))))
    xmIFGsq = np.multiply(x - IGF,x - IGF)

    if (plot): plotWasser(xp,Fp,Gp,t,IF,IG,x,IGF,xmIFGsq,iFGdiff)

    
    W1est,W2est = wasserNumInt(source,target)
    print(' my W1 distance from f to g by numerical integration along t',W1est)
    print(' my W2 distance from f to g by numerical integration along t',W2est)
    
    dx = fx[1]-fx[0]
    Fc = np.cumsum(f) # points on the cummulative distribution of f
    Gc = np.cumsum(g) # points on the cummulative distribution of g
    if(len(f)==len(g)):
        W1 = dx*np.linalg.norm((Fc-Gc),ord=1) # 1-Wassertein metric using exact solution written as a norm, 
#                                       only for regularly spaced data at same locations between f and g
        print(' my W1 distance from f to g using simple formula for regularly spaced data',W1)
    else:
        print(' Can not use my simple W1 formula because data are different lengths')


# This method assumes the data domain is (0,1) and hence regular separation is dx = 1/(n-1)
# need to convert to domain (0,(n-1)dx)
    if(len(f)==len(g)):
        mapping,cost = _optimaltransport(source,target)
        W2monge = float(cost)*((n-1)*dx)**2
        print(' W2 from Solution of Monge-Ampere optimal transport equation',W2monge)
        T2f2g_ot = np.dot(mapping.T,np.diag(1/f)) # calculate transport matrix from f to g from heat map
        T2g2f_ot = np.dot(np.diag(f),np.dot(T2f2g_ot.T,np.diag(1/g))) # calculate transport matrix from g to f
        print(' Error in W2 transport plan of Monge-Ampere solution from f to g:',np.linalg.norm(np.dot(T2f2g_ot,f)-g))
        print(' Error in W2 transport plan of Monge-Ampere solution from g to f:',np.linalg.norm(np.dot(T2g2f_ot,g)-f))
    else:
        print(' Can not use Monge-Ampere OT routine because data are different lengths')

    if(False): # my attempt to evaluate int (x-G_inv(F(x))^2 f(x) dx
        Fform = np.dot(f,np.transpose(list(map(lambda x: np.heaviside(x-fx,0.0), fx)))) #cdf of f
        IGFform = gx[0] + np.dot(dgx,np.transpose(list(map(lambda x: np.heaviside(x-cg,0.0), Fform))))
        W2estform = np.dot(f,np.multiply(fx - IGFform,fx - IGFform))
        print(' W2 from formulae assuming 0.0 for heaviside',W2estform)
        Fform = np.dot(f,np.transpose(list(map(lambda x: np.heaviside(x-fx,0.5), fx)))) #cdf of f
        IGFform = gx[0] + np.dot(dgx,np.transpose(list(map(lambda x: np.heaviside(x-cg,0.5), Fform))))
        W2estform = np.dot(f,np.multiply(fx - IGFform,fx - IGFform))
        print(' W2 from formulae assuming 0.5 for heaviside',W2estform)
        Fform = np.dot(f,np.transpose(list(map(lambda x: np.heaviside(x-fx,1.0), fx)))) #cdf of f
        IGFform = gx[0] + np.dot(dgx,np.transpose(list(map(lambda x: np.heaviside(x-cg,1.0), Fform))))
        W2estform = np.dot(f,np.multiply(fx - IGFform,fx - IGFform))
        print(' W2 from formulae assuming 1.0 for heaviside',W2estform)
    
# test with linear programming

    if(len(f)==len(g) and True):
        W1lin,H1lin = Wasser_LinProg(source,target,distfunc='W1') # compute W1,W2 and transport maps
        W2lin,H2lin = Wasser_LinProg(source,target,distfunc='W2') # compute W1,W2 and transport maps
        T1f2glin = np.dot(H1lin.T,np.diag(1/f))
        T2f2glin = np.dot(H2lin.T,np.diag(1/f))
        T1g2flin = np.dot(np.diag(f),np.dot(T1f2glin.T,np.diag(1/g))) # Tf2g.f = g, Tg2f.g = f
        T2g2flin = np.dot(np.diag(f),np.dot(T2f2glin.T,np.diag(1/g))) # Tf2g.f = g, Tg2f.g = f
        print(' W1 from linear programming',W1lin)
        print(' W2 from linear programming',W2lin)
        print(' Error in W1 LP transport plan from f to g:',np.linalg.norm(np.dot(T1f2glin,f)-g))
        print(' Error in W1 LP transport plan from g to f:',np.linalg.norm(np.dot(T1g2flin,g)-f))
        print(' Error in W2 LP transport plan from f to g:',np.linalg.norm(np.dot(T2f2glin,f)-g))
        print(' Error in W2 LP transport plan from g to f:',np.linalg.norm(np.dot(T2g2flin,g)-f))
    else:
        print(' Can not use linear programming routine because data are different lengths')

# my second exact formulae for Wp directly from notes
    
    Wasser1,Wasser2,H = wasser(source,target,'W12',returnplan=True)
    print(' W1 from compact formulae for exact integration along t axis',Wasser1)
    print(' W2 from compact formulae for exact integration along t axis',Wasser2)
    s1,H1 = wasser_find_optplan(source,target,Wasser1,distfunc='W1') # calculate optimal plan from Wassertein distance
    s2,H2 = wasser_find_optplan(source,target,Wasser2,distfunc='W2') # calculate optimal plan from Wassertein distance
    T1f2g = np.dot(H1.T,np.diag(1/f))
    T2f2g = np.dot(H2.T,np.diag(1/f))
    
    if(s1):
        T1g2f = np.dot(np.diag(f),np.dot(T1f2g.T,np.diag(1/g))) # Tf2g.f = g, Tg2f.g = f
        print(' Error in W1 transport plan from f to g:',np.linalg.norm(np.dot(T1f2g,f)-g))
        print(' Error in W1 transport plan from g to f:',np.linalg.norm(np.dot(T1g2f,g)-f))
    if(s2):
        T2g2f = np.dot(np.diag(f),np.dot(T2f2g.T,np.diag(1/g))) # Tf2g.f = g, Tg2f.g = f
        print(' Error in W2 transport plan from f to g:',np.linalg.norm(np.dot(T2f2g,f)-g))
        print(' Error in W2 transport plan from g to f:',np.linalg.norm(np.dot(T2g2f,g)-f))

    if(False):
        snaps = plot_optimal_transform_frames(source,target,100)
        
    print(' Difference between optimal plan from compact formulae and linear solver',np.mean(H-H2))
    print(' Difference between optimal plan from compact formulae and LP solver',np.mean(H-H2lin))
    
    W2pot,H2pot = wasserPOT(source,target,returnplan=True)
    W1pot,H1pot = wasserPOT(source,target,distfunc='W1',returnplan=True)
    print(' W1 from POT linear programming',W1pot)
    print(' W2 from POT linear programming',W2pot)
    print(' Difference between optimal plan from compact formulae and simplex POT solver for W2',np.mean(H-H2pot))

