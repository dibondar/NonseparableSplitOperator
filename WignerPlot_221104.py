#from numba import njit # compile python
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt # plotting facility
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib import cm
from matplotlib import gridspec
import numpy as np
import os, datetime, math
import time
import scipy.fftpack as fftpack
from scipy.fftpack import fft, ifft

def my_function():
    print("Hello World from Wigner plot")


# psi  = ((np.array(wavefunctions)[jj,:])*einsen)
# psiC = ((np.array(wavefunctions)[jj,:])*einsen).conj()
# W  = np.copy(psi)
# WC =  np.copy(psiC)

# Xn = XCoord.size
# Pn = PCoord.size
# pMax = abs(PCoord[0])
# xMax = abs(XCoord[0])

# PCoord = PCoord.reshape(PCoord.size,1)
# XCoord = XCoord.reshape(1,XCoord.size)

# #phase_shear_y  = np.exp(+1j * XCoord * PCoord * Xn * Pn  /xMax**2 / pMax**2  * np.pi*392968/1000000 )
# # X_gridDIM OK    # P_gridDIM OK    # pMax OK    # xMax OK

# phase_shear_y  = np.exp(+1j * XCoord * PCoord * Xn * Pn /xMax**2 /pMax**2 * np.pi**2/8 )
# phase_shear_ym = phase_shear_y.conj()

def W__by_operator(wavefunctions,jj,rescaler,einsen,dP,P_gridDIM,XCoord,PCoord):
    """ Wigner distribution determined by roll-over loop
    """
    
    psi  = ((np.array(wavefunctions)[jj,:])*einsen)
    psiC = ((np.array(wavefunctions)[jj,:])*einsen).conj()
    W  = np.copy(psi)
    WC =  np.copy(psiC)

    Xn = XCoord.size
    Pn = PCoord.size
    pMax = abs(PCoord[0])
    xMax = abs(XCoord[0])

    PCoord = PCoord.reshape(PCoord.size,1)
    XCoord = XCoord.reshape(1,XCoord.size)

    #phase_shear_y  = np.exp(+1j * XCoord * PCoord * Xn * Pn  /xMax**2 / pMax**2  * np.pi*392968/1000000 )
    # X_gridDIM OK    # P_gridDIM OK    # pMax OK    # xMax OK

    phase_shear_y  = np.exp(+1j * XCoord * PCoord * Xn * Pn /xMax**2 /pMax**2 * np.pi**2/8 )
    phase_shear_ym = phase_shear_y.conj()

    W = fft(W, axis=1, overwrite_x=True)
    W = fftpack.fftshift(W,axes=1)
    W *= phase_shear_ym
    W = fftpack.ifftshift(W,axes=1)
    W = ifft(W, axis=1, overwrite_x=True)
    
    WC = fft(WC, axis=1, overwrite_x=True)
    WC = fftpack.fftshift(WC,axes=1)
    WC *= phase_shear_y
    WC = fftpack.ifftshift(WC,axes=1)
    WC = ifft(WC, axis=1, overwrite_x=True)

    W = fftpack.ifftshift(fftpack.ifft(fftpack.fftshift(W*WC,axes=0),axis=0),axes=0) / dP

    # perform the checks
    assert np.linalg.norm(W.imag.reshape(-1), np.infty), "there should be no imaginary part"
    assert (W.real <= 2.).all() and (W.real > -2.).all(), "The Cauchy-Schwarz inequality is violated"

    return W#.real

    
def W_carved_by_hand(wavefunctions,jj,rescaler,einsen,dP,P_gridDIM):
    """ Wigner distribution determined by roll-over loop
    """
    psi = ((np.array(wavefunctions)[jj,:])*einsen)
    Wdist = np.copy(psi)

    for rollParameter in range(0,P_gridDIM):
        smoothRollover = (rollParameter-P_gridDIM/2)*rescaler
        discreteRollover = int(round(smoothRollover))
        fractionRoll = smoothRollover-discreteRollover
        DRollover = int(np.sign(fractionRoll))
        
        Wdist[rollParameter][:] = (1-abs(fractionRoll))*(np.roll(psi[rollParameter,:],-discreteRollover)).conj() * (np.roll(psi[rollParameter,:],discreteRollover)) + abs(fractionRoll)*(np.roll(psi[rollParameter,:],-discreteRollover-DRollover)).conj() * (np.roll(psi[rollParameter,:],discreteRollover+DRollover))

        #This simpler assignment gives 'ghost images' in W!
        #Wdist[rollParameter][:] = np.roll(psi[rollParameter,:],-discreteRollover).conj() * np.roll(psi[rollParameter,:],discreteRollover)

    Wdist = fftpack.ifftshift(fftpack.ifft(fftpack.fftshift(Wdist,axes=0),axis=0),axes=0) / dP
    return Wdist.real


def multicolor_ylabel(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,**kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn
    https://stackoverflow.com/questions/33159134/matplotlib-y-axis-label-with-multiple-colors#33162465
    """
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors[::-1]) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.05*1, -0.05), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)


def Projection_variableDirection(W,direction,dX,dP):
    if direction=='x':
        return (W).sum(axis=0)*dP
    if direction=='p':
        return (W).sum(axis=1)*dX
    else:
        print('Neither "x" nor "p" projection?')
        sys.exit()

def Hamiltonian_t(W,time,potentialVec,posVec,momVec,dX,dP,nonLinearWeight,NLSEexp,KerrCase=False):  
    '''Time-dependent Hamiltonian, term 0*XoldCoord to broadcast matrix for V(x)=0'''
    
    probX =  Projection_variableDirection(W,'x',dX,dP)
    if KerrCase == True:
        #print("kerr case = true")
        return (momVec**2/2 + posVec**2/2 )**2
    else:
        #print("kerr case = false")
        return momVec**2/2 + 2*nonLinearWeight/(NLSEexp+2)*probX**(NLSEexp/2) + potentialVec  


def PlotWignerDistr_Projections_Title(W,time,XcoordVec,PcoordVec,zoomFactorX,zoomFactorP,bondarsNewPropagator,title,potVec,gamma,smallProb=0,ProbMax=-1,numberOfPeaks=1,NLSEexp=2,KerrCase=False):
    '''zoomFactor = 1 ~ no zoom, or above 1: zoomed in
     'unbalanced' figsize, e.g. (8,5), gives (buggy) 'ghost grids' in image 
    ProbMax = max(harmonic_osc.wavefunction*harmonic_osc.wavefunction.conj()) = Maximum of density P(x)
    '''

    lsize=30 #LabelSize
    fsize=40 #FontSize

    P_gridDIM,X_gridDIM = W.shape

    XMin = XcoordVec[0]
    XMax = XcoordVec[-1]
    dX = (XMax-XMin)/(X_gridDIM-1)

    PMin = PcoordVec[0]
    PMax = PcoordVec[-1]
    dP = (PMax-PMin)/(P_gridDIM-1)

    
    x_min =  XMin/zoomFactorX
    x_max =  XMax/zoomFactorX
    p_min =  PMin/zoomFactorP
    p_max =  PMax/zoomFactorP

    if bondarsNewPropagator == False:
        W=fftpack_fftshift(W, bondarsNewPropagator)

    Pr_x = Projection_variableDirection(W,'x',dX,dP)
    Pr_p = Projection_variableDirection(W,'p',dX,dP)

    print("############################################# Kerrcase value in plot routine:",KerrCase)
    EnergyLandscape = Hamiltonian_t(W,time,potVec,XcoordVec,PcoordVec[:,np.newaxis],dX,dP,gamma,NLSEexp,KerrCase)

    EnergyLandscape = EnergyLandscape[round(P_gridDIM*(1-1/zoomFactorP)/2):round(P_gridDIM*(1+1/zoomFactorP)/2),\
                  round(X_gridDIM*(1-1/zoomFactorX)/2):round(X_gridDIM*(1+1/zoomFactorX)/2)]
    
    W=W[round(P_gridDIM*(1-1/zoomFactorP)/2):round(P_gridDIM*(1+1/zoomFactorP)/2),\
                  round(X_gridDIM*(1-1/zoomFactorX)/2):round(X_gridDIM*(1+1/zoomFactorX)/2)]

    fig=plt.figure(1,figsize=((15,15)))

    gs = gridspec.GridSpec(3,3, width_ratios=[0.0001,0.475,0.1],height_ratios=[0.0001,2,0.4]) 
    ax = plt.subplot(gs[1,1])
    
    global_color_max = W.max() *0 + 1./3.14      
    global_color_min = W.min() *0 - 1./3.14
    
    axp = ax.imshow( W ,origin='lower',interpolation='none',
                    extent=[ x_min , x_max, p_min, p_max]
                    ,vmin=-global_color_max,vmax=global_color_max, cmap=cm.seismic)

    cbaxes = fig.add_axes([0.05-0.04, 0.3, 0.007, 0.5])  # [xo,yo,Dx,Dy] position for the colorbar
    cb1 = plt.colorbar(axp, cax = cbaxes)
    cb1.ax.tick_params(labelsize=int(lsize/1.0))

    ax.set_aspect(XMax/PMax*zoomFactorP/zoomFactorX)
    ax.tick_params('x', colors='black',labelsize=lsize)
    ax.tick_params('y', colors='black',labelsize=lsize)
    
    plt.suptitle(title,color='black',fontsize=fsize-20)

    ax.set_ylabel('Momentum P', fontsize=fsize)
    ax.yaxis.tick_right()
    ax.set_xlabel('Position X', fontsize=fsize)
    ax.xaxis.set_label_position("top")

    ax.contour( EnergyLandscape ,
    np.arange(1, 2701, 55 ),origin='lower',extent=[x_min,x_max,p_min,p_max],
    linewidths=0.25,colors='k')
    
    ax1 = plt.subplot(gs[1,2], sharey=ax)
    ax1.plot(Pr_p,PcoordVec,'g')
#    ax1.set_xlabel('|ψ(p)|\u00b2', color='g',fontsize=fsize)
    ax1.set_xlabel(r'$|\tilde\psi(p)|^2$', color='g', fontsize=fsize)
    ax1.tick_params('x', colors='g',labelsize=lsize)
    ax1.tick_params('y', colors='g',labelsize=0) #suppress shared axis labels
    ax1.yaxis.grid(which="major", color='g', linestyle='-', linewidth=1)
    ax1.set_ylim([p_min,p_max])
    
    ax2 = plt.subplot(gs[2,1])    
#    ax2.set_ylabel(r'$|\psi(x)|^2$', color='b', fontsize=fsize)#, horizontalalignment='left')
    if ProbMax > 0: # if a positive argument is passed the y-axis is fixed to [0,ProbMax]
        ax2.axes.set_ylim([0,ProbMax])
        multicolor_ylabel(ax2,(r'$|\psi(x)|^2$',r'[%.3f]'%ProbMax),('b','r'),axis='y',size=fsize)#,weight='bold')
        oneline = np.copy(Pr_x)
        oneline.fill(smallProb)
        ax2.plot(XcoordVec,oneline,color='r', linestyle='dotted', linewidth=1.51,alpha=1.0)
        oneline.fill(smallProb*numberOfPeaks*2)
        ax2.plot(XcoordVec,oneline,color='r', linewidth=1.51,alpha=1.0)
    else:
        ax2.set_ylabel(r'$|\psi(x)|^2$', color='b', fontsize=fsize)#, horizontalalignment='left')
    
    ax2.yaxis.tick_right()
    ax2.plot(XcoordVec, Pr_x,'b',linewidth=2)

    ax2.tick_params('y', colors='b',labelsize=lsize)
    ax2.tick_params('x', colors='b',labelsize=0*lsize)
    ax2.set_xlim([x_min,x_max])
    
    ax.xaxis.grid(which="major", color='black', linestyle='-', linewidth=1)
    ax.yaxis.grid(which="major", color='gray', linestyle='-.', linewidth=1)
    ax1.yaxis.grid(which="major", color='gray', linestyle='-.', linewidth=1)
    ax2.xaxis.grid(which="major", color='black', linestyle='-', linewidth=1)

    return fig,ax



def PlotWignerProjNOTitle(W,time,XcoordVec,PcoordVec,zoomFactorX,zoomFactorP,bondarsNewPropagator,title,potVec,gamma,NLSEexp=2):
    '''zoomFactor = 1 ~ no zoom, or above 1: zoomed in'''
    
    lsize=20 #LabelSize
    fsize=20 #FontSize

    P_gridDIM,X_gridDIM = W.shape

    XMin = XcoordVec[0]
    XMax = XcoordVec[-1]
    dX = (XMax-XMin)/(X_gridDIM-1)

    PMin = PcoordVec[0]
    PMax = PcoordVec[-1]
    dP = (PMax-PMin)/(P_gridDIM-1)

    
    x_min =  XMin/zoomFactorX
    x_max =  XMax/zoomFactorX
    p_min =  PMin/zoomFactorP
    p_max =  PMax/zoomFactorP

    if bondarsNewPropagator == False:
        W=fftpack_fftshift(W, bondarsNewPropagator)

    Pr_x = Projection_variableDirection(W,'x',dX,dP)
    Pr_p = Projection_variableDirection(W,'p',dX,dP)

    EnergyLandscape = Hamiltonian_t(W,time,potVec,PcoordVec[:,np.newaxis],dX,dP,gamma,NLSEexp)

    EnergyLandscape = EnergyLandscape[round(P_gridDIM*(1-1/zoomFactorP)/2):round(P_gridDIM*(1+1/zoomFactorP)/2),\
                  round(X_gridDIM*(1-1/zoomFactorX)/2):round(X_gridDIM*(1+1/zoomFactorX)/2)]
    
    W=W[round(P_gridDIM*(1-1/zoomFactorP)/2):round(P_gridDIM*(1+1/zoomFactorP)/2),\
                  round(X_gridDIM*(1-1/zoomFactorX)/2):round(X_gridDIM*(1+1/zoomFactorX)/2)]

    fig=plt.figure(1,figsize=((15,15)))

    gs = gridspec.GridSpec(3,3, width_ratios=[0.0001,0.475,0.1],height_ratios=[0.0001,2,0.4]) 
    ax = plt.subplot(gs[1,1])
    
    global_color_max = W.max()       
    global_color_min = W.min() 
    
    axp = ax.imshow( W ,origin='lower',interpolation='none',
                    extent=[ x_min , x_max, p_min, p_max]
                    ,vmin=-global_color_max,vmax=global_color_max, cmap=cm.seismic)

    # AVOID color bar (with large tick labels)
    # cbaxes = fig.add_axes([0.09, 0.3, 0.01, 0.5]) #[left, bottom, width, height] out of {0..1}
    # cb1 = plt.colorbar(axp, cax = cbaxes)
    # cbaxes.yaxis.set_ticks_position('left')
    # cb1.ax.tick_params(labelsize=int(lsize/1.0))
    cbaxes = fig.add_axes([0.05, 0.3, 0.007, 0.5])  # [xo,yo,Dx,Dy] position for the colorbar
    cb1 = plt.colorbar(axp, cax = cbaxes)
    cb1.ax.tick_params(labelsize=int(lsize/1.5))

    ax.set_aspect(XMax/PMax*zoomFactorP/zoomFactorX)
    ax.tick_params('x', colors='black',labelsize=lsize)
    ax.tick_params('y', colors='black',labelsize=lsize)
    
    ax.set_ylabel('Momentum $p$', fontsize=fsize)
    ax.yaxis.tick_right()
    ax.set_xlabel('Position $x$', fontsize=fsize)
    ax.xaxis.set_label_position('top') 

    ax.contour( EnergyLandscape ,
    np.arange(1, 2701, 55 ),origin='lower',extent=[x_min,x_max,p_min,p_max],
    linewidths=0.25,colors='k')
    
    ax1 = plt.subplot(gs[1,2], sharey=ax)
    ax1.plot(Pr_p,PcoordVec,'g')
#    ax1.set_xlabel('|ψ(p)|\u00b2', color='g',fontsize=fsize)
    ax1.set_xlabel(r'$|\tilde\psi(p)|^2$', color='g', fontsize=fsize)
#    ax.set_ylabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}', fontsize=fsize)
    ax1.tick_params('x', colors='g',labelsize=lsize)
    ax1.tick_params('y', colors='g',labelsize=0) #suppress shared axis labels
    ax1.yaxis.grid(which="major", color='green', linestyle='-', linewidth=1)
    ax1.xaxis.grid(which="major", color='gray', linestyle='-', linewidth=1)
    plt.xticks([0, (round(10*Pr_p.max()-0.5))/10.0 ])
    ax1.set_ylim([p_min,p_max])
    
    ax2 = plt.subplot(gs[2,1], sharex=ax)
#    ax2.set_ylabel('|ψ($x$)|\u00b2', color='b',fontsize=fsize)
    ax2.set_ylabel(r'$|\psi(x)|^2$', color='b', fontsize=fsize)
    ax2.yaxis.tick_right()
    ax2.plot(XcoordVec, Pr_x,'b',linewidth=2)

    ax2.tick_params('y', colors='b',labelsize=lsize)
    ax2.tick_params('x', colors='b',labelsize=0*lsize)
    ax2.set_xlim([x_min,x_max])
    
    ax.xaxis.grid(which="major", color='black', linestyle='-', linewidth=1)
    ax.yaxis.grid(which="major", color='gray', linestyle='-.', linewidth=1)
    ax1.yaxis.grid(which="major", color='gray', linestyle='-.', linewidth=1)
    ax2.xaxis.grid(which="major", color='black', linestyle='-', linewidth=1)

    return fig,ax
