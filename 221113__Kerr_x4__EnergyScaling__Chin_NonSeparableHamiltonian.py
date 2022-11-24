#!/usr/bin/env/python3

# Code to implement Chin's 9-Exponentials propagation code for non-separable Hamiltonians, here: KERR ! 
# I added a modification using 7 Exponentials which needs the CorrectionPotential(x).
# This file is based on Max' code for the frames to study scaling behaviour.
# It shows the expected Energy-Scaling of -2 for an 0.05*1*x**4 oscillator using STRANG splitting

# Loading packages
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftpack
import scipy.integrate as integrate
import scipy.stats as stats
import time
start_time = time.time()
import WignerPlot_221104


# Try to reach 2nd Order using STRANG splitting
StrangSplitting = True # False #

# Kerr (NONseparable Hamiltonian)
KerrCase = True #False #

#extraMotto = '_X_Theta__To__Lambda_P_AND_Lambda_P__To__X_Theta__CHANGED_'
extraMotto = 'MaxsPropagator'
#extraMotto = 'OlesPropagator'

# Use 7 Exponentials (instead of 9)
exp7 = True #False #
if extraMotto == 'MaxsPropagator':
    exp7 = False

discretizationX = 512//8           # Number of GRID-points in x
discretizationP = discretizationX  # Number of GRID-points in p    
    
# time steps
list_dt_4_separableSystem = [0.05, 0.01]#, 0.001, 0.0005]#0.04, 0.01, 0.005]
#list_dt_4_Kerr = [0.01, 0.005]#, 0.001]#, 0.0001]#, 0.00002]
list_dt_4_Kerr = [0.0015, 0.00010]#, 0.00002]
    
# Initial state (x0,p0)
M_SQRT1_2 = 1./np.sqrt(2)
R = 4-1 # Distance from the origin
initial_offsets = [( R*M_SQRT1_2, R*M_SQRT1_2), 
                   ( R*1,         0), 
                   ( R*M_SQRT1_2,-R*M_SQRT1_2), 
                   ( 0,          -R*1), 
                   (-R*M_SQRT1_2,-R*M_SQRT1_2), 
                   (-R*1,         0), 
                   (-R*M_SQRT1_2, R*M_SQRT1_2), 
                   ( 0,           R*1)]

initial_offsets = [( R*M_SQRT1_2, R*M_SQRT1_2)]#,
#                    ( R*1,         0)]#, 
#                    # ( R*M_SQRT1_2,-R*M_SQRT1_2)]

# where to plot the above
subplot_indices = [(0,2),(1,2),(2,2),(2,1),(2,0),(1,0),(0,0),(0,1)]
# energy data storage for plotting later
energy_data = [[[],[],[]],[[],[],[]],[[],[],[]]]
overlap_data = [[[],[],[]],[[],[],[]],[[],[],[]]]

# Specifying parameters in atomic units
mass = 1.             # particle's mass
k = 1.                # Harmonic oscillator spring constant
amplitudeX = 9.0     # xrange = [-amplitudeX,amplitudeX]
amplitudeP = 16.0-7.0     # prange = [-amplitudeP,amplitudeP]
hbar = 1              # Planck constant
L = 1.                # Kerr lambda
period = np.pi/L if KerrCase else 2*np.pi


# Defining x vector, p vector, theta vector and lambda vector
# containing the respective range of values
xvector = \
    np.linspace(-np.abs(amplitudeX), np.abs(amplitudeX*(1.-2./discretizationX)), discretizationX)
pvector = \
    np.linspace(-np.abs(amplitudeP), np.abs(amplitudeP*(1.-2./discretizationP)), discretizationP)
thetavector = fftpack.fftshift( \
    2.*np.pi*fftpack.fftfreq(pvector.size, pvector[1] - pvector[0]))
lambdavector = fftpack.fftshift( \
    2.*np.pi*fftpack.fftfreq(xvector.size, xvector[1] - xvector[0]))
# Defining X, P, Lambda and Theta grids
Theta, X = np.meshgrid(thetavector,  xvector)
P,Lambda = np.meshgrid(pvector, lambdavector)


if StrangSplitting == True:
    halveStep = 0.5
else:
    halveStep = 1.

if exp7 == True:
    blankOutFactor=0
else:
    blankOutFactor=1


# Defining the kinetic and potential energies and Hamiltonian
if KerrCase:
    motto=f"Kerr__{extraMotto}_{'7_Exp' if (exp7 and extraMotto == 'OlesPropagator') else '9_Exp'}"
    dirLabel=motto
    def Potential(x):
        return (1./2)*k*x**2

    if exp7 == True and  extraMotto == 'OlesPropagator':
        def CorrectionPotential(x):
            return - x**6 /(9.0 * np.sqrt(2.0) * t2**3)
    else:
        def CorrectionPotential(x):
            return 0    

    def Potential2(x):
        return (L**2)*Potential(x)**2 + CorrectionPotential(x) 
    def Kinetic(p):
        return (1./(2*mass))*p**2
    def Kinetic2(p):
        return (L**2)*Kinetic(p)**2
    def Hamiltonian(x,p):
        return (L**2)*(Kinetic(p) + Potential(x))**2
else:
    def Kinetic(p):
        return 0.5*(1./mass)*p**2
    def Potential(x):
#        return 0.5*k*x**2  
        return 0.05*k*x**4
    motto=f"V(x)=0.05*1*x**4_{extraMotto}_"
    dirLabel='V(x4)'
    def Hamiltonian(x,p):
        return Kinetic(p) + Potential(x)

#    _{[\psi_0({x_off:.3f},{p_off:.3f})]}

# Integrate
def IntegrateXP(F):
    return integrate.simps(integrate.simps(F.real, pvector), xvector)
# Plot Wigner Distribution
def Plot_Wigner(t, dt, plots_dir, W):
    zero=0 # I am meant to pass only the V(xvector) instead of Hamiltonian(xvector,zero*pvector)

    WignerPlot_221104.PlotWignerDistr_Projections_Title(np.transpose(W.real),t*dt,xvector,pvector,1.8,1.8,\
        True,(r"$W_{[\Psi_0(%6.3f,%6.3f)]}(t=%6.3f)$" % (x_off,p_off,t*dt)),Hamiltonian(xvector,zero*pvector),0,0,0,1,0,KerrCase)
    
    filename = f"t_{t*dt:3.3f}_[{discretizationX}x{discretizationP}]_R_{R}_{motto}_{'_2ndStrang' if StrangSplitting else ''}_psi({x_off:.3f},{p_off:.3f})_dt{dt}"
    plt.savefig(f'{plots_dir}/{t}__{filename}.pdf')
    plt.close('all')
# Representation Transformations
#    (p <-> theta)
# W(x,p) - B(x,theta)
# |             | (x <-> lambda)
# A(lambda,p) - Z(lambda,theta)
#

# #
# ## Do the FFT.s commute? No!
# #
# Plot_Wigner(t, dt, plots_dir, fftpack.ifft(fftpack.fft(W, overwrite_x=True, axis=0),overwrite_x=True, axis=1) - fftpack.fft(fftpack.ifft(W, overwrite_x=True, axis=1),overwrite_x=True, axis=0))

# fftpack.ifft(fftpack.fft(W, overwrite_x=True, axis=0),overwrite_x=True, axis=1) -
# fftpack.fft(fftpack.ifft(W, overwrite_x=True, axis=1),overwrite_x=True, axis=0))

def X_P__To__X_Theta(W):
    return fftpack.fft(W, overwrite_x=True, axis=1)


def X_Theta__To__Lambda_P_old(B):
    Z = fftpack.fft(B, overwrite_x=True, axis=0)
    return fftpack.ifft(Z, overwrite_x=True, axis=1)
def X_Theta__To__Lambda_P(Bprime):
    ''' switched FFT trafos '''
    Zprime = fftpack.ifft(Bprime, overwrite_x=True, axis=1)
    return fftpack.fft(Zprime, overwrite_x=True, axis=0)


def Lambda_P__To__X_Theta_old(A):
    W = fftpack.ifft(A, overwrite_x=True, axis=0)
    return fftpack.fft(W, overwrite_x=True, axis=1)
def Lambda_P__To__X_Theta(A):
    W = fftpack.fft(A, overwrite_x=True, axis=1)
    return fftpack.ifft(W, overwrite_x=True, axis=0)


def Lambda_P__To__X_P(A):
    return fftpack.ifft(A, overwrite_x=True, axis=0)
# Energy Landscape
Energy_Landscape = np.array(Hamiltonian(X,P))

# Begin of Frame LOOP
for idx,(x_off,p_off) in enumerate(initial_offsets):
    # subplot index
    i,j = subplot_indices[idx]
    # Specifying the initial state
    Winit = (1./np.pi)*np.exp(-(X-x_off)**2-(P-p_off)**2) + 0j
    Einit = IntegrateXP(Energy_Landscape*Winit)
    WoWinit = 2*np.pi*IntegrateXP(Winit*Winit)

    
    # Begin of dt- LOOP

    # Range of dt to test for, kerr case requires smaller timesteps for meaningful tests
#    for dt in ([0.04, 0.02, 0.01, 0.0075, 0.005] if not KerrCase else [0.01, 0.006, 0.001, 0.0006, 0.0001, 0.00005]):
    for dt in (list_dt_4_separableSystem if not KerrCase else list_dt_4_Kerr):
#    for dt in ([0.04, 0.01, 0.005] if not KerrCase else [0.01, 0.005, 0.001, 0.0005, 0.0001]):

        timeStepsN = round(period/dt) + 1
        # output dir for intermediate plots
                
        plots_dir = f"wigframes_[{discretizationX}x{discretizationP}]_R_{R}_{motto}_dt{list_dt_4_Kerr  if KerrCase else list_dt_4_separableSystem}{'_2ndStrang' if StrangSplitting else ''}/psi({x_off:.3f},{p_off:.3f})/{dt}/"
        os.makedirs(plots_dir, exist_ok=True)
        # Defining the propagator factors
        if KerrCase:
            # Defining Chin's parameters for the non-separable integrator
            # eps = np.cbrt(dt)             # This requires a very small dt for stability
            t2 = -1.*np.cbrt(6)            #2 Free parameter (using value '+10.0' gives poor performance)
            t1 = -t2                       #1 
            v1 = +1./(t2**2)               #2
            v2 = -v1/2                     #1
            v0 = -2*(v1 + v2)              #1

            ### Propagator factor for X^4
            potentialPropagatorFactor = fftpack.ifftshift( \
                np.exp(-halveStep*1j*dt*(Potential2(X - hbar* Theta/2.) - Potential2(X + hbar* Theta/2.))), axes=(1,))
            ### Propagator factor for P^4
            kineticPropagatorFactor = fftpack.ifftshift( \
                np.exp(-1j*dt*(  Kinetic2(P + hbar*Lambda/2.) -   Kinetic2(P - hbar*Lambda/2.))), axes=(0,))

            
            def factor(func, mult, r, s, t):
                '''Helper function for defining the following propagation factors'''
                eps = np.cbrt(dt)
                factorTwo = np.cbrt(2) ## compensation for removal of second cross term P^2X^2
                return fftpack.ifftshift( \
                    np.exp(-1j*eps*factorTwo*mult*(func(r - hbar* s/2.) - func(r + hbar* s/2.))), axes=(t,))

            
            ### Effective Potentials for creation of Propagator factors for non-separable terms X^2P^2
            def Cross_Eff_T2(p): # x^2 p^2
                return ((L*k)/24)*p**4 + (L/2)*p
                        
            def Cross_Eff_V2(x): # x^2 p^2
                return (1./2)*(np.sqrt(L/mass)/np.sqrt(2))*x**2


            # def Cross_Eff_T2(p): 
            #     return ((L*k)) /  (2.0*np.sqrt(2.0)) *p**4                        
            # def Cross_Eff_V2(x):
            #     return (np.sqrt(L/mass)) / 12.0 * x**4
            notAllowed = 0
            
            effKineticPropagatorFactor1_2   = factor(Cross_Eff_T2, v2, P, -Lambda, 0)
            effKineticPropagatorFactor2_2   = factor(Cross_Eff_T2, v1, P, -Lambda, 0)
            effKineticPropagatorFactor3_2   = factor(Cross_Eff_T2, v0, P, -Lambda, 0)
            effKineticPropagatorFactor4_2   = factor(Cross_Eff_T2, v1, P, -Lambda, 0)
            effKineticPropagatorFactor5_2   = factor(Cross_Eff_T2, v2, P, -Lambda, 0) + notAllowed * kineticPropagatorFactor

            effPotentialPropagatorFactor1_2 = factor(Cross_Eff_V2, t2, X, Theta, 1)
            effPotentialPropagatorFactor2_2 = factor(Cross_Eff_V2, t1, X, Theta, 1)
            effPotentialPropagatorFactor3_2 = factor(Cross_Eff_V2, t1, X, Theta, 1)
            effPotentialPropagatorFactor4_2 = factor(Cross_Eff_V2, t2, X, Theta, 1)
            
        else:
            potentialPropagatorFactor = fftpack.ifftshift( \
                np.exp(-halveStep*1j*dt*(Potential(X - hbar* Theta/2.) - Potential(X + hbar* Theta/2.))), axes=(1,))
            kineticPropagatorFactor = fftpack.ifftshift( \
                np.exp(-1j*dt*(  Kinetic(P + hbar*Lambda/2.) -   Kinetic(P - hbar*Lambda/2.))), axes=(0,))


        def SeparableHamiltonianPropagatorOneStep(W):
                # Transforming to the X-Theta representation
                B = X_P__To__X_Theta(W)
                # Applying the factor associated with the potential
                B *= potentialPropagatorFactor
                # Transforming to the Lambda-P representation
                A = X_Theta__To__Lambda_P(B)
                # Applying the factor associated with the kinetic energy
                A *= kineticPropagatorFactor
                # returning to the X-P representation
                W = Lambda_P__To__X_P(A)
                if StrangSplitting:
                        # Transforming to the X-Theta representation
                        B = X_P__To__X_Theta(W)
                        # Apply potential factor
                        B *= potentialPropagatorFactor
                        W = Lambda_P__To__X_P(X_Theta__To__Lambda_P(B))    
                return W

        #
        ##
        ### BEGIN: of creation of Oles Propagators
        ##
        #

        def KineticStep(W,expOp,unusedSlot):
            '''Kinetic part evolution. W:WignerDistribution; expOp: evolutionOperator'''
            #W = FFT( W, True, 1) #>#np.fft.fft( W, axis = 1 )
            W = fftpack.fft(W, overwrite_x=True, axis=0)
            W *=expOp
            #W = FFT( W, False, 1) #>#np.fft.ifft( W, axis = 1 )
            W = fftpack.ifft(W, overwrite_x=True, axis=0)
            return W

        def PotentialStep(W,expOp,unusedSlot):
            '''Potential part evolution. W:WignerDistribution; expOp: evolutionOperator'''
            #W = FFT( W, True, 0)    #>#W = np.fft.fft( W, axis = 0 )
            W = X_P__To__X_Theta(W)  #>#fftpack.fft(W, overwrite_x=True, axis=1)
            W *=expOp
            #W = FFT( W, False, 0) #W = np.fft.ifft( W, axis = 0 )
            W = fftpack.ifft(W, overwrite_x=True, axis=1)
            return W


        def SeparableHamiltonianPropagatorOneStep_OLE(W):
            unusedSlot = 0

            W = PotentialStep(W,potentialPropagatorFactor,unusedSlot)                
            W = KineticStep(W,kineticPropagatorFactor,unusedSlot)

            if StrangSplitting:
                W = PotentialStep(W,potentialPropagatorFactor,unusedSlot)                

            return W



        #
        ## BEGIN: of creation of Oles KERR Propagators
        #

        def KerrPropagatorOneStep__Max__in_OLE_Formulation(W):
            unusedSlot = 0

            W = PotentialStep(W,potentialPropagatorFactor,unusedSlot)
            W = KineticStep(W,effKineticPropagatorFactor1_2,unusedSlot)

            W = PotentialStep(W,effPotentialPropagatorFactor1_2,unusedSlot)
            W = KineticStep(W,effKineticPropagatorFactor2_2,unusedSlot)

            W = PotentialStep(W,effPotentialPropagatorFactor2_2,unusedSlot)
            W = KineticStep(W,effKineticPropagatorFactor3_2,unusedSlot)

            W = PotentialStep(W,effPotentialPropagatorFactor3_2,unusedSlot)
            W = KineticStep(W,effKineticPropagatorFactor4_2,unusedSlot)

            W = PotentialStep(W,effPotentialPropagatorFactor4_2,unusedSlot)
            
            W = KineticStep(W,effKineticPropagatorFactor5_2 + notAllowed*kineticPropagatorFactor,unusedSlot)  # ? #
            W = KineticStep(W,(1-notAllowed)*kineticPropagatorFactor,unusedSlot)  # ? #
            
            if StrangSplitting:
                W = PotentialStep(W,potentialPropagatorFactor,unusedSlot)

            return W


        if KerrCase and extraMotto == 'OlesPropagator':
            print(" I entered extraMotto = 'OlesPropagator' propagation definitions")
                    
            def Vx4(xx):
                return xx**4/4 - (1-blankOutFactor) * CorrectionPotential(xx)
            
            def Tp4(pp):
                return pp**4/4
            
            def Veff(xx):
                '''Effective Potentials for creation of Propagator factors for non-separable terms X^2P^2'''
                return xx**4/12
            
            def Teff(pp):
                '''Effective Potentials for creation of Propagator factors for non-separable terms X^2P^2'''
                return pp**2/2/np.sqrt(2)

            # Defining Chin's parameters for the non-separable integrator
            t2Magn = np.cbrt(6)

            v1 =  1/t2Magn**2
            v2 = -v1/2 * blankOutFactor
            v0 = -2*(v1+v2)
            t1 =  t2Magn
            t2 = -t2Magn

            
            def factorOLE(func, mult, r, s, t):
                '''Helper function for defining the following propagation factors'''
                eps = np.cbrt(dt)
                factorTwo = 1 ## was  np.cbrt(2) in Maximilian's case, here, NO compensation NEEDED!
                return fftpack.ifftshift( \
                    np.exp(-1j*eps*factorTwo*mult*(func(r - hbar* s/2.) - func(r + hbar* s/2.))), axes=(t,))

            expv0V        = factorOLE(Veff, v0, X, Theta, 1)
            expv1V        = factorOLE(Veff, v1, X, Theta, 1)
            expv2V        = factorOLE(Veff, v2, X, Theta, 1)

            expt1T        = factorOLE(Teff, t1, P, -Lambda, 0)    
            expt2T        = factorOLE(Teff, t2, P, -Lambda, 0)
            
            expdtVx4half = fftpack.ifftshift( \
                np.exp(-halveStep*1j*dt*(Vx4(X - hbar* Theta/2.) - Vx4(X + hbar* Theta/2.))), axes=(1,))

            ### Propagator factor for X^4
            expdtVx4 = fftpack.ifftshift( \
                np.exp(-1j*dt*(Vx4(X - hbar* Theta/2.) - Vx4(X + hbar* Theta/2.))), axes=(1,))

            ### Propagator factor for P^4
            expdtTp4 = fftpack.ifftshift( \
                np.exp(-1j*dt*(  Tp4(P + hbar*Lambda/2.) -   Tp4(P - hbar*Lambda/2.))), axes=(0,))
            

            def evolStepKerr1_Ole_9_Exp(W,empty):
                '''Chin 9-Exponentials method'''
                unused_dummy = 0

                if StrangSplitting == True:
                    W = PotentialStep( W, expdtVx4half,  unused_dummy)
                else:
                    W = PotentialStep( W, expdtVx4,  unused_dummy)

                W = PotentialStep( W, expv2V,  unused_dummy)
                W = KineticStep(   W, expt2T, unused_dummy)
                W = PotentialStep( W, expv1V,  unused_dummy)
                W = KineticStep(   W, expt1T, unused_dummy)
                W = PotentialStep( W, expv0V,  unused_dummy)
                W = KineticStep(   W, expt1T, unused_dummy)
                W = PotentialStep( W, expv1V,  unused_dummy)
                W = KineticStep(   W, expt2T, unused_dummy)
                W = PotentialStep( W, expv2V,  unused_dummy)    

                W = KineticStep(   W, expdtTp4, unused_dummy)

                if StrangSplitting == True:
                    W = PotentialStep( W, expdtVx4half,  unused_dummy)

                W = np.real(W)

                return W      

            def evolStepKerr1_Ole_7_Exp(W,empty):
                '''Chin 7-Exponentials method'''
                unused_dummy = 0

                if StrangSplitting == True:
                    W = PotentialStep( W, expdtVx4half,  unused_dummy)
                else:
                    W = PotentialStep( W, expdtVx4,  unused_dummy)

#                W = PotentialStep( W, expv2V,  unused_dummy)
                W = KineticStep(   W, expt2T, unused_dummy)
                W = PotentialStep( W, expv1V,  unused_dummy)
                W = KineticStep(   W, expt1T, unused_dummy)
                W = PotentialStep( W, expv0V,  unused_dummy)
                W = KineticStep(   W, expt1T, unused_dummy)
                W = PotentialStep( W, expv1V,  unused_dummy)
                W = KineticStep(   W, expt2T, unused_dummy)
#                W = PotentialStep( W, expv2V,  unused_dummy)    

                W = KineticStep(   W, expdtTp4, unused_dummy)

                if StrangSplitting == True:
                    W = PotentialStep( W, expdtVx4half,  unused_dummy)

                W = np.real(W)

                return W      

        #
        ## BEGIN: of creation of Oles KERR Propagators
        #

        #
        ##
        ### END: of creation of Oles Propagators
        ##
        #

        # def KerrPropagatorOneStep_without_Comments(W):

        #     B = X_P__To__X_Theta(W)
        #     B *= potentialPropagatorFactor
        #     A = X_Theta__To__Lambda_P(B)
        #     A *= effKineticPropagatorFactor1_2
        #     B = Lambda_P__To__X_Theta(A)
        #     B *= effPotentialPropagatorFactor1_2
        #     A = X_Theta__To__Lambda_P(B)
        #     A *= effKineticPropagatorFactor2_2
        #     B = Lambda_P__To__X_Theta(A)
        #     B *= effPotentialPropagatorFactor2_2
        #     A = X_Theta__To__Lambda_P(B)
        #     A *= effKineticPropagatorFactor3_2
        #     B = Lambda_P__To__X_Theta(A)
        #     B *= effPotentialPropagatorFactor3_2
        #     A = X_Theta__To__Lambda_P(B)
        #     A *= effKineticPropagatorFactor4_2
        #     B = Lambda_P__To__X_Theta(A)
        #     B *= effPotentialPropagatorFactor4_2
        #     A = X_Theta__To__Lambda_P(B)
        #     A *= effKineticPropagatorFactor5_2
        #     A *= kineticPropagatorFactor
        #     W = Lambda_P__To__X_P(A)

        #     if StrangSplitting:
        #         B = X_P__To__X_Theta(W)
        #         B *= potentialPropagatorFactor
        #         W = Lambda_P__To__X_P(X_Theta__To__Lambda_P(B))

        #     return W


        # def KerrPropagatorOneStep(W):
        #         # ----- Propagate the X^4 term -----
        #         # Transforming to the X-Theta representation
        #         B = X_P__To__X_Theta(W)
        #         # Apply potential factor
        #         B *= potentialPropagatorFactor

        #         # ----- Propagate the first cross term X^2P^2 -----
        #         # Transforming to the Lambda-P representation
        #         A = X_Theta__To__Lambda_P(B)
        #         # Apply effective kinetic 1
        #         A *= effKineticPropagatorFactor1_2
        #         B = Lambda_P__To__X_Theta(A)
        #         # Apply effective potential 1
        #         B *= effPotentialPropagatorFactor1_2
        #         A = X_Theta__To__Lambda_P(B)
        #         # Apply effective kinetic 2
        #         A *= effKineticPropagatorFactor2_2
        #         B = Lambda_P__To__X_Theta(A)
        #         # Apply effective potential 2
        #         B *= effPotentialPropagatorFactor2_2
        #         A = X_Theta__To__Lambda_P(B)
        #         # Apply effective kinetic 3
        #         A *= effKineticPropagatorFactor3_2
        #         B = Lambda_P__To__X_Theta(A)
        #         # Apply effective potential 3
        #         B *= effPotentialPropagatorFactor3_2
        #         A = X_Theta__To__Lambda_P(B)
        #         # Apply effective kinetic 4
        #         A *= effKineticPropagatorFactor4_2
        #         B = Lambda_P__To__X_Theta(A)
        #         # Apply effective potential 4
        #         B *= effPotentialPropagatorFactor4_2
        #         A = X_Theta__To__Lambda_P(B)
        #         # Apply effective kinetic 5
        #         A *= effKineticPropagatorFactor5_2

        #         # Apply kinetic factor
        #         A *= kineticPropagatorFactor
        #         # Returning to the X-P representation
        #         W = Lambda_P__To__X_P(A)
        #         if StrangSplitting:
        #                 # Transforming to the X-Theta representation
        #                 B = X_P__To__X_Theta(W)
        #                 # Apply potential factor
        #                 B *= potentialPropagatorFactor
        #                 W = Lambda_P__To__X_P(X_Theta__To__Lambda_P(B))
        #         return W




        # def KerrPropagatorOneStep__ALT(W):

        #     B = X_P__To__X_Theta(W)
        #     B *= potentialPropagatorFactor
        #     A = X_Theta__To__Lambda_P(B)
        #     A *= effKineticPropagatorFactor1_2
        #     B = Lambda_P__To__X_Theta(A)
        #     B *= effPotentialPropagatorFactor1_2
        #     A = X_Theta__To__Lambda_P(B)
        #     A *= effKineticPropagatorFactor2_2
        #     B = Lambda_P__To__X_Theta(A)
        #     B *= effPotentialPropagatorFactor2_2
        #     A = X_Theta__To__Lambda_P(B)
        #     A *= effKineticPropagatorFactor3_2
        #     B = Lambda_P__To__X_Theta(A)
        #     B *= effPotentialPropagatorFactor3_2
        #     A = X_Theta__To__Lambda_P(B)
        #     A *= effKineticPropagatorFactor4_2
        #     B = Lambda_P__To__X_Theta(A)
        #     B *= effPotentialPropagatorFactor4_2
        #     A = X_Theta__To__Lambda_P(B)
        #     A *= effKineticPropagatorFactor5_2
        #     A *= kineticPropagatorFactor
        #     W = Lambda_P__To__X_P(A)

        #     if StrangSplitting:
        #         B = X_P__To__X_Theta(W)
        #         B *= potentialPropagatorFactor
        #         W = Lambda_P__To__X_P(X_Theta__To__Lambda_P(B))

        #     return W



        # Plot initial state
        Plot_Wigner(0, dt, plots_dir, Winit)
        # Propagation
        W = np.array(Winit)
        Es = [Einit]
        WoW = [WoWinit]

        #
        ##
        ### Time  Loop
        ##
        #
        for t in range(1,timeStepsN):                
            if KerrCase:

               if extraMotto == 'OlesPropagator':
                   if exp7:
                       W = evolStepKerr1_Ole_7_Exp(W,0)
                   else:
                       W = evolStepKerr1_Ole_9_Exp(W,0)
               else:
                   W = KerrPropagatorOneStep__Max__in_OLE_Formulation(W) # KerrPropagatorOneStep(W)

            else: # separable Hamiltonian

                if extraMotto == 'OlesPropagator':
                    W = SeparableHamiltonianPropagatorOneStep_OLE(W)
                else:
                    W = SeparableHamiltonianPropagatorOneStep(W)

            E = IntegrateXP(Energy_Landscape*W)
            Woverlap = 2*np.pi*IntegrateXP(Winit*W)
            Es.append(E)
            WoW.append(Woverlap)
            if (t % int(timeStepsN/29) == 1):
                print(f"dt={dt} T_run:{time.time()-start_time:.1f} sec [{t}/{timeStepsN}:t={dt*t:.4f}] {motto} R={R}:psi_0({x_off:.3f},{p_off:.3f}), Grid[{discretizationX}x{discretizationP}]   E[{t}]={E:.3f}  <Wo|W>[{t}]={Woverlap:.3f}")

            # Intermediate plots
            if (t % int(timeStepsN/3) == 1):
                Plot_Wigner(t, dt, plots_dir, W)


        # Plot final state
        Plot_Wigner(t, dt, plots_dir, W)
        # Store energies for plotting later so as to not conflict with Wigner distribution plotting
        energy_data[i][j].append((dt,timeStepsN,Es.copy()))
        overlap_data[i][j].append((dt,timeStepsN,WoW.copy()))

        #
        ##
        ### END of Time  Loop
        ##
        #
        

# Plot energy scaling
def span(X, X0):
    return max((np.max(X) - X0), (X0 - np.min(X)))

compTime = f"[{time.time()-start_time:.1f} sec]"

suptitleString=f"RunT:{compTime:s} Grid[{discretizationX}x{discretizationP}] {motto} R={R} dt={list_dt_4_Kerr if KerrCase else list_dt_4_separableSystem}  Order[{'2Strang' if StrangSplitting else '1?'}]"

fig,axs = plt.subplots(3,3,figsize=(19.2,14.4))
# fig.suptitle(f"(Energy Scaling: R={R} {'Kerr Oscillator [Max: T~1/24*p**4 +p/2; V~1/2/sqrt(2)*x**2]' if KerrCase else 'x4-Oscillator'}) dt={list_dt_4_Kerr if KerrCase else list_dt_4_separableSystem}  Order[{'2Strang' if StrangSplitting else '1?'}]  {motto}   Grid[{discretizationX}x{discretizationP}] ")

fig.suptitle(f"Energy Scaling: {suptitleString}")

for idx,(x_off,p_off) in enumerate(initial_offsets):
    i,j = subplot_indices[idx]
    # Get data
    logDts = []
    logMaxErrors = []
    for dt,timeStepsN,Es in energy_data[i][j]:
        logDts.append(np.log10(dt))
        logMaxErrors.append(np.log10(np.abs(span(Es, Es[0]))))
    # Plot data points
    axs[i, j].scatter(logDts, logMaxErrors, label="Data Points", color='blue')
#    axs[i, j].plot(logDts, logMaxErrors, linewidth=3.0, label="Data Points")
    # Best fit
    gradient, intercept, r_value, p_value, std_err = stats.linregress(logDts, logMaxErrors)
    axs[i, j].plot(logDts, [gradient*logdt+intercept for logdt in logDts], label="Best-Fit Line", color='red')
    axs[i, j].legend()
    plt.setp(axs[i, j], xlabel=r"$\log_{10} \left (dt \right )$", ylabel=r"$\log_{10} | E_{max} - E_0 |$", \
             title=f"Gradient[psi({x_off:.3f},{p_off:.3f})]={gradient:.3f}")
    # TODO fit this into the plot?
    print(f"Energy-Gradient[psi({x_off:.3f},{p_off:.3f})]={gradient}")
axs[1, 1].axis("off")
plt.savefig(f"{discretizationX}x{discretizationP}_Energy_Scaling_R_{R}_{motto}{'_Kerr' if KerrCase else ''}{'_2nd_Strang' if StrangSplitting else ''}_dt{list_dt_4_Kerr if KerrCase else list_dt_4_separableSystem}.png")


# Plot energies comparison
fig,axs = plt.subplots(3,3,figsize=(19.2,14.4))
fig.suptitle(f"(Energy Fluctuations: R={R} {'Kerr Oscillator [Max: T~1/24*p**4 +p/2; V~1/2/sqrt(2)*x**2]' if KerrCase else 'x4-Oscillator'}) dt={list_dt_4_Kerr if KerrCase else list_dt_4_separableSystem}  Order[{'2Strang' if StrangSplitting else '1?'}][{motto}]   Grid[{discretizationX}x{discretizationP}]")
for idx,(x_off,p_off) in enumerate(initial_offsets):
    i,j = subplot_indices[idx]
    print(f"Energy-Fluctuations[psi({x_off:.3f},{p_off:.3f})]")	
    for dt,timeStepsN,Es in energy_data[i][j]:
        axs[i, j].plot(np.linspace(0, dt*timeStepsN, num=timeStepsN), Es, label=f"dt={dt}")
    axs[i, j].legend()
    plt.setp(axs[i, j], xlabel='t', ylabel='E', title=f'psi({x_off:.3f},{p_off:.3f})')
axs[1, 1].axis('off')
plt.savefig(f"{discretizationX}x{discretizationP}_Energy_Fluctuations_R_{R}_{motto}{'_Kerr' if KerrCase else ''}{'_2ndStrang' if StrangSplitting else ''}_dt{list_dt_4_Kerr if KerrCase else list_dt_4_separableSystem}.png")

# Plot scaling <W(t)|Wo>

fig,axs = plt.subplots(3,3,figsize=(19.2,14.4))
fig.suptitle(f"(|<W(t)|Wo>|-Scaling: R={R} {'Kerr Oscillator [Max: T~1/24*p**4 +p/2; V~1/2/sqrt(2)*x**2]' if KerrCase else 'x4-Oscillator'}) dt={list_dt_4_Kerr if KerrCase else list_dt_4_separableSystem}  Order[{'2Strang' if StrangSplitting else '1?'}][{motto}]    Grid[{discretizationX}x{discretizationP}]")

for idx,(x_off,p_off) in enumerate(initial_offsets):
    i,j = subplot_indices[idx]
    # Get data
    logDts = []
    logMaxErrors = []
    for dt,timeStepsN,WoW in overlap_data[i][j]:
        logDts.append(np.log10(dt))
        logMaxErrors.append(np.log10(np.abs(WoW[timeStepsN-1]-1)))
#        logMaxErrors.append(np.log10(np.abs(span(WoW, WoW[0]))))
    # Plot data points
    axs[i, j].scatter(logDts, logMaxErrors, linewidth=3.0, label="Data Points", color='black')
    # Best fit
    gradient, intercept, r_value, p_value, std_err = stats.linregress(logDts, logMaxErrors)
    axs[i, j].plot(logDts, [gradient*logdt+intercept for logdt in logDts], label="Best-Fit Line", color='green')
    axs[i, j].legend()
    plt.setp(axs[i, j], \
             xlabel=r"$\log_{10} \left (dt \right )$", \
             ylabel=r"$\log_{10} | <Wo|W> - 1 |$", \
             title=f"|<W(t)|Wo>|-Scaling Gradient[psi({x_off:.3f},{p_off:.3f})]={gradient:.3f}"
            )
    # TODO fit this into the plot?
    print(f"|<W(t)|Wo>|-Scaling Gradient[psi({x_off:.3f},{p_off:.3f})]={gradient}")
axs[1, 1].axis("off")
plt.savefig(f"{discretizationX}x{discretizationP}_Overlap_Scaling_R_{R}_{motto}{'_Kerr' if KerrCase else ''}{'_2ndStrang' if StrangSplitting else ''}_dt{list_dt_4_Kerr if KerrCase else list_dt_4_separableSystem}.png")

        
# Plot |<W(t)|Wo>| comparison
fig,axs = plt.subplots(3,3,figsize=(19.2,14.4))
fig.suptitle(f"(|<W(t)|Wo>|-Fluctuations: R={R} {'Kerr Oscillator [Max: T~1/24*p**4 +p/2; V~1/2/sqrt(2)*x**2]' if KerrCase else 'x4-Oscillator'}) dt={list_dt_4_Kerr if KerrCase else list_dt_4_separableSystem}  Order[{'2Strang' if StrangSplitting else '1?'}][{motto}]     Grid[{discretizationX}x{discretizationP}]")
for idx,(x_off,p_off) in enumerate(initial_offsets):
    i,j = subplot_indices[idx]
    for dt,timeStepsN,WoW in overlap_data[i][j]:
        axs[i, j].plot(np.linspace(0, dt*timeStepsN, num=timeStepsN), WoW, label=f"dt={dt}")
    axs[i, j].legend()
    plt.setp(axs[i, j], xlabel='t', ylabel='<Wo|W>', title=f'psi({x_off:.3f},{p_off:.3f})')
axs[1, 1].axis('off')
plt.savefig(f"{discretizationX}x{discretizationP}_OverlapFluctuations_R_{R}_{motto}{'_2ndStrang' if StrangSplitting else ''}_dt{list_dt_4_Kerr if KerrCase else list_dt_4_separableSystem}.png")
