#!/usr/bin/python3

"""
Calculation of Berthelot equation of state, including coexistence regime
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy import integrate
from cmath import sqrt

#%%

'''Functions used in the calculations'''
# Berthelot equation of state as function of Tp = T/Tc and Vp = V/Vc
def pb(Vp,Tp):
    return 8*Tp/(3*Vp-1)-3/(Tp*Vp**2)

# Example plot
Vdl = np.arange(.2,10,.01) # range set such that the three extremal points are
pdl = pb(Vdl,0.9)           # clearly visible
plt.figure()
plt.plot(Vdl,pdl,'k-')
#plt.ylim(-5,10)

#roots of dP/dV we'll need later on
def root1(Tp): #This root we can actually ignore
    complex = 3/(4*Tp**2) + (-2916 + 2592*Tp**2)/(1296*Tp**2 *(-27 + 36*Tp**2 
            - 8*Tp**4 + 8*sqrt(-Tp**6 + Tp**8))**(1/3)) - (-27 + 36*Tp**2 - 
            8*Tp**4 + 8*sqrt(-Tp**6 + Tp**8))**(1/3)/(4*Tp**2)
    return complex.real

def root2(Tp): #Relevant local maximum of pb(V,T)
    complex = 3/(4*Tp**2) - (1 + 1j*sqrt(3))*(-2916 + 2592*Tp**2)/(2592*Tp**2 *(
        -27 + 36*Tp**2 - 8*Tp**4 + 8*sqrt(-Tp**6 + Tp**8))**(1/3)) + (1 - 1j*sqrt(3))*(-27 
        + 36*Tp**2 - 8*Tp**4 + 8*sqrt(-Tp**6 + Tp**8))**(1/3)/(8*Tp**2)
    return complex.real

def root3(Tp): #Relevant local minimum of pb(V,T)
    complex = 3/(4*Tp**2) - (1 - 1j*sqrt(3))*(-2916 + 2592*Tp**2)/(2592*Tp**2 *(
         -27 + 36*Tp**2 - 8*Tp**4 + 8*sqrt(-Tp**6 + Tp**8))**(1/3)) + (1 + 1j*sqrt(3))*(-27 
         + 36*Tp**2 - 8*Tp**4 + 8*sqrt(-Tp**6 + Tp**8))**(1/3)/(8*Tp**2)
    return complex.real

# Test for reduced temperature Tp = 0.9, for which we have made an example plot
print(root1(0.9))
print(root2(0.9))
print(root3(0.9))

#%%

# This function finds Vp value for a given pp value
def Vpf(pp,Tp):
    # Root finding function giving values for Vp at given pp
    def func(Vp,*param):
        pp,Tp = param
        return pp - pb(Vp, Tp)
    pT = (pp,Tp)
    if Tp>=1:
        # In this case, there is only one solution
        Vpe = fsolve(func, 1/pT[0], args=pT)
    else:
        # One can show that dp/dV = 0 has three solutions V(T), these solutions
        # are quite messy and we of course only need two to determine the
        # coexistence region. Further inspection shows that the roots with the
        # highest volume values are the ones we need. The solutions
        # are actually complex but the imaginary part is tiny so we neglect it
        dpdV00 = root3(Tp)
        # print(dpdV00)
        # print(pb(dpdV00,Tp))
        dpdV01 = root2(Tp)
        # print(dpdV01)    
        # print(pb(dpdV01,Tp))
        # If p is outside of the range bounded by the local p minimum and maximum,
        # there is only one solution
        if pp>pb(dpdV01, Tp):
            Vpe = fsolve(func, (1-1e-1)*dpdV00, args=pT,factor=0.1)
            #Vpe = brentq(func, root1(Tp), dpdV00, args=pT)
        elif pp<pb(dpdV00, Tp):
            Vpe = fsolve(func, (1+1e-1)*dpdV01, args=pT)
        else:
            # Here, there are three solutions
            # Establish search range for volumes
            Vmin = fsolve(func, (1-1e-1)*dpdV00, args=(pb(dpdV01, Tp), Tp))
            Vmax = fsolve(func, (1+1e-1)*dpdV01, args=(pb(dpdV00, Tp), Tp))
            Vpe = np.zeros(3)
            #print(Vmin)
            #print(Vmax)
            # Vpe = []
            # for pi in [1/Tp - 1.01*np.sqrt(1-Tp)/Tp, 1/Tp, 1/Tp + 1.01*np.sqrt(1-Tp)/Tp]:
            #     Vpe = np.append(Vpe,fsolve(func,pi,args=pT))
            #Vpe = fsolve(func,[1/Tp - 1.1*np.sqrt(1-Tp)/Tp, 1/Tp, 1/Tp + 1.1*np.sqrt(1-Tp)/Tp],args=pT)
            Vpe[0] = brentq(func, Vmin, root3(Tp), args=pT)
            Vpe[1] = brentq(func, root3(Tp), root2(Tp), args=pT)
            Vpe[2] = brentq(func, root2(Tp), Vmax, args=pT)
    return Vpe 

# # Test calculations
print(Vpf(1.0,1.2))
print(Vpf(0.72,0.9))
print(Vpf(0.25,0.9))

#%%

# This function is used for the Maxwell construction; 
# Calculate the area under the p(V) curve for Vbegin to Vend minus the rectangle p(Vend - Vbegin)
def pdVint(pe0,Tp):
    Vpl = Vpf(pe0,Tp)
    #print(Vpl)
    intval = integrate.quad(pb,Vpl[0], Vpl[2], args=(Tp,)) - pe0*(Vpl[2] - Vpl[0])
    #print(intval)
    return intval[0]

# Root finding function for the function above 
# (giving the pressure at which the Maxwell construction is fulfilled)
def func2(pe,*param):
    [Tp] = param
    val = pdVint(pe,Tp)
    return val

# Tests of functions
Tp = 0.9
# Test integration at pb(1/Tp,Tp)
print(pdVint(pb(1/Tp,Tp), Tp))
# Trial function to find coexistence pressure
pe = brentq(func2,np.max([1e-3,(1+1e-9)*pb(root3(Tp),Tp)]), (1-1e-9)*pb(root2(Tp),Tp),args=Tp)
print(Vpf(pe,Tp))
# Check if Maxwell construction indeed gives zero
print(pdVint(pe, Tp))

#%%

# Calculate pressure incorporating coexistence
# Input are actual volume (scalar or array) and temperature and critical pressure, volume and temperature
# Output is the pressure under these circumstances
def pb_coex(V,T,pc,Vc,Tc):
    Vp = np.array([V]).flatten()/Vc
    Tp = T/Tc
    p = np.zeros(len(Vp))
    for i in np.arange(len(Vp)):
        Vpi = Vp[i]
        if Tp >= 1:
            # No coexistence possible; return Berthelot pressure
            p[i] = pc*pb(Vpi,Tp)
        else:
            # Coexistence possible 
            # Step 1: determine coexistence pressure 
            pe = brentq(func2, np.max([1e-3,(1+1e-9)*pb(root3(Tp),Tp)]), 
                        (1-1e-9)*pb(root2(Tp), Tp), args=Tp)
            # Step 2: determine coexistence range of volumes
            [V1,V2,V3] = Vpf(pe, Tp)
            # Step 3: determine whether Vp is inside coexistence range
            if V1<Vpi and Vpi<V3:
                p[i] = pc*pe
            else:
                p[i] = pc*pb(Vpi, Tp)            
    return p

#%%
'''Trial result for ethane'''

# Critical parameters
Tc = 305.33 # [K], 32.17 degree C
pc = 48.718e5 # [Pa]
Vc = 0.146 # l/mol

T = 290.0
V = np.arange(0.08,1.5,0.01)

# Single value
print(pb_coex(V[0],T,pc,Vc,Tc))

# Array of values
pl = pb_coex(V,T,pc,Vc,Tc)
# Sample plot
plt.figure()
plt.plot(V,pb_coex(V,T,pc,Vc,Tc),'k-')
plt.xlabel('$V$')
plt.ylabel('$p$')
plt.show()


