#!/usr/bin/python3

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:48:59 2021
@author: capel102

Calculation of Dieterici equation of state, including coexistence regime
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy import integrate
import scipy.odr as odr

#%%
'''Functions used in the calculations'''

# Dieterici equation of state as function of Tp = T/Tc and Vp = V/Vc
def pd(Vp,Tp):
    return(Tp/(2*Vp - 1))*np.exp(2*(1-1/(Vp*Tp)))

# # Example plot
# Vdl = np.arange(.501,10,.01)
# pdl = pd(Vdl,0.9)
# plt.figure()
# plt.plot(Vdl,pdl,'k-')

# This function finds Vp value for a given pp value
# At coexistence, there are three solutions
def Vpf(pp,Tp):
    # Root finding function giving values for Vp at given pp
    def func(Vp,*param):
        pp,Tp = param
        return pp - pd(Vp,Tp)
    pT = (pp,Tp)
    if Tp>=1:
        # In this case, there is only one solution
        Vpe = fsolve(func,1/pT[0],args=pT)
    else:
        # One can show that points where dp/dV = 0 are at Vp = 1/Tp +/- sqrt(1-Tp)/Tp
        # If pressure between the two corresponding pressures, there are three solutions 
        dpdV00 = 1/Tp - np.sqrt(1-Tp)/Tp
        dpdV01 = 1/Tp + np.sqrt(1-Tp)/Tp
        # If p is outside of the range bounded by the local p minimum and maximum,
        # there is only one solution
        if pp>pd(dpdV01,Tp):
            Vpe = fsolve(func, (1-1e-1)*dpdV00, args=pT)
        elif pp<pd(dpdV00, Tp):
            Vpe = fsolve(func, (1+1e-1)*dpdV01, args=pT)
        else:
            # Here, there are three solutions
            # Establish search range for volumes
            Vmin = fsolve(func,(1-1e-1)*dpdV00,args=(pd(dpdV01,Tp),Tp))
            Vmax = fsolve(func,(1+1e-1)*dpdV01,args=(pd(dpdV00,Tp),Tp))
            Vpe = np.zeros(3)
            # print(Vmin)
            # print(Vmax)
            # Vpe = []
            # for pi in [1/Tp - 1.01*np.sqrt(1-Tp)/Tp, 1/Tp, 1/Tp + 1.01*np.sqrt(1-Tp)/Tp]:
            #     Vpe = np.append(Vpe,fsolve(func,pi,args=pT))
            #Vpe = fsolve(func,[1/Tp - 1.1*np.sqrt(1-Tp)/Tp, 1/Tp, 1/Tp + 1.1*np.sqrt(1-Tp)/Tp],args=pT)
            Vpe[0] = brentq(func,Vmin, 1/Tp - np.sqrt(1-Tp)/Tp,args=pT)
            Vpe[1] = brentq(func,1/Tp - np.sqrt(1-Tp)/Tp,1/Tp + np.sqrt(1-Tp)/Tp,args=pT)
            Vpe[2] = brentq(func,1/Tp + np.sqrt(1-Tp)/Tp,Vmax,args=pT)
    return Vpe

# # Test calculations
#print(Vpf(1.0,1.2))
#print(Vpf(0.72,0.9))

# This function is used for the Maxwell construction; 
# Calculate the area under the p(V) curve for Vbegin to Vend minus the rectangle p(Vend - Vbegin)
def pdVint(pe0,Tp):
    Vpl = Vpf(pe0,Tp)
    intval = integrate.quad(pd,Vpl[0],Vpl[2],args=(Tp,)) - pe0*(Vpl[2] - Vpl[0])
    return intval[0]

# Root finding function for the function above 
# (giving the pressure at which the Maxwell construction is fulfilled)
def func2(pe,*param):
    [Tp] = param
    val = pdVint(pe,Tp)
    return val

# # Tests of functions
# Tp = 0.9
# # Test integration at pd(1/Tp,Tp)
# print(pdVint(pd(1/Tp,Tp), Tp))
# # Trial function to find coexistence pressure
# pe = brentq(func2,(1+1e-9)*pd(1/Tp - np.sqrt(1-Tp)/Tp,Tp), (1-1e-9)*pd(1/Tp + np.sqrt(1-Tp)/Tp,Tp),args=Tp)
# print(Vpf(pe,Tp))
# # Check if Maxwell construction indeed gives zero
# print(pdVint(pe, Tp))

# Calculate pressure incorporating coexistence
# Input are actual volume (scalar or array) and temperature and critical pressure, volume and temperature
# Output is the pressure under these circumstances
def pd_coex(V,T,pc,Vc,Tc):
    Vp = np.array([V]).flatten()/Vc
    Tp = T/Tc
    p = np.zeros(len(Vp))
    for i in np.arange(len(Vp)):
        Vpi = Vp[i]
        if Tp >= 1:
            # No coexistence possible; return Dieterici pressure
            p[i] = pc*pd(Vpi,Tp)
        else:
            # Coexistence possible 
            # Step 1: determine coexistence pressure 
            pe = brentq(func2,(1+1e-9)*pd(1/Tp - np.sqrt(1-Tp)/Tp,Tp), (1-1e-9)*pd(1/Tp + np.sqrt(1-Tp)/Tp,Tp),args=Tp)
            # Step 2: determine coexistence range of volumes
            [V1,V2,V3] = Vpf(pe,Tp)
            # Step 3: determine whether Vp is inside coexistence range
            if V1<Vpi and Vpi<V3:
                p[i] = pc*pe
            else:
                p[i] = pc*pd(Vpi,Tp)            
    return p

def fit_Diet(B, X):
	p = []
	for i in range(len(X[0])):
		p.append(pd_coex(X[0][i], X[1][i], B[0], B[1], B[2])[0])
	return np.array(p)


##%%
#'''Trial result for ethane'''
#
## Critical parameters
#Tc = 305.33 # [K], 32.17 degree C
#pc = 48.718e5 # [Pa]
#Vc = 0.146 # l/mol
#
#T = 295
#V = np.arange(0.08,1.5,0.005)
#
## Single value
#print(pd_coex(V[0],T,pc,Vc,Tc))
#
## Array of values
#pl = pd_coex(V,T,pc,Vc,Tc)
## Sample plot
#plt.figure()
#plt.plot(V,pl,'k.-')
#plt.xlabel('$V$')
#plt.ylabel('$p$')
#plt.show()


def run_fit(T, p, p_sig, X, X_sig): 
	print("Dieterici model")
	
	B_start = [38e5, 263e-6, 330] # pc, Vc, Tc
	
	### ODR routine
	odr_model = odr.Model(fit_Diet)
	odr_data  = odr.RealData(X, p, sx=X_sig, sy=p_sig)
	odr_obj   = odr.ODR(odr_data, odr_model, beta0=B_start)
	#odr_obj.set_job(fit_type=2)
	odr_res   = odr_obj.run()
	# Quick overview of results - expand on it yourself
	#odr_res.pprint()
	
	print("Pc:", round(odr_res.beta[0]/(10**5), 2), "+/-", round(odr_res.sd_beta[0]/(10**5), 2), "10^5 Pa, Vc:", round(odr_res.beta[1]*10**6, 2), "+/-", round(odr_res.sd_beta[1]*10**6, 2), "ml, Tc:", round(odr_res.beta[2], 2), "+/-", round(odr_res.sd_beta[2], 2), "°K =", round(odr_res.beta[2] - 273.15, 2), "°C")
	print("Vc for 2.3 mmol:", round(odr_res.beta[1] * 10**6 / 435, 2), "+/-", round(odr_res.sd_beta[1] * 10**6 / 435, 3), "ml")
	print("Chi-squared:", round(odr_res.sum_square, 4))
	return odr_res

