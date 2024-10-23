#!/usr/bin/python3

"""
Created on Wed Sep  8 14:48:59 2021
@author: hees0101
Modified 14-12-2021
@author: capel102
Modified 29-08-2022
@author: capel102
Added example fit for a single isotherm

Calculation of Van der Waals equation of state, including coexistence regime
"""

from numpy import sinh, cosh, exp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import numpy as np
import scipy.odr as odr

# Inputs: volume (single value or array), temperature and values for critical parameters
# Output: pressure (1-element or multi-element array) according to VdW
def p_vdW(V, T, pc, Vc, Tc):
	"""
	Compute the pressure in a gas ---
	with critical values for pressure, volume and temperature
	given by  pc, Vc and Tc ---
	for given volume V and temperature T.

	Start with the van der Waals equation:
	p = R T / (V-b) - a / V**2   Blundell (26.12) with

	R = 8/3 * pc * Vc / Tc	   Blundell (26.21)
	b = Vc /3					Blundell (26.18)
	a = 27 * pc / b**2		   Blundell (26.20)

	This gives the following relation for p in terms of V, T, pc, Vc and Tc
	"""
	Vp = np.array([V]).flatten()/Vc
	Tp = T/Tc
	p = np.zeros(len(Vp))

	for i in np.arange(len(Vp)):
		p[i] = pc * ((8*Tp)/ (3*Vp[i]-1) - 3 / (Vp[i])**2 )
		if (T<Tc):
			"""
			Below the critical temperature compute vmn and vmx and
			adjust p according to Maxwell's construction. If vmn < V /Vc < vmx,
			see Fig 26.5 in Blundell
			"""
			def function(y):
				" a local-function that drops from 1-T/Tc to -T/Tc "
				sy = sinh(y)
				cy = cosh(y)
				fy = (y * cy - sy) / (sy * cy - y)
				gy = 1 +  2 * cy * fy + fy**2
				f  = 27/4 * fy * (cy+fy) / gy**2
				return f - Tp

	# make sure function(ymin) has different sign than function y(max) otherwise brentq does not work.
			ymin = 2 * np.sqrt(1-Tp) # if T/Tc -> 1, y -> 3*np.sqrt(1-T/Tc)
			ymax = 4 * np.sqrt(1-Tp)
			y = brentq(function, ymin, ymax)
			sy = sinh(y)
			cy = cosh(y)
			fy = (y * cy - sy) / (sy * cy - y)
			vmn = (1+exp(-y)/fy)/3
			vmx = (1+exp(+y)/fy)/3
			if (vmn < Vp[i] < vmx):
				p[i] = pc * ((8*Tp)/ (3*vmn-1) - 3 / (vmn)**2)
			else:
				p[i] = pc * ((8*Tp)/ (3*Vp[i]-1) - 3 / (Vp[i])**2)
	return p


# Inputs: temperature (single value or array) and values for critical parameters
# Output: list of volumes and corresponding pressures at edge of coexistence range
#   If T >= Tc, then the output is NaN
def Vminmax(T, pc, Vc, Tc):
	"""
	This function returns the edges of the coexistence region for the vdW model with Maxwell
	construction for certain critical point values.
	"""
	Tp = np.array([T]).flatten()/Tc
	# output arrays
	Vp = np.zeros((2*len(Tp),2))

	def function(y,Tp):
		" a local-function that drops from 1-T/Tc to -T/Tc "
		sy = sinh(y)
		cy = cosh(y)
		fy = (y * cy - sy) / (sy * cy - y)
		gy = 1 +  2 * cy * fy + fy**2
		f  = 27/4 * fy * (cy+fy) / gy**2
		return f - Tp

	for i in np.arange(len(Tp)):
		if Tp[i] >= 1:
			# Above critical temperature; no coexistence
			Vp[2*i:2*(i+1),:] = np.nan
		else:
			# make sure function(ymin) has different sign than function y(max) otherwise brentq does not work.
			ymin = 2 * np.sqrt(1-Tp[i]) # if T/Tc -> 1, y -> 3*np.sqrt(1-T/Tc)
			ymax = 4 * np.sqrt(1-Tp[i])
			y = brentq(function, ymin, ymax, args=(Tp[i],))
			sy = sinh(y)
			cy = cosh(y)
			fy = (y * cy - sy) / (sy * cy - y)
			vmn = (1+exp(-y)/fy)/3
			vmx = (1+exp(+y)/fy)/3
			Vp[2*i] = [Vc*vmn, pc * ((8*Tp[i])/ (3*vmn-1) - 3 / (vmn)**2)]
			Vp[2*i+1] = [Vc*vmx, pc * ((8*Tp[i])/ (3*vmn-1) - 3 / (vmn)**2)]
	return Vp[Vp[:, 0].argsort()]


##%% Trial results for ethane
#
## Critical parameters
#Tc = 338.03 # [K], 32.17 degree C
#pc = 38.84e5 # [Pa]
#Vc = 0.26349 # l/mol
#
## Range of volumes
#V = np.arange(0.08,1.5,0.005)
## Single temperatures above and below Tc
#Tabove = 345
#Tbelow = 295
#
## Calculate pressures
#print(p_vdW(V[0],Tabove,pc,Vc,Tc))
#print(p_vdW(V[0],Tbelow,pc,Vc,Tc))
## Calculate (V,p) at edges of coexistence region
#print(Vminmax(Tabove, pc, Vc, Tc))
#print(Vminmax(Tbelow, pc, Vc, Tc))
#
## Array of pressures
#pl = p_vdW(V,Tbelow,pc,Vc,Tc)
#
## Coexistence region edges for a range of temperatures
#Trange = np.arange(250,345,0.1)
#Vpcoex = Vminmax(Trange, pc, Vc, Tc)
#
## Sample plot (unstyled!)
#plt.figure()
## isotherm
#plt.plot(V,pl,'r.-')
## Edges of coexistence region
#plt.plot(Vpcoex[:,0],Vpcoex[:,1],'k-')
#plt.xlabel('$V$')
#plt.ylabel('$p$')
#plt.show()


#%% Fit routine

'''
This is the ordinary odr fit function form, with B the parameter array and X the
independent variable array. There are two differences from the
conventional fit function that you know:
	1. the function p required to find the model values is the van-der-Waals model
	with Maxwell construction, and it is therefore (partially) a numerical rather
	than analytical function.
	2. There are two independent variables in X, namely V and T. When calling odr,
	both can be given an uncertainty
When using this fit function yourself, you can choose to fit each isotherm separately
or fit data for all isotherms simultaneously
Note that the input of odr must be organized into rows, so p (and uncertainty) is a n x 1 array,
and X is a n x 2 array with Vi and Ti (i = 1, ..., n) in each column
'''
def fit_vdW(B, X):
	p_maxwell = []
	for i in range(len(X[0])):
		p_maxwell.append(p_vdW(X[0][i], X[1][i], B[0], B[1], B[2])[0])
	return np.array(p_maxwell)


def run_fit(T, p, p_sig, X, X_sig): 
	print("van der Waals model")
	
	B_start = [38e5, 263e-6, 330] # pc, Vc, Tc
	
	### ODR routine
	odr_model = odr.Model(fit_vdW)
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

