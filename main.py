#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr

import vdw
import dieterici
import berthelot


Ts = []
ps = []
p_sigs = []
Vs = []

p=[]
T=22.5
V=[4,3.75,3.5,3.25,3,2.75,2.5,2.25,2,1.9,1.85,1.8,1.75,1.7,1.65,1.6,1.5,1.25,1,0.75,0.6,0.55,0.5,0.45,0.4,0.35,0.31]
p0=[11.5,12.2,13,14,14.8,16,17,18.5,20,20.5,21,21.4,21.4,21.5,21.5,21.5,21.5,21.8,21.8,21.8,22,22,22,22.3,23.5,24.5,43.5]
p1=[11.6,12.3,13.1,14,15,16,17.2,18.6,20.1,20.6,21,21.4,21.5,21.6,21.6,21.6,21.7,21.9,22,22.1,22.3,22.3,22.3,22.4,23,24.1,48]

for i in range(len(p0)):
	p.append(np.mean([p0[i],p1[i]]))
ps.append(p)
p_sigs.append(np.transpose(np.ones(len(p0)) * 2.5*10**4 / np.sqrt(2)))
Vs.append(V)
Ts.append(T)

p = []
s = []
T = 25.4
V=[4,3.75,3.5,3.25,3,2.75,2.5,2.25,2,1.75,1.7,1.6,1.5,1.4,1.25,1,0.75,0.7,0.6,0.5,0.45,0.4,0.35]
p0=[11.5,12.4,13.2,14,15,16,16.4,17.9,20.25,22,0,0,23,0,23,23,23.5,0,0,24,0,24.2,0]
p1=[11.8,12.5,13.3,14.2,15.1,16.2,17.4,18.9,20.4,22.1,22.4,22.7,23,22.9,23.1,23.1,23.4,23.2,23.5,23.6,23.6,24.1,26.6]
p2=[11.9,13.3,14.2,14.3,15.2,16.3,17.5,19,20.5,22,22.4,22.8,23,23.1,23.1,23.2,23.4,23.9,23.6,23.5,23.9,24.1,26]
for i in range(len(p0)):
	if p0[i] == 0:
		p.append(np.mean([p1[i], p2[i]]))
		s.append(2.5*10**4/np.sqrt(2))
	else: #TODO: SD PROPER
		p.append(np.mean([p0[i],p1[i], p2[i]]))
		s.append(2.5*10**4/np.sqrt(3))
ps.append(p)
p_sigs.append(np.transpose(np.array(s)))
Vs.append(V)
Ts.append(T)

p = []
T = 30.1
V = [4,3.75,3.5,3.25,3,2.75,2.5,2.25,2,1.9,1.8,1.7,1.6,1.5,1.45,1.4,1.35,1.3,1.25,1,0.75,0.55,0.5,0.45,0.4,0.35]
p0= [12,12.9,13.6,14.6,15.7,16.7,18,19.6,21.1,21.8,22.6,23.4,24.1,24.9,25.4,25.7,25.7,25.7,25.8,26,26.1,26.5,26.5,26.9,26.9,31]       
p1= [12.1,12.9,13.7,14.6,15.6,16.8,18,19.5,21.1,21.9,22.6,23.4,24.2,24.9,25.2,25.6,25.6,25.7,25.7,26.1,26.2,26.2,26.2,26.6,26.9,28.7] 
p2= [12.1,13,13.6,14.6,15.6,16.8,18,19.6,21.1,21.9,22.6,23.4,24.2,25.1,25.2,25.7,25.8,25.8,25.8,26.1,26.2,26.5,26.5,26.5,27.1,30.1]  
for i in range(len(p0)):
	p.append(np.mean([p0[i],p1[i], p2[i]]))
ps.append(p)
p_sigs.append(np.transpose(np.ones(len(p0)) * 2.5*10**4 / np.sqrt(3)))
Vs.append(V)
Ts.append(T)

T = 35.1
V = [4,3.75,3.5,3.25,3,2.75,2.5,2.25,2,1.75,1.5,1.4,1.3,1.25,1.2,1.15,1.1,1,0.9,0.8,0.7,0.6,0.55,0.5,0.45,0.4,0.35]
p = [12.4,13.1,14,15,16,17.1,18.6,20.1,21.9,23.8,26,27,27.8,28.2,28.7,28.7,28.8,28.9,29,29,29.1,29.2,29.1,29.2,29.2,29.6,38.5]
Ts.append(T)
Vs.append(V)
ps.append(p)
p_sigs.append(np.transpose(np.ones(len(p0)) * 2.5*10**4))

p = []
T = 40.1
V = [4,3.75,3.5,3.25,3,2.75,2.5,2.25,2,1.75,1.5,1.4,1.3,1.2,1.1,1,1.05,0.95,0.9,0.85,0.8,0.7,0.6,0.5,0.45,0.4,0.35]
p0= [12.7,13.5,14.4,15.4,16.5,17.7,19.2,20.9,22.6,24.8,27.2,28.1,29.2,30.2,31.1,32,31.6,32.2,32.4,32.4,32.5,32.6,33,33,33.1,33.8,50]     
p1= [12.8,13.6,14.3,15.2,16.4,17.7,19.2,20.9,22.6,24.9,27.1,28.2,29.1,30.2,31.2,32.1,31.4,32.4,32.5,32.6,32.6,32.6,32.7,33.2,33.4,34,50] 

for i in range(len(p0)):
	p.append(np.mean([p0[i],p1[i]]))
ps.append(p)
p_sigs.append(np.transpose(np.ones(len(p0)) * 2.5*10**4 / np.sqrt(2)))
Vs.append(V)
Ts.append(T)

T=45.1
V=[4,3.75,3.5,3.25,3,2.75,2.5,2.25,2,1.75,1.5,1.4,1.3,1.25,1.2,1.15,1.1,1,0.9,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.34]
p=[12.9,13.7,14.6,15.7,16.9,18.2,19.6,21.4,23.2,25.5,28,29.3,30.5,31,31.6,32.3,32.6,34,35,35.7,36.2,36.5,36.5,36.5,36.5,36.8,37,40,50]
Ts.append(T)
Vs.append(V)
ps.append(p)
p_sigs.append(np.transpose(np.ones(len(p0)) * 2.5*10**4))

T=50.1
V=[4,3.75,3.5,3.25,3,2.75,2.5,2.25,2,1.75,1.5,1.4,1.3,1.25,1.2,1.15,1.1,1,0.9,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.39]
p=[13.4,14.2,15,16.1,17.4,18.6,20.2,22,24,26.4,29,30.5,31.8,32.5,33,33.9,34.5,36,37.4,38.5,39,39.5,40,40.2,40.7,41.2,42.3,48,49.3]
Ts.append(T)
Vs.append(V)
ps.append(p)
p_sigs.append(np.transpose(np.ones(len(p0)) * 2.5*10**4))

T = 54
V = [4,3.75,3.5,3.25,3,2.75,2.5,2.25,2,1.75,1.5,1.25,1,0.75,0.7,0.65,0.6,0.55,0.5,0.45]
p = [13.5,14.5,15.2,16.5,17.7,19.1,20.8,22.4,24.6,27.1,30,33.5,37.4,41.5,42.1,42.4,43.2,44.2,45.1,47.5]
Ts.append(T)
Vs.append(V)
ps.append(p)
p_sigs.append(np.transpose(np.ones(len(p0)) * 2.5*10**4))

#0 is vdw, 1 is diet, 2 is bert
Vc = [np.zeros(len(ps)), np.zeros(len(ps)), np.zeros(len(ps))]
Vc_sig = [np.zeros(len(ps)), np.zeros(len(ps)), np.zeros(len(ps))]
pc = [np.zeros(len(ps)), np.zeros(len(ps)), np.zeros(len(ps))]
pc_sig = [np.zeros(len(ps)), np.zeros(len(ps)), np.zeros(len(ps))]
Tc = [np.zeros(len(ps)), np.zeros(len(ps)), np.zeros(len(ps))]
Tc_sig = [np.zeros(len(ps)), np.zeros(len(ps)), np.zeros(len(ps))]

for i in range(len(ps)):
	print("\nT=", Ts[i])	
	T = Ts[i] + 273.15
	V = np.array(Vs[i]) * 435 * 10**-6
	p0 = np.array(ps[i]) * 10**5
	
	p = np.transpose(p0)
	V_sig = np.ones(len(V))*0.025*10**-6
	
	X = np.array([V, np.ones(len(V)) * T])
	X_sig = np.array([V_sig,np.ones(len(V))*0.5])
	
	vdw_res = vdw.run_fit(T, p, p_sigs[i], X, X_sig)
	diet_res = dieterici.run_fit(T, p, p_sigs[i], X, X_sig)
	bert_res = berthelot.run_fit(T, p, p_sigs[i], X, X_sig)
	
	(pc[0][i], Vc[0][i], Tc[0][i]) = vdw_res.beta
	(pc[1][i], Vc[1][i], Tc[1][i]) = diet_res.beta
	(pc[2][i], Vc[2][i], Tc[2][i]) = bert_res.beta
	(pc_sig[0][i], Vc_sig[0][i], Tc_sig[0][i]) = vdw_res.sd_beta
	(pc_sig[1][i], Vc_sig[1][i], Tc_sig[1][i]) = diet_res.sd_beta
	(pc_sig[2][i], Vc_sig[2][i], Tc_sig[2][i]) = bert_res.sd_beta

def lin_fn(B, X):
	return B[0] + B[1] * X

print(Ts)
print(Tc[0])
print(Tc_sig[0])
print(Tc[2])
print(Tc_sig[2])

for i in range(len(Ts)):
	if Tc_sig[0][i] == 0:
		Tc_sig[0][i] = 50
	if Tc_sig[2][i] == 0:
		Tc_sig[2][i] = 50
	

Ts_sig = np.ones(len(Ts)) * 0.5
### ODR routine to fit linear to Tc
odr_model = odr.Model(lin_fn)
odr_data  = odr.RealData(Ts, Tc[2], sx=Ts_sig, sy=Tc_sig[2])
odr_obj   = odr.ODR(odr_data, odr_model, beta0=[45+273.15, 0])
odr_res   = odr_obj.run()
# Quick overview of results - expand on it yourself
print("Berthelot")
odr_res.pprint()
### ODR routine to fit linear to Tc
odr_model = odr.Model(lin_fn)
odr_data  = odr.RealData(Ts, Tc[0], sx=Ts_sig, sy=Tc_sig[0])
odr_obj   = odr.ODR(odr_data, odr_model, beta0=[45+273.15, 0])
odr_res   = odr_obj.run()
# Quick overview of results - expand on it yourself
print("van der Waals")
odr_res.pprint()


plt.figure()
plt.ylabel("Critical volume (ml/mol)")
plt.xlabel("Temperature measured at (C)")
plt.errorbar(Ts, Vc[0]*10**6, yerr=Vc_sig[0]*10**6, fmt='.', label="vdw")
plt.errorbar(Ts, Vc[1]*10**6, yerr=Vc_sig[1]*10**6, fmt='.', label="diet")
plt.errorbar(Ts, Vc[2]*10**6, yerr=Vc_sig[2]*10**6, fmt='.', label="bert")
plt.legend()
plt.show()

plt.figure()
plt.ylabel("Critical pressure (10^5 Pa)")
plt.xlabel("Temperature measured at (C)")
plt.errorbar(Ts, pc[0]*10**-5, yerr=pc_sig[0]*10**-5, fmt='.', label="vdw")
plt.errorbar(Ts, pc[1]*10**-5, yerr=pc_sig[1]*10**-5, fmt='.', label="diet")
plt.errorbar(Ts, pc[2]*10**-5, yerr=pc_sig[2]*10**-5, fmt='.', label="bert")
plt.legend()
plt.show()

plt.figure()
plt.ylabel("Critical temperature (C)")
plt.xlabel("Temperature measured at (C)")
plt.errorbar(Ts, Tc[0]-273.15, yerr=Tc_sig[0], fmt='.', label="vdw")
plt.errorbar(Ts, Tc[1]-273.15, yerr=Tc_sig[1], fmt='.', label="diet")
plt.errorbar(Ts, Tc[2]-273.15, yerr=Tc_sig[2], fmt='.', label="bert")
plt.legend()
plt.show()
