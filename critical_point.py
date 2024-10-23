#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr



volumes = [1,	0.95,	0.9,	0.85,	0.8,	0.75,	0.7,	0.65,	0.6,	0.55,	0.5,	0.45,	0.4]
pressures = [35.1,	36,	36.5,	37,	37.4,	37.8,	38.1,	38.3,	38.6,	39,	39.5,	40.3,	45]

volumes = [1.3,	1.25,	1.2,	1.15,	1.1,	1.05,	1,	0.95,	0.9,	0.85,	0.8,	0.75,	0.7,	0.65,	0.6,	0.55,	0.5,	0.45,	0.4]
pressures = [30.3,	31,	31.6,	32.2,	32.8,	33.3,	34.1,	34.6,	35.2,	35.6,	36,	36.2,	36.5,	36.7,	36.7,	36.9,	37.1,	37.2,	39.2]

def deriv(X, Y):
	deriv = []
	ynew = []
	
	for i in range(len(X)-1):
		ynew.append(np.mean([X[i], X[i+1]]))
		deriv.append((Y[i] - Y[i+1]) / (X[i] - X[i+1]))
	return (ynew, deriv)

(vol_new, deriv1) = deriv(volumes, pressures)
(vol_new2, deriv2) = deriv(vol_new, deriv1)

plt.plot(volumes, pressures)
plt.plot(vol_new, deriv1)
plt.plot(vol_new2, deriv2)
plt.show()
