#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:10:01 2018

@author: samla
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:47:24 2018

@author: Moritz
"""

import getpass as gp
name = gp.getuser()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

print("Start:{}".format(time.ctime()))
#time.sleep(7200)

import sys
sys.path.append('/Library/')

from Functions import *

################################ Data Import ##################################
freq = "M"
years = 5

returns, rf_rate, market, estLength, nAssets = get_Data(freq, years)

start = "1970-01-01"
end = "2018-02-01"
datesformat = "%Y-%m-%d"
start_date = datetime.strptime(start, datesformat)
end_date = datetime.strptime(end, datesformat)

returns = returns.loc[start_date:end_date + relativedelta(months=1)]
rf_rate = rf_rate.loc[start_date:end_date + relativedelta(months=1)]
market = market.loc[start_date:end_date + relativedelta(months=1)]

meanRet = np.asmatrix(np.mean(returns, axis=0)).T
varCovar = np.asmatrix(np.cov(returns.T, ddof=1))
'''Define number of draws and Seed in Monte Carlo'''
MCSize = 10000
print("Monte Carlo Size: {}".format(MCSize))

#'''Define Parameters for estimation'''
rf_perc = np.array(rf_rate)
rf_aux = np.asmatrix([(1. + float(x))**(1 / 12.) - 1 for x in rf_perc]).T
rf = np.mean(rf_aux)
rf_annual = (1. + rf)**12. - 1

gamma = 0.5  # risk aversion parameter
print("Gamma: {}".format(gamma))

short_sale_allowed = False
print("Short Sale Allowed: {}".format(short_sale_allowed))

nAssets = 8  # number of assets to be considered

estLength_list = [48, 60, 72, 96, 120]  # length of estimation period

progress = len(estLength_list) * MCSize

epsilon_list = [0.5, 1, 1.5, 2, 3, 9]

if short_sale_allowed:
	pf_utility = ["max_sharpe", "max_sharpe_LEDOIT"]
	pf_no_util = [
	    "min_var", "over_N_pf", "RP_pf", "hierarchical_pf", "meanVar_pf", "LPM"
	]

	pf_TF = ["threeFund_pf"]
	pf_names = pf_utility + pf_no_util + pf_TF
else:
	pf_utility = ["max_sharpe", "max_sharpe_LEDOIT"]
	pf_no_util = [
	    "min_var", "over_N_pf", "RP_pf", "hierarchical_pf", "meanVar_pf", "LPM"
	]
	pf_names = pf_utility + pf_no_util

pf_names_GUW = [
    "utility_GWPF_" + str(int(10 * epsilon)) for epsilon in epsilon_list
]
'''Simple Return Calculations'''

#SIMPLE RETURNS including the market portfolio, useful to create proper correlated returns
ret_assets_incl_market = np.asmatrix(pd.concat([returns, market], axis=1))

#calculates the vector of mean returns and variance covariance matrix. They include the market
meanRet_incl_market = np.mean(ret_assets_incl_market, axis=0).T
varCov_incl_market = np.asmatrix(np.cov(ret_assets_incl_market.T, ddof=1))
'''Excess Return Calculations'''

#Just Assets
meanRet_excess = meanRet - rf
varCovar_excess = varCovar

#EXCESS RETURNS including the market portfolio
meanRet_incl_market_excess = meanRet_incl_market - rf
varCovar_incl_market_excess = varCov_incl_market

#CHOLESKY
choleskyMat = np.linalg.cholesky(varCov_incl_market)
'''Calculate the true weights and true portfolio characteristics'''
#truePi = tangWeights(meanRet_excess, varCovar_excess, gamma)
if short_sale_allowed:
	truePi = maxSRPF1(meanRet, varCovar, rf)
else:
	truePi = maxSRPF_noshort(meanRet, varCovar, rf)

trueMu = float(np.dot(truePi.T, np.array(meanRet)))
trueSigma = float(np.sqrt(np.dot(truePi.T, np.dot(varCovar, truePi))))

w_true = (trueMu - rf) / (gamma * trueSigma**2)
trueMu_total = w_true * trueMu + (1 - w_true) * rf
trueSigma_total = abs(w_true) * trueSigma
trueCEQ = float(utility_MV(trueMu_total, trueSigma_total, gamma))

CEQ_all = np.ones(
    (len(pf_names) + len(epsilon_list)) * len(estLength_list)).reshape(
        len(pf_names) + len(epsilon_list), len(estLength_list))

for counter, estLength in enumerate(estLength_list):
	for pfname in pf_names:
		globals()["utility_{}".format(pfname)] = []
	for l in epsilon_list:
		globals()["utility_GWPF_" + str(int(10 * l))] = []

	np.random.seed(110693)

	varying_eps = []

	for n in range(MCSize):
		'''assign random values to random value matrix'''
		randomMat = np.random.normal(0., 1., (estLength, nAssets + 1))
		'''induce correlation to the random values by multiplying those with the Cholesky Decomposition'''
		corrRandomMat = np.dot(randomMat, choleskyMat.T)
		'''simulate correlated returns over 60 months for nAssets + Market'''
		corrReturns_incl_market = (
		    np.array(meanRet_incl_market.T) + np.array(corrRandomMat))
		corrReturns_incl_market_excess = (
		    np.array(meanRet_incl_market_excess.T) + np.array(corrRandomMat))
		'''market return inside the simulation'''
		market_ret = corrReturns_incl_market[:, -1]
		corrReturns = corrReturns_incl_market[:, :-1]
		corrReturns_excess = corrReturns - rf
		estMu = np.asmatrix(np.mean(corrReturns, axis=0)).T

		estSigma = np.asmatrix(np.cov(corrReturns.T, ddof=1))
		estSigma_LEDOIT = np.asmatrix(LW(corrReturns, assume_centered=True)[0])

		varyingEpsilon = varying_epsilon(corrReturns_excess, market_ret, rf)
		varying_eps.append(varyingEpsilon)

		# Insert all the asset allocation methods here and calculate utility based on the estimated weights and true parameters
		'''Calculate portfolio weights'''
		if short_sale_allowed:
			min_var = minVarPF1(estSigma)
			max_sharpe = maxSRPF1(estMu, estSigma, rf)
			max_sharpe_LEDOIT = maxSRPF1(estMu, estSigma_LEDOIT, rf)
			over_N_pf = np.asmatrix([1. / nAssets for i in range(nAssets)]).T
			threeFund_pf = threeFundSeparation(estMu, estSigma, estLength,
			                                   gamma, rf)
			RP_pf = riskParity(estSigma)
			LPM = lpm_port(estMu, corrReturns, exp_ret_chosen=0.062 / 12)
			hierarchical_pf = getHRP(estSigma)
			meanVar_pf = meanVarPF_one_fund(estMu, estSigma, gamma)
		else:
			min_var = minVarPF_noshort(estSigma)
			max_sharpe = maxSRPF_noshort(estMu, estSigma, rf)
			max_sharpe_LEDOIT = maxSRPF_noshort(estMu, estSigma_LEDOIT, rf)
			over_N_pf = np.asmatrix([1. / nAssets for i in range(nAssets)]).T
			RP_pf = riskParity_noshort(estSigma)
			LPM = lpm_port_noshort(
			    estMu, corrReturns, exp_ret_chosen=0.062 / 12)
			hierarchical_pf = getHRP(estSigma)
			meanVar_pf = meanVarPF_one_fund_noshort(estMu, estSigma, gamma)

		# Calculate CER for portfolios IN!! CARA utility setting
		pf_weights_util = []

		for names in pf_utility:
			pf_weights_util.append(globals()[names])

		for pf, pfname in zip(pf_weights_util, pf_names):
			mu_w = PF_return(pf, estMu)
			sigma_w = np.sqrt(PF_variance(pf, estSigma))
			w = weight_risky_assets(mu_w, rf, sigma_w, gamma)

			mu = PF_return(pf, meanRet)
			sigma = np.sqrt(PF_variance(pf, varCovar))
			if short_sale_allowed == False and w > 0. or short_sale_allowed:
				mu_total = PF_return_risky_rf(w, pf, meanRet, rf)
				sigma_total = PF_sigma_risky_rf(w, pf, varCovar)
			else:
				mu_total = rf
				sigma_total = 0.
			globals()["utility_{}".format(pfname)].append(
			    utility_MV(mu_total, sigma_total, gamma))

		# Calculate CER for portfolios NOT IN!! CARA utility setting
		pf_weights_no_util = []
		for names in pf_no_util:
			pf_weights_no_util.append(globals()[names])

		for pf, pfname in zip(pf_weights_no_util, pf_no_util):
			mu = PF_return(pf, meanRet)
			sigma = np.sqrt(PF_variance(pf, varCovar))
			globals()["utility_{}".format(pfname)].append(
			    utility_MV(mu, sigma, gamma))
		if short_sale_allowed:
			w_TF = np.sum(threeFund_pf)
			mu_TF = PF_return_risky_rf(w_TF, threeFund_pf, meanRet, rf)
			sigma_TF = PF_sigma_risky_rf(w_TF, threeFund_pf, varCovar)
			utility_threeFund_pf.append(utility_MV(mu_TF, sigma_TF, gamma))

		for epsilon in epsilon_list:
			if epsilon != 9:
				if short_sale_allowed:
					GWPF = GWweights1(corrReturns, estMu, estSigma, epsilon,
					                  gamma)
				else:
					GWPF = GWweights_noshort(corrReturns, estMu, estSigma,
					                         epsilon, gamma)
			else:
				if short_sale_allowed:
					GWPF = GWweights1(corrReturns, estMu, estSigma,
					                  varyingEpsilon, gamma)
				else:
					GWPF = GWweights_noshort(corrReturns, estMu, estSigma,
					                         varyingEpsilon, gamma)

			meanRet_GWPF = PF_return(GWPF, meanRet)
			sigma_GWPF = np.sqrt(PF_variance(GWPF, varCovar))
			globals()["utility_GWPF_" + str(int(10 * epsilon))].append(
			    utility_MV(meanRet_GWPF, sigma_GWPF, gamma))

		if (n + 1) % (100) == 0:
			to_print = ((counter) * 100. / len(estLength_list) +
			            (n + 1.) / progress * 100)
			print("{0:.2f}%".format(to_print))

	index = np.arange(1, MCSize + 1, 1)
	for pfname in pf_names:
		#        if short_sale_allowed:
		#            os.chdir('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Monte Carlo Utility Analysis/{}/'.format(name, pfname))
		#        else:
		#            os.chdir('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Monte Carlo Utility Analysis/noshort/{}/'.format(name, pfname))
		globals()["utility_{}".format(pfname)] = pd.DataFrame(
		    globals()["utility_{}".format(pfname)], index)
		df_util = globals()["utility_{}".format(pfname)]
		#        df_util.to_csv("utility_MCSize{}_{}_{}_gamma_{}.csv".format(MCSize,
		#                                                     pfname, estLength, gamma))
		globals()["certainty_eq_{}".format(pfname)] = np.mean(
		    winsorize(np.array(df_util), 0.05))
		del globals()["utility_{}".format(pfname)]

#    if short_sale_allowed:
#        os.chdir('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Monte Carlo Utility Analysis/GWPF/'.format(name))
#    else:
#        os.chdir('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Monte Carlo Utility Analysis/noshort/GWPF/'.format(name))
	for epsilon in epsilon_list:
		globals()["utility_GWPF_{}".format(int(10 * epsilon))] = pd.DataFrame(
		    globals()["utility_GWPF_{}".format(int(10 * epsilon))], index)
		#        globals()["utility_GWPF_"+str(int(10 * epsilon))].to_csv("utility_GWPF_MCSize{}_{}_{}_gamma_{}.csv".format(MCSize, int(epsilon), str(estLength), gamma))
		globals()["certainty_eq_GWPF_{}".format(str(int(
		    10 * epsilon)))] = np.mean(
		        winsorize(
		            np.array(
		                globals()["utility_GWPF_" + str(int(10 * epsilon))]),
		            0.05))
		del globals()["utility_GWPF_" + str(int(10 * epsilon))]

	ceq = {}

	for ind, pfname in enumerate(pf_names):
		ceq["certainty_eq_{}".format(pfname)] = globals()[
		    "certainty_eq_{}".format(pfname)]
		CEQ_all[ind, counter] = globals()["certainty_eq_{}".format(pfname)]
	for ind2, o in enumerate(epsilon_list):
		ceq["certainty_eq_GWPF_{}".format(int(
		    10 * o))] = globals()["certainty_eq_GWPF_" + str(int(10 * o))]
		CEQ_all[ind2 + ind + 1, counter] = globals()["certainty_eq_GWPF_"
		                                             + str(int(10 * o))]

	ceq["varyingEpsilon"] = np.mean(np.array(varying_eps))
	ceq["meanRet"] = meanRet
	ceq["True CEQ"] = trueCEQ
	ceq["varCovar"] = varCovar
	ceq["MCSize"] = MCSize
	ceq["rf_annual"] = rf_annual
	ceq["rf"] = rf
	ceq["gamma"] = gamma
	ceq["epsilon"] = epsilon
	ceq["nAssets"] = nAssets
	ceq["estLength"] = estLength

	if short_sale_allowed:
		filename_bt = "CEQ_Analysis_{}".format(estLength)
	else:
		filename_bt = "CEQ_Analysis_{}".format(estLength) + str("noshort")

#    if not os.path.exists('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Monte Carlo Utility Analysis/{}'.format(name, filename_bt)):
#        os.makedirs('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Monte Carlo Utility Analysis/{}'.format(name, filename_bt))

#    os.chdir('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Monte Carlo Utility Analysis/{}'.format(name, filename_bt))
	utility_output = pd.Series(ceq, name='Utility Loss')
#    utility_output.to_csv('Cert_Equiv_MCSize_{}-rf_annual_{:.3f}-gamma_{:.1f}-epsilon{:.1f}-nAssets{}-estLength{}.csv'.format(MCSize, rf_annual, gamma, epsilon, nAssets, estLength))

#os.chdir('/Users/{}/OneDrive/Master Thesis/Data/Analysis_Skripts/Monte Carlo Utility Analysis/Results'.format(name))
index_CEQ = pf_names + pf_names_GUW

if short_sale_allowed:
	yesorno = 'Yes'
else:
	yesorno = 'No'

CEQ_output = pd.DataFrame(CEQ_all, index=index_CEQ, columns=estLength_list)
#CEQ_output.to_csv('MC_Results_MCSize_{}_gamma_{}_shortsale_{}_CEQ_{:.5f}.csv'.format(MCSize, gamma, yesorno, trueCEQ))

print("End:{}".format(time.ctime()))
