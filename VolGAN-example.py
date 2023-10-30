#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:23:23 2023
Example of the VOlGAN code

@author: vuletic@maths.ox.ac.uk
"""


import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import VolGAN

plt.rcParams['figure.figsize'] = [15.75, 9.385]
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rcParams.update({'font.size': 12})

datapath = "datapath"
### datapath is the location of the implied vol data downloaded from the OptionMetrics Implied Volatility Surface File
surfacepath = "surfacepath"
###surfacepath is the location of the "surfacestransform" file containing daily implied vols on the fixed (m,\tau grid in the vector form)
surfaces_transform, prices, prices_prev, log_rtn, m, tau, ms, taus, dates_dt = VolGAN.SPXData(datapath, surfacepath)

realised_vol_t = np.zeros(len(log_rtn)-21)
for i in range(len(realised_vol_t)):
    realised_vol_t[i] = np.sqrt(252 / 21) * np.sqrt(np.sum(log_rtn[i:(i+21)]**2))

plt.plot(dates_dt[21:],realised_vol_t)
#shift the time
dates_t = dates_dt[22:]
#log-return at t, t-1, t-2
log_rtn_t = log_rtn[22:]
log_rtn_tm1 = log_rtn[21:-1]
log_rtn_tm2 = log_rtn[20:-2]
#log implied vol at t and t-1
log_iv_t = np.log(surfaces_transform[22:])
log_iv_tm1 = np.log(surfaces_transform[21:-1])
#we want to simulate the increment at time t (t - t-1)
log_iv_inc_t = log_iv_t - log_iv_tm1

tr = 0.85
noise_dim = 32
hidden_dim = 16
device = 'cpu'
n_epochs = 10000
n_grad = 25
val = True

gen, gen_opt, disc, disc_opt, true_train,true_test, condition_train, condition_test, dates_t,  m, tau, ms, taus  = VolGAN.VolGAN(datapath,surfacepath, tr, noise_dim = noise_dim, hidden_dim = hidden_dim, n_epochs = n_epochs,n_grad = n_grad, lrg = 0.0001, lrd = 0.0001, batch_size = 100, device = 'cpu')


ntr = true_train.shape[0]
B = 10000
dtm = tau * 365
#calculate the panlaty matrices
mP_t,mP_k,mPb_K = VolGAN.penalty_mutau_tensor(m,dtm,device)
m_t = torch.tensor(ms,dtype=torch.float,device=device)
t_t = torch.tensor(taus, dtype= torch.float, device=device)
n_test = true_test.shape[0]
tots_test = np.zeros((n_test,B))
Pks_t_test = mP_k.unsqueeze(0).repeat((n_test,1,1))
Pkbs_t_test = mPb_K.unsqueeze(0).repeat((n_test,1,1))
Ks_t_test = m_t.unsqueeze(0).repeat((n_test,1,1))
ts_t_test = t_t.unsqueeze(0).repeat((n_test,1,1))
ms_t_test = m_t.unsqueeze(0).repeat((n_test,1,1))

gen.eval()
fk = torch.empty(size = (B,n_test,10,8))
fk_ent = np.zeros((B, n_test, 80))
fk_inc = np.zeros((B, n_test, 80))
ret_u = np.zeros((B, n_test))
#tots 
tots_test = np.zeros((n_test,B))
with torch.no_grad():
    for l in tqdm(range(B)):
        #sample noise
        noise = torch.randn((n_test,noise_dim), device=device,dtype=torch.float)
        #sample from the generator
        fake = gen(noise,condition_test[:,:])
        surface_past_test = condition_test[:,3:]
        #simulated implied vol surfaces as vectors
        fake_surface = torch.exp(fake[:,1:] +  surface_past_test)
        #simulated surfaces as a matrices
        fk_ent[l, :, :] = fake_surface.cpu().detach().numpy()
        #simulated increments
        fk_inc[l, :, :] = fake[:,1:].cpu().detach().numpy()
        ret_u[l, :] = fake[:,0].cpu().detach().numpy()
        for i in range(8):
            fk[l,:,:,i] = fake_surface[:,(i*10):((i+1)*10)]
        #calculating arbitrage penalties
        Pks_t_test = mP_k.unsqueeze(0).repeat((n_test,1,1))
        Pkbs_t_test = mPb_K.unsqueeze(0).repeat((n_test,1,1))
        BS = VolGAN.smallBS_tensor(ms_t_test[:,:,:],ts_t_test[:,:,:],fk[l,:,:,:],0)

        _,_,_,tot = VolGAN.arbitrage_penalty_tensor(BS,mP_t.unsqueeze(0).repeat((n_test,1,1)),Pks_t_test[:,:,:],Pkbs_t_test[:,:,:])
        tots_test[:,l] = tot.detach().numpy()
#Check arbitrage penalties
print("Sim ", np.mean(tots_test),np.std(np.mean(tots_test,axis=1)), np.median(np.mean(tots_test,axis=1)))

