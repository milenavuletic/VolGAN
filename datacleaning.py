#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:06:04 2024

@author: milenavuletic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import scipy.interpolate as interpolate
from scipy.interpolate import BSpline
from scipy.interpolate import interp1d

def optionemtricsdata_transform(spxoptions_df, startdate, enddate):
    """
    Function taking as input the spxoptions_df data frame of option prices
    from OptionMetrics
    startdate, enddate: for getting the SPX prices from yahoo fiannce
    returns
    spxoptions_df: included times to maturity in fractions of the year, and date
    strike: converted to align the scale of the spot (the spot data)
    spot: time series of spot prices
    observation_dates: observation dates in the implied vol data
    """
    spxoptions_df['date_dt'] = pd.to_datetime(spxoptions_df['date'], format = '%Y-%m-%d')
    spxoptions_df['exdate_dt'] = pd.to_datetime(spxoptions_df['exdate'], format = '%Y-%m-%d')
    spxoptions_df['time_to_expiry'] = spxoptions_df['exdate_dt'] - spxoptions_df['date_dt']
    spxoptions_df['days_to_exp'] = spxoptions_df['time_to_expiry'].dt.days
    #calendar dates difference, use 356 days for time to maturity
    spxoptions_df['ttm'] = spxoptions_df['days_to_exp'] / 365
    spxoptions_df['strike'] = spxoptions_df['strike_price'] / 1000
    # Define the ticker symbol for S&P 500
    ticker_symbol = "^GSPC"  # This is the ticker symbol for S&P 500 index
    # Fetch historical data from Yahoo Finance
    spxspot_df = yf.download(ticker_symbol, start=startdate, end=enddate)
    spxspot_df['date_dt'] = pd.to_datetime(spxspot_df.index, format = "%Y-%m-%d")
    observation_dates = spxoptions_df['date_dt'].unique()
    spot = np.zeros(len(observation_dates))
    for i in range(len(observation_dates)):
        spot[i] = spxspot_df[spxspot_df['date_dt'] == observation_dates[i]]['Adj Close'].item()
    return spxoptions_df, spxspot_df, spot, observation_dates
    

def data_day(date, spxoptions_df, spxspot_df):
    """
    take the pre-processed spxoptions_df (previous function) and spxspot_df
    return the dataframe with OTM calls and puts and estimated risk-free
    interest rate r
    """
    df_temp = spxoptions_df[spxoptions_df['date_dt'] == date].reset_index()
    df_temp['spot'] = spxspot_df[spxspot_df['date_dt'] == date]['Adj Close'].item()
    df_temp['moneyness'] = df_temp['strike_price'] / (1000 * df_temp['spot'])
    df_temp['mid_price'] = 0.5 * (df_temp['best_offer'] + df_temp['best_bid'])
    #pair up calls and puts to calculate the implied risk-free interest rate
    call_df = df_temp[df_temp['cp_flag'] == 'C']
    put_df = df_temp[df_temp['cp_flag'] == 'P']
    merged_df = pd.merge(call_df, put_df, on=['strike_price', 'exdate', 'spot', 'ttm'])
    #put-call parity
    r_temp = np.log((merged_df['strike_price'] / 1000)/ (merged_df['spot'] - merged_df['mid_price_x'] + merged_df['mid_price_y'])) / merged_df['ttm']
    df_temp['r'] = np.median(r_temp)
    r = np.median(r_temp)
    #out-of-the-money calls and puts
    subset_df = df_temp[((df_temp['cp_flag'] == 'C') & (df_temp['moneyness'] >= 1)) | ((df_temp['cp_flag'] == 'P') & (df_temp['moneyness'] < 1))]
    return subset_df, r


def kernel(x, y, h1, h2):
    """
    2D Gaussiana kernel
    x, y: moneyness and time to maturity (usually)
    h1, h2: bandwidth parameters
    """
    return (np.exp(- x * x / (2 * h1)) * np.exp(- y * y / (2 * h2))) / (2 * np.pi)

def Smooth(I_in,m_in,tau_in,m_want,tau_want,h1,h2):
    """
    Nadaraya-Watson kernel smoothing function using 2-D Gaussian kernel
    I_in: array of implied volatilities in the data
    m_in: moneyness array in the data
    tau_in: time to maturity array in the data
    I_in[i] corresponds to m_in[i] and tau_in[i]
    *doesn't have to be sorted
    m_want, tau_want: the wanted grid, both given as 1d arrays
    """
    I_out = np.zeros((len(m_want),len(tau_want)))
    for i in range(len(m_want)):
        m = m_want[i]
        for j in range(len(tau_want)):
            tau = tau_want[j]
            weights = kernel(m - m_in, tau - tau_in, h1, h2)
            #for each pair (m, tau) on the wanted grid return the
            #Nadaraya-Watson estimator
            I_out[i,j] = np.sum(I_in * weights) / np.sum(weights)
    return I_out

def interpolate_m(m_known,iv_known, m_want, k = 1):
    """
    for FIXED time to maturity tau and implied vol IV(m_known, tau)
    return IV(m_want, tau)
    """
    t, c, k = interpolate.splrep(m_known, iv_known, k=k)
    spl = BSpline(t, c, k)
    return spl(m_want)

def interpolate_tau(tau_known, iv_known, tau_want, k = 1):
    """
    for FIXED moneyness m and implied vol IV(m, tau_known)
    return IV(m, tau_known)
    """
    t, c, k = interpolate.splrep(tau_known, iv_known, k=k)
    spl = BSpline(t, c, k)
    return spl(tau_want)


def interpolate_and_extrapolate(x_data, y_data, x):
    """
    Perform linear interpolation and extrapolation for given x_data and y_data.
    
    Parameters:
        x_data: numpy array
            The observed x values.
        y_data: numpy array
            The observed y values corresponding to x_data.
        x: numpy array
            The x values for which interpolation or extrapolation is desired.
    
    Returns:
        y_interp_extrap: numpy array
            The interpolated and extrapolated y values corresponding to input x.
    """
    # Ensure x_data and y_data have the same length
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length.")

    # Sort the data points by 'x' values
    sorted_indices = np.argsort(x_data)
    x_sorted = x_data[sorted_indices]
    y_sorted = y_data[sorted_indices]

    # Perform linear interpolation
    linear_interp = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')

    # Interpolate the new values
    y_interp = linear_interp(x)

    return y_interp

def interpolate_data(otm_df,m_want,tau_want,ttm_available):
    iv_interpol1 = np.zeros((len(m_want), len(ttm_available)))
    # m1 = np.zeros((len(m_want), len(ttm_available)))
    # t1 = np.zeros((len(m_want), len(ttm_available)))
    for i in range(len(ttm_available)):
        ss_df = otm_df[otm_df['ttm'] == otm_df['ttm'].unique()[i]].reset_index()
        indx = np.argsort(ss_df['moneyness'])
        mavail = ss_df['moneyness'][indx]
        ivavail = ss_df['impl_volatility'][indx]
        iv_interpol1[:, i] = interpolate_and_extrapolate(mavail,ivavail, m_want)
    #     m1[:, i] = m_want
    #     t1[:, i] = ttm_available[i] * np.ones(len(m_want))
    iv_interpol2 = np.zeros((len(m_want), len(tau_want)))
    indx = np.argsort(ttm_available)
    for i in range(len(m_want)):
        iv_interpol2[i, :] = interpolate_and_extrapolate(ttm_available[indx],iv_interpol1[i,indx], tau_want)
    return iv_interpol2

def interpolate_surface(iv_in,m_in,tau_in,m_out,tau_out):
    """
    
    Parameters
    ----------
    iv_in :implied volatility surface len(m_in) x len(m_tau)
    m_in : moneyness grid (1d array)
    tau_in : time to maturity grid (1d array)
    m_out : moneyness value wanted
    tau_out : time to maturity wanted
    Returns
    -------
    iv_interpol2 : interpolated implied vol surface len(m_out) x len(tau_out)
    
    m_in, tau_in, m_out, tau_out should be sorted

    """
    iv_interpol1 = np.zeros((len(m_out), len(tau_in)))
    # first fix time to maturity
    for i in range(len(tau_in)):
        iv_interpol1[:, i] = interpolate_and_extrapolate(m_in,iv_in[:,i], m_out)
    #now fix moneyness
    iv_interpol2 = np.zeros((len(m_out), len(tau_out)))
    for i in range(len(m_out)):
        iv_interpol2[i, :] = interpolate_and_extrapolate(tau_in,iv_interpol1[i,:], tau_out)
    return iv_interpol2

def interpolate_dataBspline(otm_df,m_want,tau_want,ttm_available):
    iv_interpol1 = np.zeros((len(m_want), len(ttm_available)))
    # m1 = np.zeros((len(m_want), len(ttm_available)))
    # t1 = np.zeros((len(m_want), len(ttm_available)))
    for i in range(len(ttm_available)):
        ss_df = otm_df[otm_df['ttm'] == otm_df['ttm'].unique()[i]].reset_index()
        indx = np.argsort(ss_df['moneyness'])
        mavail = ss_df['moneyness'][indx]
        ivavail = ss_df['impl_volatility'][indx]
        iv_interpol1[:, i] = interpolate_m(mavail,ivavail, m_want, k = 1)
    #     m1[:, i] = m_want
    #     t1[:, i] = ttm_available[i] * np.ones(len(m_want))
    iv_interpol2 = np.zeros((len(m_want), len(tau_want)))
    indx = np.argsort(ttm_available)
    for i in range(len(m_want)):
        iv_interpol2[i, :] = interpolate_tau(ttm_available[indx],iv_interpol1[i,indx], tau_want, k = 1)
    return iv_interpol2

def kernelVega(x, y, h1, h2, vega):
    """
    2D Gaussiana kernel
    x, y: moneyness and time to maturity (usually)
    h1, h2: bandwidth parameters
    """
    return vega * (np.exp(- x * x / (2 * h1)) * np.exp(- y * y / (2 * h2))) / (2 * np.pi)

def SmoothVega(I_in,m_in,tau_in,m_want,tau_want,h1,h2, vega):
    """
    Nadaraya-Watson kernel smoothing function using 2-D Gaussian kernel
    I_in: array of implied volatilities in the data
    m_in: moneyness array in the data
    tau_in: time to maturity array in the data
    I_in[i] corresponds to m_in[i] and tau_in[i]
    *doesn't have to be sorted
    m_want, tau_want: the wanted grid, both given as 1d arrays
    """
    I_out = np.zeros((len(m_want),len(tau_want)))
    for i in range(len(m_want)):
        m = m_want[i]
        for j in range(len(tau_want)):
            tau = tau_want[j]
            weights = kernelVega(m - m_in, tau - tau_in, h1, h2, vega)
            #for each pair (m, tau) on the wanted grid return the
            #Nadaraya-Watson estimator
            I_out[i,j] = np.sum(I_in * weights) / np.sum(weights)
    return I_out