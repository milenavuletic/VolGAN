# VolGAN
Code to accompany the paper "VolGAN: a generative model for arbitrage-free implied volatility surfaces"
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4617536


VolGAN.py contains the necessary functions to train VolGAN, alongside arbitrage penalty calculation functions.
VolGAN-example.py is an example of how to use the file and check the arbitrage penalties in the simulations.

datacleaning.py contains functions used for cleaning up and extracting implied volatility data from the Option Prices file downloaded from OptionMetrics. It also contains smoothing (Nadaraya-Watson and vega-weighted Nadaraya-Watson) functions (and interpolation functions)


datapath contains data.csv file, which is the data file downloaded from OptionMetrics Implied Volatility Surface File
surfacepath contains surfaces_transform.csv file, which has daily implied volatility surfaces on a pre-defined (m,tau) grid, in vector form



For any questions, please email vuletic [at] maths [dot] ox [dot] ac [dot] uk
