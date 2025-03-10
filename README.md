# VolGAN
Code to accompany the paper "VolGAN: a generative model for arbitrage-free implied volatility surfaces"
https://www.tandfonline.com/doi/full/10.1080/1350486X.2025.2471317


VolGAN.py contains the necessary functions to train VolGAN, alongside arbitrage penalty calculation functions.
VolGAN-example.py is an example of how to use the file and check the arbitrage penalties in the simulations.

datacleaning.py contains functions used for cleaning up and extracting implied volatility data from the Option Prices file downloaded from OptionMetrics. It also contains smoothing (Nadaraya-Watson and vega-weighted Nadaraya-Watson) functions (and interpolation functions)

Description of the .csv files:
  -datapath contains data.csv file, which is the data file downloaded from OptionMetrics Implied Volatility Surface File
  -surfacepath contains surfaces_transform.csv file, which has daily implied volatility surfaces on a pre-defined (m,tau) grid, in vector form ('flattened', use the detangle_kt and entangle_kt functions for this) 

Unfortunately, I am not able to share the pre-process data. All data used is downloaded from OptionMetrics. Two types of implied volatility data sets can be downloaded from OptionMetrics, and both can be utilised for VolGAN (option prices or implied vol surface, which is pre-smoothed). Processing the raw data to reach a suitable format is relatively straightforward.

Some advice on data processing:
  -If opting for the Option Prices file, make sure to use options which have a non-zero volume traded
  -The repo has .py files which clean and prepare the data from the Option Prices file
  -Implied Volatility Surface file has implied vols on a fixed (delta, tau) grid, and it is pre-smoothed, so fewer manipulations are required
  -To reach a fixed (m, tau) grid, some interpolation/extrapolation is necessary



For any other questions, please email vuletic [at] maths [dot] ox [dot] ac [dot] uk
