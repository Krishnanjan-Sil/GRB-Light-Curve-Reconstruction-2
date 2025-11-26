import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pandas import DataFrame as df
import os
import csv
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import norm,cauchy,lognorm
import array as arr
from scipy import stats as st
from matplotlib import cm
from sklearn.model_selection import KFold

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn import preprocessing

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import Counter
from numpy.random import seed
from numpy.random import randn

from google.colab import drive
drive.mount('/content/drive')


header_names=['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']
GRB_parameters = pd.read_csv("/content/drive/MyDrive/545_GRBs_parameters.csv", header=0, index_col=0)

#import sys

# --- Get GRB index from SLURM ---
#if len(sys.argv) < 2:
  #  raise ValueError("Need GRB index as command line argument")

#task_id = int(sys.argv[1]) - 1   # SLURM arrays start at 1, Python is 0-based

GRB_new = pd.read_csv("/content/drive/MyDrive/GRB_NAMES.csv", header=0, usecols=[0])

#GRBIDs_arr = GRB_new.iloc[:,0]
#ARRAY TO STORE GRB NAMES
#Names=[]

# --- Pick the GRB for this job ---
#GRB_Name = GRBIDs_arr.iloc[task_id]
#print(f"Running TCN for GRB: {GRB_Name} (index {task_id+1})")
GRB_Name = 'GRB060719'

print(GRB_Name)

trimmed_data = pd.read_csv("/content/drive/MyDrive/GRBs_trimmed/"+GRB_Name+"_trimmed.csv", verbose=False, skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)

#print(GRBIDs_arr.head())
#for i in range(528,len(GRBIDs_arr)):
#ARRAYS TO STORE VALUES OF ORIGINAL WILLINGALE PARAMETERS FOR ALL GRBs IN THE LOOP
print(GRB_Name)

#cleaned_data = pd.read_csv("C:/Users/biagi/Desktop/GRB-SFR/LCR/All_GRBs_reconstruction/LC Reconstruction 2/GRBs_cleaned/"+GRB_Name+"_cleaned.csv", verbose=False, skiprows=2, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names, na_filter=True)
#CLEANED DATA CONTAINS FLUX VS TIME DATA OF PROMPT AS WELL AS AFTERGLOW REGION (COMPLETE LC.). PLEASE REFER TO THE DESCRIPTION AT THE BEGINNING.
#TRIMMED DATA CONTAINS FLUX VS TIME DATA OF AFTERGLOW REGION.
#DEFINING DENSITY FACTOR
density_factor = 1

#Here we obtain the fitting parameters.
#Ta is in log scale. Fa in log scale. Alpha is linear scale.
#And tt and tfinal in log scale.

#log_Tfinal = GRB_parameters.loc[GRB_Name, "logTfinal"]
#FETCHING MAXIMUM AND MINIMUM VALUE OF FLUXES AND TIME VALUES FROM TRIMMED DATA
#THESE VALUES ARE IN LINEAR SCALE
max_fluxes = np.max(trimmed_data["flux"])
min_fluxes = np.min(trimmed_data["flux"])
max_ts = np.max(trimmed_data["t"])
min_ts = np.min(trimmed_data["t"])

#ABOVE VALUES IN LOG SCALE
log_max_fluxes = np.log10(max_fluxes)
log_min_fluxes = np.log10(min_fluxes)
log_max_ts = np.log10(max_ts)
log_min_ts = np.log10(min_ts)

#DEFINING THE ERROR VALUES FROM DATA FILE IN LINEAR SCALE
#for time
positive_ts_err = trimmed_data["pos_t_err"]
negative_ts_err = trimmed_data["neg_t_err"]

#for flux
positive_fluxes_err = trimmed_data["pos_flux_err"]
negative_fluxes_err = trimmed_data["neg_flux_err"]

#READING TIME AND FLUXES FROM THE TRIMMED DATA
#THESE VALUES ARE IN LINEAR SCALE
ts, fluxes = trimmed_data["t"].to_numpy(), trimmed_data["flux"].to_numpy()

#ABOVE VALUES IN LOG SCALE
log_ts, log_fluxes = np.log10(ts), np.log10(fluxes)

# ERROR ON THE FLUXES
pos_fluxes= fluxes + positive_fluxes_err
neg_fluxes= fluxes + negative_fluxes_err

# GENERATES TIME VALUES AT EQUAL INTERVALS IN RANGE OF TS IN LINEAR SCALE
# THIS IS TO BE USED FOR GENERATING THE TIMES AT WHICH WE RECONSTRUCT THE LC
# IT IS EQUAL TO THE NUMBER OF DATA POINTS AS THE ORIGINAL LIGHT CURVE
recon_t = np.geomspace(np.min(ts), np.max(ts), density_factor*len(ts))

#ABOVE VALUE IN LOG SCALE
log_recon_t = np.log10(recon_t)
log_recon_t = log_recon_t.reshape(-1,1)
totdensity=len(log_ts)/(max(log_ts)-min(log_ts))

# GAP-AWARE reconstruction: Creating time points with higher density in large gaps
gaps = np.diff(log_ts)
min_gap = 0.05  # Minimum gap size to consider for adding points
recon_log_t = [log_ts[0]]  # Starting with the first time point
total_span = log_ts[-1] - log_ts[0]  # Total time span in log scale
# Adjusting fraction of points based on light curve density
if len(ts) > 500:  # Densest light curve
    fraction = 0.05
elif len(ts) > 250:
    fraction = 0.1
elif len(ts) > 100:
    fraction = 0.3
else:
    fraction = 0.4
n_points = max(20, int(fraction * len(ts)))  # Minimum 20 points
for i in range(len(ts) - 1):
    gap_size = log_ts[i + 1] - log_ts[i]
    if gap_size > min_gap:
        # Allocating points proportional to gap size
        interval_points = max(2, int(n_points * gap_size / total_span))
        interval = np.linspace(log_ts[i], log_ts[i + 1], interval_points, endpoint=True)
        recon_log_t.extend(interval[1:])  # Adding points, excluding the start
recon_log_t = np.array(recon_log_t)
recon_t = 10 ** np.array(recon_log_t)  # Converting back to linear scale
recon_t = np.unique(recon_t)  # Ensuring unique time points
log_recon_t = np.log10(recon_t).reshape(-1, 1)  # Reshaping for model input


#CALCULATING ERRORBAR IN LINEAR SCALE
ts_error = (positive_ts_err - negative_ts_err )/2
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2

#CALCULATING ERRORBAR IN LOG SCALE
pos_log_fluxes = np.log10(pos_fluxes)
neg_log_fluxes = np.log10(neg_fluxes)

import keras
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam

# Initialize He Normal initializer
he_init = initializers.HeNormal()

# Create a Sequential model
model = Sequential()

# Input layer
model.add(Input(shape=(1, 1)))

# Convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, padding='same',
                kernel_initializer=he_init,
                kernel_regularizer=regularizers.l2(0.0001),
                activation='relu'))
model.add(MaxPooling1D(pool_size=1))
#model.add(BatchNormalization())

# First Bidirectional LSTM
model.add(Bidirectional(LSTM(100, kernel_initializer=he_init,
                              kernel_regularizer=regularizers.l2(0.0001),
                              activation='relu', return_sequences=True)))

# Second Bidirectional LSTM
model.add(Bidirectional(LSTM(100, kernel_initializer=he_init,
                              kernel_regularizer=regularizers.l2(0.0001),
                              activation='swish', return_sequences=True)))

# Third Bidirectional LSTM
model.add(Bidirectional(LSTM(100, kernel_initializer=he_init,
                              kernel_regularizer=regularizers.l2(0.0001),
                              activation='swish', return_sequences=True)))

# Output layer
model.add(Dense(1))

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

X_train=log_ts.reshape(-1,1)
y_train=log_fluxes.reshape(-1,1)

#Fit the model
history = model.fit(
    X_train,
    y_train,
    epochs=1000,
    # validation_data=(X_train, y_train),
    batch_size=35,
)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error




log_recon_seq = log_recon_t.reshape(log_recon_t.shape[0], 1, 1)
log_recon_seq_reshaped = log_recon_seq.reshape(log_recon_seq.shape[0], -1)

#Predict on new time
recon_pred = model.predict(log_recon_t.reshape(-1,1)).reshape(-1,1)


#CALCULATING TIME ERROR IN LINEAR SCALE
ts_error = (positive_ts_err - negative_ts_err)/2

#CALCULATING TIME ERROR IN LOG SCALE
log_ts_error = ts_error/(ts*np.log(10))
errparameters = st.norm.fit(log_ts_error)

#GAUSSIAN FITTING ON TIME ERROR DISTRIBUTION
err_dist_time = st.norm(loc=errparameters[0], scale=errparameters[1])
recon_logtimeerr=err_dist_time.rvs(size=len(log_recon_t))

#len(log_ts_error)
#adding noise
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2
logfluxerrs = fluxes_error/(fluxes*np.log(10))
errparameters = st.norm.fit(logfluxerrs)

#GAUSSIAN FITTING ON ERROR-BAR DISTRIBUTION
err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])
recon_errorbar=err_dist.rvs(size=len(log_recon_t))

#Point specific noise
point_specific_noise = []
for j in range(len(recon_pred)):
    fitted_dist = norm(loc=recon_pred[j], scale=recon_errorbar[j])
    point_noise = fitted_dist.rvs() - recon_pred[j]
    point_specific_noise.append(point_noise)
point_specific_noise = np.array(point_specific_noise)

#Jiggle reconstructed points
jiggled_points = recon_pred + point_specific_noise

num_samples = 1000  # Number of realizations
jiggled_realizations = []
for _ in range(num_samples):
    point_specific_noise = []
    for j in range(len(recon_pred)):
        fitted_dist = norm(loc=recon_pred[j], scale=recon_errorbar[j])
        point_noise = fitted_dist.rvs() - recon_pred[j]
        point_specific_noise.append(point_noise)
    jiggled_realizations.append(recon_pred + np.array(point_specific_noise))

jiggled_realizations = np.array(jiggled_realizations)
# Compute mean and 95% confidence intervals
mean_jiggled = np.mean(jiggled_realizations, axis=0)
ci_95_lower = np.percentile(jiggled_realizations, 2.5, axis=0)  # 2.5th percentile
ci_95_upper = np.percentile(jiggled_realizations, 97.5, axis=0)  # 97.5th percentile

ci_95_lower = ci_95_lower.flatten()
ci_95_upper = ci_95_upper.flatten()
log_recon_t = log_recon_t.flatten()
jiggled_points = jiggled_points.flatten()
recon_fluxes_up=recon_pred.flatten()

plt.figure(figsize=(8,6))
plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes-neg_log_fluxes,pos_log_fluxes-log_fluxes], label=r"$\log_{10}\,flux$", linestyle="",zorder=4)
plt.errorbar(log_recon_t, jiggled_points, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, color='yellow',zorder=3, label = "Reconstructed Points")

plt.scatter(log_ts,log_fluxes,label='Observations',zorder=5)
plt.plot(log_recon_t,recon_fluxes_up,label='Mean predictions',zorder=2)
plt.fill_between(log_recon_t, ci_95_lower, ci_95_upper,color='orange', alpha=0.5, label='95% Confidence Interval',zorder=1)
plt.legend(loc='lower left')
plt.xlabel('log$_{10}$(Time) (s)',fontsize="15")
plt.ylabel("log$_{10}$(Flux) ($erg$ ${cm^{-2}}$$s^{-1}$)",fontsize="15")
plt.title(f'CNN-BiLSTM on '+str(GRB_Name), fontsize=18)
plt.savefig('/content/drive/MyDrive/CNN/images/'+str(GRB_Name)+".png", dpi=300)
plt.show()



df = trimmed_data.copy(deep=True)

for k in range(0, len(log_recon_t)):
    new_row = {
        "t": 10**log_recon_t[k],
        "pos_t_err": 10**recon_logtimeerr[k],
        "neg_t_err": 10**recon_logtimeerr[k],
        "flux": 10**jiggled_points[k],
        "pos_flux_err": 10**jiggled_points[k] * np.log(10) * recon_errorbar[k],
        "neg_flux_err": 10**jiggled_points[k] * np.log(10) * recon_errorbar[k]
    }

    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)

#Names.append(GRB_Name)


df.to_csv('/content/drive/MyDrive/CNN/csv/'+str(GRB_Name)+'.csv')


