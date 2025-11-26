import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim

import torchbnn as bnn
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

from google.colab import drive
drive.mount('/content/drive')


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pandas import DataFrame as df
import fnmatch
import os
import csv
#from lmfit import Model
import sympy as sym
import scipy.optimize as opt
from datetime import datetime
from IPython.display import Latex

import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import norm,cauchy,lognorm
import array as arr
from scipy import stats as st
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
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

#DEFINING WILLINGALE MODEL
def Willingale_if(t, F_a, alpha, T_a):
    if t<T_a:
        return F_a * np.exp(alpha - (t*alpha)/T_a)
    else:
        return F_a * np.power((t / T_a),(-alpha))

def Willingale(t, F_a, alpha, T_a):
    y = np.zeros(t.shape)
    for j in range(len(y)):
        y[j]=Willingale_if(t[j], F_a, alpha, T_a)
    return y

def log_Willingale_if(logt, logFa, alpha, logTa):
    if logt<logTa:
        return logFa + np.log10(np.e) * alpha * (1.0 - 10**logt/(10**logTa))
    else:
        return logFa - alpha * (logt - logTa)

def log_Willingale(logt, logFa, alpha, logTa):
    y = np.zeros(logt.shape)
    for j in range(len(y)):
        y[j]=log_Willingale_if(logt[j], logFa, alpha, logTa)
    return y

#DEFINING CAUCHY LORENTZIAN FUNCTION
def Cauchy_Lorentz(x, x_0, gamma):
    return ( 1 / (np.pi * gamma * (1 + ( (x-x_0) / gamma )**2 )))


header_names=['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']

#GRB_parameters = pd.read_csv("/home/aditi/Downloads/GRB_project/545_GRBs_parameters.csv", header=0, index_col=0)

#GRB_new = pd.read_csv("/content/drive/MyDrive/Astro/LC Reconstruction 2/GRB_segregated_new.csv", header=0)


#ARRAY TO STORE GRB NAMES
Names=[]

GRB_Name = 'GRB060719'


#print(GRBIDs_arr.head())
#for i in range(528,len(GRBIDs_arr)):

    #ARRAYS TO STORE VALUES OF ORIGINAL WILLINGALE PARAMETERS FOR ALL GRBs IN THE LOOP

print(GRB_Name)
#cleaned_data = pd.read_csv("C:/Users/biagi/Desktop/GRB-SFR/LCR/All_GRBs_reconstruction/LC Reconstruction 2/GRBs_cleaned/"+GRB_Name+"_cleaned.csv", verbose=False, skiprows=2, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names, na_filter=True)
trimmed_data = pd.read_csv("/content/drive/MyDrive/GRBs_trimmed/"+GRB_Name+"_trimmed.csv", verbose=False, skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)
#CLEANED DATA CONTAINS FLUX VS TIME DATA OF PROMPT AS WELL AS AFTERGLOW REGION (COMPLETE LC.). PLEASE REFER TO THE DESCRIPTION AT THE BEGINNING.
#TRIMMED DATA CONTAINS FLUX VS TIME DATA OF AFTERGLOW REGION.

#DEFINING DENSITY FACTOR
density_factor = 1

#Here we obtain the fitting parameters.

#Ta is in log scale. Fa in log scale. Alpha is linear scale.
#And tt and tfinal in log scale.


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
    # print(totdensity)

n, bins, patches = plt.hist(log_ts, bins=np.arange(min(log_ts), max(log_ts)+0.1, step=0.1), color='turquoise')
nmean=np.mean(n)
nstd=np.std(n)

edges=np.histogram_bin_edges(log_ts, bins='auto')
#print(edges)

# seed random number generator
#print(n)
#print(np.max(n)/2)
# plt.hist(log_ts, bins=np.arange(min(log_ts), max(log_ts)+0.1, step=0.1), color='turquoise')
plt.xlabel("$\log_{10}\,time$")
plt.ylabel("$N.\,of\,datapoints$")
plt.show()
plt.clf()

gapslist=[]

for ff in range(0,len(log_ts)-1):
    lowbound=log_ts[ff]
    upbound=log_ts[ff+1]

    if np.abs(upbound-lowbound)>=0.03: #np.min(totalgaps): #0.10:

        gapslist.append([lowbound,upbound,np.abs(upbound-lowbound)])

pivotkeep=[]
for ii in range(len(log_recon_t)):
    for jj in range(len(gapslist)):
        if gapslist[jj][0]<=log_recon_t[ii]<=gapslist[jj][1]:
            pivotkeep.append(ii)

#print(pivotkeep)

logtimekeep=[[log_recon_t[f][0]] for f in pivotkeep]

logtimekeeparray=np.array(logtimekeep)

#print(type(log_recon_t))
#print(type(logtimekeeparray))

log_recon_t = logtimekeeparray

#GAP-AWARE reconstruction
gaps = np.diff(log_ts)

#new code for min gap 0.05, assign points based on the no. of original points
min_gap = 0.05
recon_log_t = [log_ts[0]]
total_span = log_ts[-1] - log_ts[0]
if len(ts) > 500:   # densest LC
    fraction = 0.05
elif len(ts) > 250:
    fraction = 0.1
elif len(ts) > 100:
    fraction = 0.3
else:
    fraction = 0.4
n_points = max(20, int(fraction * len(ts)))
for i in range(len(ts) - 1):
    gap_size = log_ts[i+1] - log_ts[i]

    if gap_size > min_gap:
        # allocate points proportional to gap size
        interval_points = max(2, int(n_points * gap_size / total_span))
        interval = np.linspace(log_ts[i], log_ts[i+1], interval_points, endpoint=True)
        recon_log_t.extend(interval[1:])

recon_log_t = np.array(recon_log_t)
recon_t = 10**np.array(recon_log_t)
recon_t = np.unique(recon_t)

log_recon_t = np.log10(recon_t).reshape(-1, 1)
#test_x = (log_recon_t - log_ts_mean) / log_ts_std

#CALCULATING ERRORBAR IN LINEAR SCALE
ts_error = (positive_ts_err - negative_ts_err )/2
fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2

#CALCULATING ERRORBAR IN LOG SCALE

pos_log_fluxes = np.log10(pos_fluxes)
neg_log_fluxes = np.log10(neg_fluxes)


#MinMax Transform before training

log_ts = log_ts.reshape(-1,1)
log_fluxes = log_fluxes.reshape(-1,1)

s_log_ts = MinMaxScaler((0,1)).fit(log_ts)
sc_log_ts = s_log_ts.transform(log_ts)

s_log_fx = MinMaxScaler((0,1)).fit(log_fluxes)
sc_log_fluxes = s_log_fx.transform(log_fluxes)

#s_log_recon_t = MinMaxScaler((0,1)).fit(log_recon_t)
sc_log_recon_t = s_log_ts.transform(log_recon_t)

train_ts = torch.tensor(sc_log_ts, dtype=torch.float32).clone().detach()
train_flux = torch.tensor(sc_log_fluxes, dtype=torch.float32).clone().detach()

from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(sc_log_ts, sc_log_fluxes, test_size=0.3, random_state=42)
y_train = y_train.ravel()
y_val = y_val.ravel()

#Convert to pytorch tensors
train_X = torch.tensor(X_train, dtype=torch.float32).clone().detach()
train_y = torch.tensor(y_train, dtype=torch.float32).clone().detach()
val_X = torch.tensor(X_val, dtype=torch.float32).clone().detach()
val_y = torch.tensor(y_val, dtype=torch.float32).clone().detach()

import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Set to True if performance is more important than strict reproducibility
    torch.use_deterministic_algorithms(True)  # Ensures deterministic results

set_seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn  # Assuming torchbnn is used
import optuna

#activation functions
def get_activation(name: str):
    name = name.lower()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "swish":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: {name}")

def build_bnn_model(trial, input_dim=1, output_dim=2):
    hidden_units_1 = trial.suggest_categorical("hidden_units_1", [32, 64, 128])
    hidden_units_2 = trial.suggest_categorical("hidden_units_2", [32, 64, 128])
    hidden_units_3 = trial.suggest_categorical("hidden_units_3", [16, 32, 64])
    num_layers = trial.suggest_int("num_layers", 1, 3)

    dropout_rate_1 = trial.suggest_uniform("dropout_rate_1", 0.1, 0.4)
    dropout_rate_2 = trial.suggest_uniform("dropout_rate_2", 0.1, 0.4)

    activation = get_activation(trial.suggest_categorical("activation", ["leakyrelu", "tanh", "swish"]))

    PRIOR_MU = 0.0
    PRIOR_SIGMA = 0.1

    layers = [nn.Linear(input_dim, hidden_units_1), activation, nn.Dropout(dropout_rate_1)]
    if num_layers >= 1:
        layers += [bnn.BayesLinear(prior_mu=PRIOR_MU, prior_sigma=PRIOR_SIGMA,
                                   in_features=hidden_units_1, out_features=hidden_units_2),
                   activation, nn.Dropout(dropout_rate_1)]
    if num_layers >= 2:
        layers += [bnn.BayesLinear(prior_mu=PRIOR_MU, prior_sigma=PRIOR_SIGMA,
                                   in_features=hidden_units_2, out_features=hidden_units_3),
                   activation, nn.Dropout(dropout_rate_2)]
        last_in = hidden_units_3
    else:
        last_in = hidden_units_2 if num_layers == 1 else hidden_units_1

    layers += [bnn.BayesLinear(prior_mu=PRIOR_MU, prior_sigma=PRIOR_SIGMA,
                               in_features=last_in, out_features=output_dim)]
    return nn.Sequential(*layers)

def gaussian_nll_loss_weighted(mean, log_var, target, weights):
    var = torch.exp(log_var)
    nll = 0.5 * ((target - mean) ** 2 / var + log_var)
    return torch.sum(weights * nll) / torch.sum(weights)

def weighted_mse_loss(mean, target, weights):
    return torch.sum(weights * (mean - target) ** 2) / torch.sum(weights)

def compute_loss(model, inputs, targets, kl_weight=1e-3, weights=None, lambda_log_var=1e-3):
    outputs = model(inputs)
    mean, log_var = outputs[:, 0], outputs[:, 1]

    if weights is None:
        weights = torch.ones_like(targets)

    weights = weights / torch.sum(weights)

    mse_loss = weighted_mse_loss(mean, targets.squeeze(), weights)
    nll_loss = gaussian_nll_loss_weighted(mean, log_var, targets.squeeze(), weights)

    #KL divergence
    kl_loss_fn = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl = kl_loss_fn(model)

    #Penalize large log-variance
    log_var_penalty = torch.mean(log_var ** 2)
    log_var_penalty_term = lambda_log_var * log_var_penalty

    total_loss = mse_loss + nll_loss + kl_weight * kl + log_var_penalty_term
    return total_loss, mse_loss, nll_loss, kl

def validate_bnn_model(model, val_X, val_y, weights=None, lambda_log_var=1e-3):
    model.eval()
    with torch.no_grad():
        outputs = model(val_X)
        mean, log_var = outputs[:, 0], outputs[:, 1]

        if weights is None:
            weights = torch.ones_like(val_y)
        weights = weights / torch.sum(weights)

        nll_loss = gaussian_nll_loss_weighted(mean, log_var, val_y.squeeze(), weights)
        mse_loss = weighted_mse_loss(mean, val_y.squeeze(), weights)

        #Log-variance penalty
        log_var_penalty = torch.mean(log_var ** 2)

        kl_loss_fn = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        kl_loss = kl_loss_fn(model)

        total_loss = mse_loss + nll_loss + kl_loss + lambda_log_var * log_var_penalty
        return total_loss.item()

def train_bnn_model(model, train_X, train_y, weights = None, epochs=100, batch_size=32, lr=1e-3, kl_weight=1e-3, verbose = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):

        for i in range(0, train_X.size(0), batch_size):

            batch_X = train_X[i:i + batch_size]
            batch_y = train_y[i:i + batch_size]
            batch_w = weights[i:i + batch_size] if weights is not None else None

            optimizer.zero_grad()

            lambda_log_var = 1e-4
            loss, mse_loss, nll_loss, kl_loss = compute_loss(model, batch_X, batch_y, kl_weight=1e-3, lambda_log_var=lambda_log_var)


           # loss, mse_loss, nll_loss, kl_loss = compute_loss(model, batch_X, batch_y, kl_weight, weights=batch_w)

            loss.backward()
            optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, MSE: {mse_loss.item()}, NLL: {nll_loss.item()}, KL: {kl_loss.item()}")

def objective(trial):
    model = build_bnn_model(trial, input_dim=train_X.shape[1], output_dim=2)

    with torch.no_grad():
        base_pred = torch.mean(train_y)
        train_weights = 1.0 / (torch.abs(train_y.squeeze() - base_pred) + 1e-6)
        train_weights = train_weights / torch.sum(train_weights)

        val_base_pred = torch.mean(val_y)
        val_weights = 1.0 / (torch.abs(val_y.squeeze() - val_base_pred) + 1e-6)
        val_weights = val_weights / torch.sum(val_weights)

    train_bnn_model(model, train_X, train_y, weights=train_weights,
                    epochs=100, batch_size=32, lr=1e-3, kl_weight=1e-3)

    return validate_bnn_model(model, val_X, val_y, weights=val_weights)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)




best_trial = study.best_trial
print(f"Best trial: {best_trial.params}")

best_params = best_trial.params


final_model = build_bnn_model(best_trial, input_dim=train_X.shape[1], output_dim=2)


#train on entire data
train_bnn_model(final_model, train_ts, train_flux, epochs=1500, batch_size=32, lr=1e-3, kl_weight=1e-3)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def weighted_mse_loss(predictions, targets, weights):
        return torch.mean(weights * (predictions - targets) ** 2)

def calculate_weights(predictions, targets, method='distance', sigma=1.0):#Calculate weights based on the method:

    if method == 'distance':
        #Calculate distance between prediction and actual value
        distances = torch.abs(predictions - targets)
        weights = 1.0 / (distances + 1e-6)

    elif method == 'uncertainty':
        log_var = predictions[:, 1]
        weights = torch.exp(-log_var)

    else:
        raise ValueError("Unknown weight method: choose either 'distance' or 'uncertainty'")

    weights = weights / torch.sum(weights)

    return weights

def get_lowest_weighted_mse_prediction(model, val_X, val_y, num_samples=1000, method='distance'):
    model.eval()
    all_predictions = []
    all_weighted_mse_losses = []

    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(val_X)
            mean, log_var = outputs[:, 0], outputs[:, 1]

            weights = calculate_weights(mean, val_y, method=method)

            weighted_mse_loss_value = weighted_mse_loss(mean, val_y.squeeze(), weights).item()

            all_predictions.append(mean.cpu().numpy())
            all_weighted_mse_losses.append(weighted_mse_loss_value)

        min_mse_index = np.argmin(all_weighted_mse_losses)
        best_prediction = all_predictions[min_mse_index]
        lowest_weighted_mse = all_weighted_mse_losses[min_mse_index]

    return best_prediction, lowest_weighted_mse

best_prediction, lowest_weighted_mse = get_lowest_weighted_mse_prediction(final_model, val_X, val_y, num_samples=100, method='distance')

print(f"Lowest Weighted MSE: {lowest_weighted_mse}")

plt.scatter(val_X, val_y, label="True Values", alpha=0.6)
plt.scatter(val_X, best_prediction, label="Predicted Values", alpha=0.6)
plt.xlabel("Input Features (val_X)")
plt.ylabel("Output (val_y / Prediction)")
plt.title(f"True vs Predicted Values\nLowest Weighted MSE: {lowest_weighted_mse:.4f}")
plt.legend()
plt.show()


log_recon_t_new = torch.Tensor(sc_log_recon_t).clone().detach()
num_samples = 1000

mu_samples = []
log_var_samples = []

final_model.eval()
with torch.no_grad():
    for _ in range(num_samples):
        outputs = final_model(log_recon_t_new)
        mean, log_var = outputs[:, 0], outputs[:, 1]
     #   log_var = torch.clamp(log_var, min=-5.0, max=2.0)  #restrict variance
        log_var = torch.clamp(log_var, min=-3.0, max=1.0)  #Restrict variance more tightly

        mu_samples.append(mean.cpu().numpy())
        log_var_samples.append(log_var.cpu().numpy())



    #predictions = mean.cpu().numpy()

mu_samples = np.array(mu_samples)


mean_predictions = np.mean(mu_samples, axis = 0)
std_predictions = np.std(mu_samples, axis=0)
std_predictions = np.sqrt(np.mean(np.exp(log_var_samples), axis=0))

lower_bound = mean_predictions - 1.96 * std_predictions
upper_bound = mean_predictions + 1.96 * std_predictions

plt.plot(train_X, train_y, 'b.', label='Actual Fluxes', alpha=0.5)
plt.plot(sc_log_recon_t, mean_predictions, 'r.', label='Predicted Fluxes', alpha=0.5)
#plt.fill_between(log_recon_t_new.squeeze().cpu(), lower_bound, upper_bound, alpha=0.3, label='95% CI')

plt.xlabel('Time')
plt.ylabel('Flux')
plt.title(f'Light Curves: Actual vs. Predicted')
plt.legend()
plt.show()

recon_pred = s_log_fx.inverse_transform(mean_predictions.reshape(-1,1))


plt.plot(log_ts, log_fluxes, 'b.', label='Actual Fluxes', alpha=0.5)
plt.plot(log_recon_t, recon_pred, 'r.', label='Predicted Fluxes', alpha=0.5)
#plt.fill_between(log_recon_t_new.squeeze().cpu(), lower_bound, upper_bound, alpha=0.3, label='95% CI')

plt.xlabel('Time')
plt.ylabel('Flux')
plt.title(f'Light Curves: Actual vs. Predicted')
plt.legend()
plt.show()

#CALCULATING TIME ERROR IN LINEAR SCALE
ts_error = (positive_ts_err - negative_ts_err)/2

#CALCULATING TIME ERROR IN LOG SCALE
log_ts_error = ts_error/(ts*np.log(10))

errparameters = st.norm.fit(log_ts_error) #GAUSSIAN FITTING ON TIME ERROR DISTRIBUTION
err_dist_time = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_logtimeerr=err_dist_time.rvs(size=len(log_recon_t)) # len(log_ts_error)

#adding noise

fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2
logfluxerrs = fluxes_error/(fluxes*np.log(10))


errparameters = st.norm.fit(logfluxerrs) #GAUSSIAN FITTING ON ERROR-BAR DISTRIBUTION
err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])

recon_errorbar=np.abs(err_dist.rvs(size=len(log_recon_t)))

#Point specific noise
point_specific_noise = []
for j in range(len(recon_pred)):
    fitted_dist = norm(loc=recon_pred[j], scale=recon_errorbar[j])
    point_noise = fitted_dist.rvs() - recon_pred[j]
    point_specific_noise.append(point_noise)

point_specific_noise = np.array(point_specific_noise)

#Jiggle reconstructed points
jiggled_points = recon_pred + point_specific_noise


num_samples = 100  # Number of realizations
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
log_fluxes = log_fluxes.flatten()

plt.figure(figsize=(6,6))
plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes-neg_log_fluxes,pos_log_fluxes-log_fluxes], label=r"$\log_{10}\,flux$", linestyle="",zorder=4)
plt.errorbar(log_recon_t, jiggled_points, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, color='yellow',zorder=3, label = "Reconstructed Points")

plt.scatter(log_ts,log_fluxes,label='Observations',zorder=5)
plt.plot(log_recon_t,recon_pred,label='Mean predictions',zorder=2)
plt.fill_between(log_recon_t, ci_95_lower, ci_95_upper,color='orange', alpha=0.5, label='95% Confidence Interval',zorder=1)
plt.legend(loc='lower left')
plt.xlabel('log$_{10}$(Time) (s)',fontsize="15")
plt.ylabel("log$_{10}$(Flux) ($erg$ ${cm^{-2}}$$s^{-1}$)",fontsize="15")
plt.title(f'BNN on '+str(GRB_Name), fontsize=18)
#plt.savefig('/content/drive/MyDrive/BNN/images/'+str(GRB_Name)+".png", dpi=300)
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

Names.append(GRB_Name)


df.to_csv('/content/drive/MyDrive/BNN/csv/'+str(GRB_Name)+'.csv')



best_params = study.best_params

hidden_1 = best_params['hidden_units_1']
hidden_2 = best_params['hidden_units_2']
hidden_3 = best_params['hidden_units_3']
num_layers = best_params['num_layers']

activation_function = best_params['activation']

dropout_rate_1 = best_params['dropout_rate_1']
dropout_rate_2 = best_params['dropout_rate_2']

file_path = '/home/aditi/Downloads/BNN_new_hyperparameters.csv'
write_header = not os.path.exists(file_path)
data = {
    "GRB_Name": [GRB_Name],
    "hidden_1": [hidden_1],
    "hidden_2": [hidden_2],
    "hidden_3": [hidden_3],
    "num_layers": [num_layers],
    "activation_function": [activation_function],
    "dropout_rate_1": [dropout_rate_1],
    "dropout_rate_2": [dropout_rate_2]
}

df = pd.DataFrame(data)

df.to_csv(file_path, mode='a', header=write_header, index=False)

# 5-fold cross validation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_mse_list = []
val_mse_list = []
fold = 1
num_epochs = 2000
for train_index, val_index in kf.split(log_ts):
    print(f"Training fold {fold}...")

    #split Data
    X_train, X_val = log_ts[train_index], log_ts[val_index]
    y_train, y_val = log_fluxes[train_index].reshape(-1,1), log_fluxes[val_index].reshape(-1,1)

    #apply scaling
    scaled_X_train = s_log_ts.transform(X_train).reshape(-1,1)
    scaled_X_val = s_log_ts.transform(X_val).reshape(-1,1)
    scaled_y_train = s_log_fx.transform(y_train).reshape(-1,1)
    scaled_y_val = s_log_fx.transform(y_val).reshape(-1,1)

    train_X = torch.tensor(scaled_X_train, dtype=torch.float32)
    train_y = torch.tensor(scaled_y_train, dtype=torch.float32)
    val_X = torch.tensor(scaled_X_val, dtype=torch.float32)
    val_y = torch.tensor(scaled_y_val, dtype=torch.float32)

    #model using the best hyperparameters from Optuna
    model = build_bnn_model(best_trial, input_dim=X_train.shape[1], output_dim=2)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_bnn_model(model, train_X, train_y,epochs=1500, batch_size=32, lr=1e-3, kl_weight=1e-3, verbose=False)

    #prediction on training set
    model.eval()
    with torch.no_grad():
        train_pred = model(train_X)
        sc_train_preds = train_pred[:,0]

    train_preds = s_log_fx.inverse_transform(sc_train_preds.reshape(-1,1))
    train_mse = mean_squared_error(train_preds, y_train)
    train_mse_list.append(train_mse)

    #prediction on validation set
    model.eval()
    with torch.no_grad():
        val_pred = model(val_X)
        sc_val_preds = val_pred[:,0]

    val_preds = s_log_fx.inverse_transform(sc_val_preds.reshape(-1,1))
    val_mse = mean_squared_error(val_preds, y_val)
    val_mse_list.append(val_mse)

    plt.scatter(X_train, y_train, label = 'actual')
    plt.scatter(X_train,train_preds, label='train')
    plt.scatter(X_val, val_preds, label='val')
    plt.legend()
    plt.show()

    print(f"Fold {fold} - Train MSE: {train_mse:.4f}, Validation MSE: {val_mse:.4f}")
    fold += 1

# Calculate average MSE across folds
avg_train_mse = np.mean(train_mse_list)
avg_val_mse = np.mean(val_mse_list)

print(f"\nAverage Train MSE: {avg_train_mse:.4f}")
print(f"Average Validation MSE: {avg_val_mse:.4f}")


file_path = '/home/aditi/Downloads/BNN_MSE_results.csv'
write_header = not os.path.exists(file_path)
data = {
    "GRB_Name": [GRB_Name],
    "Train_MSE": [avg_train_mse],
    "Validation_MSE": [avg_val_mse],
}

df = pd.DataFrame(data)

df.to_csv(file_path, mode='a', header=write_header, index=False)



