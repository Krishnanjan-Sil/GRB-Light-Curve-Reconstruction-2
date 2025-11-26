# full_script.py
import os
import gc
import argparse
import logging

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from google.colab import drive
drive.mount('/content/drive/')

# --- args
parser = argparse.ArgumentParser(description="Process GRB Name as input.")
parser.add_argument('--name', type=str, required=True, help="Specify the GRB Name")
parser.add_argument('--grb_type', type=str, required=True, default="")
parser.add_argument('--path', type=str, required=False, default="")
args = parser.parse_args()

GRB_Name = str(args.name)
grb_type = args.grb_type
path = args.path
print(grb_type)

# hyperparams / flags
lr = 1e-4
patience = 30
batch_size = 64
normalize = True
override = False
epochs = 100
threshold = 0.01
times = 5

print(normalize, override)

folder_path = f"/content/drive/MyDrive/Polynomial/Saved_Outputs/"
os.makedirs(folder_path, exist_ok=True)
os.makedirs(f"{folder_path}/Figures", exist_ok=True)
os.makedirs(f"{folder_path}/CSV_data", exist_ok=True)
os.makedirs(f"{folder_path}/MSE_data", exist_ok=True)

logfile = f"{folder_path}/MSE_data/{GRB_Name}_MSE.log"
if os.path.exists(logfile):
    os.remove(logfile)
logging.basicConfig(filename=logfile, level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"\n{GRB_Name}\n")

# --- preprocessing
print("PREPROCESSING...\n")
header_names = ['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']
GRB_parameters = pd.read_csv("/content/drive/MyDrive/GRB_Project/545_GRBs_parameters.csv", header=0, index_col=0)

# trimmed data path exactly as requested
trimmed_data = pd.read_csv(f"/content/drive/MyDrive/GRB_Project/GRBs_trimmed/{GRB_Name}_trimmed.csv")
trimmed_data = trimmed_data.sort_values(by="t").reset_index(drop=True)

density_factor = 1


log_T_a = GRB_parameters.loc[GRB_Name, "logTa_best"]
log_T_a_min = GRB_parameters.loc[GRB_Name, "logTa_min"]
log_T_a_max = GRB_parameters.loc[GRB_Name, "logTa_max"]

log_F_a = GRB_parameters.loc[GRB_Name, "logFa"]
log_F_a_min = GRB_parameters.loc[GRB_Name, "logFa_min"]
log_F_a_max = GRB_parameters.loc[GRB_Name, "logFa_max"]

alpha = GRB_parameters.loc[GRB_Name, "alpha_best"]
alpha_min = GRB_parameters.loc[GRB_Name, "alpha_min"]
alpha_max = GRB_parameters.loc[GRB_Name, "alpha_max"]

log_Tt = GRB_parameters.loc[GRB_Name, "logTt"]
log_Tfinal = GRB_parameters.loc[GRB_Name, "logTfinal"]

# data stats
max_fluxes = np.max(trimmed_data["flux"])
min_fluxes = np.min(trimmed_data["flux"])
max_ts = np.max(trimmed_data["t"])
min_ts = np.min(trimmed_data["t"])
log_max_fluxes = np.log10(max_fluxes)
log_min_fluxes = np.log10(min_fluxes)
log_max_ts = np.log10(max_ts)
log_min_ts = np.log10(min_ts)

positive_ts_err = trimmed_data["pos_t_err"].to_numpy()
negative_ts_err = trimmed_data["neg_t_err"].to_numpy()
positive_fluxes_err = trimmed_data["pos_flux_err"].to_numpy()
negative_fluxes_err = trimmed_data["neg_flux_err"].to_numpy()

ts = trimmed_data["t"].to_numpy()
fluxes = trimmed_data["flux"].to_numpy()
log_ts = np.log10(ts)
log_fluxes = np.log10(fluxes)

pos_fluxes = fluxes + positive_fluxes_err
neg_fluxes = fluxes + negative_fluxes_err
fluxes_err_sym = (pos_fluxes - neg_fluxes) / 2.0

ts_error = (positive_ts_err - negative_ts_err) / 2.0
fluxes_error = (positive_fluxes_err - negative_fluxes_err) / 2.0

pos_log_fluxes = np.log10(pos_fluxes)
neg_log_fluxes = np.log10(neg_fluxes)

log_F_a_err = (log_F_a_max - log_F_a_min) / 2.0
log_T_a_err = (log_T_a_max - log_T_a_min) / 2.0
alpha_err = (alpha_max - alpha_min) / 2.0

recon_t = np.geomspace(np.min(ts), np.max(ts), density_factor * len(ts))
o_log_recon_t = np.log10(recon_t)

# build gaps-based recon time selection
gapslist = []
for ff in range(0, len(log_ts) - 1):
    lowbound = log_ts[ff]
    upbound = log_ts[ff + 1]
    if np.abs(upbound - lowbound) >= 0.03:
        gapslist.append([lowbound, upbound, np.abs(upbound - lowbound)])

pivotkeep = []
for ii in range(len(o_log_recon_t)):
    for jj in range(len(gapslist)):
        if gapslist[jj][0] <= o_log_recon_t[ii] <= gapslist[jj][1]:
            pivotkeep.append(ii)

if len(pivotkeep) > 0:
    logtimekeeparray = np.squeeze(np.array([[o_log_recon_t[f]] for f in pivotkeep]))
    log_recon_t = logtimekeeparray
else:
    log_recon_t = np.log10(recon_t)

# denser reconstruction avoiding large gaps
gaps = np.diff(log_ts)
min_gap = 0.05
recon_log_t = [log_ts[0]]
total_span = log_ts[-1] - log_ts[0]

if len(ts) > 500:
    fraction = 0.05
elif len(ts) > 250:
    fraction = 0.1
elif len(ts) > 100:
    fraction = 0.3
else:
    fraction = 0.4

n_points = max(20, int(fraction * len(ts)))
for i in range(len(ts) - 1):
    gap_size = log_ts[i + 1] - log_ts[i]
    if gap_size > min_gap:
        interval_points = max(2, int(n_points * gap_size / total_span))
        interval = np.linspace(log_ts[i], log_ts[i + 1], interval_points, endpoint=True)
        recon_log_t.extend(interval[1:])

recon_log_t = np.array(recon_log_t)
recon_t = 10 ** np.array(recon_log_t)
recon_t = np.unique(recon_t)
log_recon_t = np.log10(recon_t).reshape(-1, 1)

# model functions
def Willingale_if(t, F_a, alpha, T_a):
    if t < T_a:
        return F_a * np.exp(alpha - (t * alpha) / T_a)
    else:
        return F_a * np.power((t / T_a), (-alpha))

def Willingale(t, F_a, alpha, T_a):
    y = np.zeros(t.shape)
    for j in range(len(y)):
        y[j] = Willingale_if(t[j], F_a, alpha, T_a)
    return y

def log_Willingale_if(logt, logFa, alpha, logTa):
    if logt < logTa:
        return logFa + np.log10(np.e) * alpha * (1.0 - 10 ** logt / (10 ** logTa))
    else:
        return logFa - alpha * (logt - logTa)

def log_Willingale(logt, logFa, alpha, logTa):
    y = np.zeros(logt.shape)
    for j in range(len(y)):
        y[j] = log_Willingale_if(logt[j], logFa, alpha, logTa)
    return y

# prepare arrays for fitting
log_ts = log_ts.reshape(-1, 1).astype(np.float32).ravel()
log_fluxes = log_fluxes.reshape(-1, 1).astype(np.float32).ravel()
log_recon_t = log_recon_t.astype(np.float32).ravel()

# polynomial curve
def curve(x, *coeffs):
    return sum(c * x ** i for i, c in enumerate(coeffs))

# placeholder mappings to avoid NameError if not provided externally
mappings = {}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
storeMSE = pd.DataFrame([], columns=["Train MSE", "Test MSE"])

for i, (trainIndex, testIndex) in enumerate(kf.split(log_ts, log_fluxes)):
    try:
        # use mapping initial guess if available else fall back to ones
        p0 = mappings.get(grb_type, None)
        if p0 is None:
            # default p0 length 3 to match many previous usage; if key exists it should provide correct length
            p0 = np.ones(3)
        popt, _ = curve_fit(curve, log_ts[trainIndex], log_fluxes[trainIndex], p0=p0)
    except Exception as e:
        logging.error(f"Curve not fitted on fold {i}: {e}")
        continue

    trainPred = curve(log_ts[trainIndex].astype(np.float32).ravel(), *popt)
    trainTrue = log_fluxes[trainIndex]
    trainMSE = mean_squared_error(trainPred, trainTrue)

    testPred = curve(log_ts[testIndex].astype(np.float32).ravel(), *popt)
    testTrue = log_fluxes[testIndex]
    testMSE = mean_squared_error(testPred, testTrue)

    storeMSE.loc[i] = [trainMSE, testMSE]
    logger.info(f"Train MSE: {trainMSE} | Test MSE: {testMSE} | Coeffs: {popt}")

# fit on full dataset (use a reasonable default p0)
try:
    popt, _ = curve_fit(curve, log_ts, log_fluxes, p0=[-0.35368742207348686, 0.7964621779786236, -0.025553201003370385])
except Exception:
    popt, _ = curve_fit(curve, log_ts, log_fluxes, p0=np.ones(3))

recon_fluxes_up = curve(log_recon_t.astype(np.float32).ravel(), *popt)

# save MSE table
storeMSE.to_csv(f"{folder_path}/MSE_data/{GRB_Name}_mse.csv", index=False)

# optional quick plot of last fold predictions (if testIndex exists)
try:
    last_train_idx, last_test_idx = trainIndex, testIndex
    popt_fold, _ = curve_fit(curve, log_ts[last_train_idx], log_fluxes[last_train_idx], p0=np.ones(4))
    testPred = curve(log_ts[last_test_idx].astype(np.float32).ravel(), *popt_fold)
    plt.scatter(log_ts[last_test_idx], testPred, c="green", label="pred")
    plt.scatter(log_ts[last_test_idx], testTrue, c="blue", label="true")
    plt.legend()
    plt.title(f"Fold {i} test predictions for {GRB_Name}")
    plt.savefig(f"{folder_path}/Figures/{GRB_Name}_fold_test.png", dpi=300)
    plt.close()
except Exception:
    pass

# JIGGLED POINTS
print("JIGGLING POINTS...\n")
logfluxerrs = fluxes_error / (fluxes * np.log(10) + 1e-12)
errparameters = st.norm.fit(logfluxerrs)
err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])
recon_errorbar = np.abs(err_dist.rvs(size=len(log_recon_t)))

point_specific_noise = []
for j in range(len(recon_fluxes_up)):
    fitted_dist = st.norm(loc=recon_fluxes_up[j], scale=recon_errorbar[j])
    point_noise = fitted_dist.rvs() - recon_fluxes_up[j]
    point_specific_noise.append(point_noise)

point_specific_noise = np.array(point_specific_noise)
jiggled_points = recon_fluxes_up + point_specific_noise
jiggled_points = np.squeeze(jiggled_points)

# realizations for CI
print("CALCULAING CONFIDENCE INTERVAL...\n")
num_samples = 1000
recon_fluxes_up = np.array(recon_fluxes_up)
recon_errorbar = np.array(recon_errorbar)

point_specific_noise = np.random.normal(loc=0, scale=recon_errorbar, size=(num_samples, len(recon_fluxes_up)))
jiggled_realizations = recon_fluxes_up + point_specific_noise
mean_jiggled = np.mean(jiggled_realizations, axis=0)
ci_95_lower = np.percentile(jiggled_realizations, 2.5, axis=0)
ci_95_upper = np.percentile(jiggled_realizations, 97.5, axis=0)

# plotting
print("SAVING PLOT...\n")
plt.errorbar(log_recon_t, jiggled_points, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, label="Reconstructed Points")
plt.errorbar(log_ts, log_fluxes, yerr=np.abs([log_fluxes - neg_log_fluxes, pos_log_fluxes - log_fluxes]), linestyle="")
plt.scatter(log_ts, log_fluxes, label="Observed Points", zorder=5)
plt.plot(log_recon_t, recon_fluxes_up, label="Mean prediction", zorder=2)
plt.fill_between(log_recon_t, np.squeeze(ci_95_lower), np.squeeze(ci_95_upper), alpha=0.5, label='95% Confidence Interval', zorder=1)
plt.legend(loc='lower left')
plt.xlabel('log$_{10}$(Time) (s)', fontsize=12)
plt.ylabel("log$_{10}$(Flux) ($erg$ ${cm^{-2}}$$s^{-1}$)", fontsize=12)
plt.title(f'Polynomial Fitting on {GRB_Name}', fontsize=14)
plt.savefig(f"{folder_path}/Figures/{GRB_Name}.png", dpi=300)
plt.close()

# calculate time errors and augment dataframe
print("SAVING DATAFRAME...")
log_ts_error = ts_error / (ts * np.log(10) + 1e-12)
errparameters_time = st.norm.fit(log_ts_error)
err_dist_time = st.norm(loc=errparameters_time[0], scale=errparameters_time[1])
recon_logtimeerr = err_dist_time.rvs(size=len(log_recon_t))

df_out = trimmed_data.copy(deep=True)

for k in range(0, len(log_recon_t)):
    new_row = {
        "t": 10 ** (log_recon_t[k]),
        "pos_t_err": 10 ** (recon_logtimeerr[k]),
        "neg_t_err": 10 ** (recon_logtimeerr[k]),
        "flux": 10 ** (jiggled_points[k]),
        "pos_flux_err": 10 ** (jiggled_points[k]) * np.log(10) * recon_errorbar[k],
        "neg_flux_err": 10 ** (jiggled_points[k]) * np.log(10) * recon_errorbar[k]
    }
    new_row_df = pd.DataFrame([new_row])
    df_out = pd.concat([df_out, new_row_df], ignore_index=True)

df_out.to_csv(f"{folder_path}/CSV_data/{GRB_Name}.csv", index=False)

gc.collect()
print("DONE.")
