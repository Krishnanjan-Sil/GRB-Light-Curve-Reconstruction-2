#Import necessary libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.model_selection import KFold

#Mount google drive
from google.colab import drive
drive.mount('/content/drive/')

import torch
import gpytorch


#Create output directory
import os
folder_path = f"/content/drive/MyDrive/Deep_GP/Saved_Outputs/"
os.makedirs(folder_path, exist_ok=True)
os.makedirs(f"{folder_path}/Figures", exist_ok=True)
os.makedirs(f"{folder_path}/CSV_data", exist_ok=True)
os.makedirs(f"{folder_path}/MSE_data", exist_ok=True)


# MODEL CODE

def run_model(grb_name):
    # Start training for a given GRB
    print(f"\n-----TRAINING FOR {grb_name}-----\n")

    from scipy import stats as st
    from gpytorch.constraints import Interval

    GRB_Name = grb_name

    # Load pre-trimmed GRB data
    trimmed_data = pd.read_csv(f"/content/drive/MyDrive/GRB_Project/GRBs_trimmed/{GRB_Name}_trimmed.csv")

    # Standardize column names depending on dataset format
    if len(trimmed_data.columns) == 6:
        trimmed_data.columns = ["t", "pos_t_err", "neg_t_err", "flux", "pos_flux_err", "neg_flux_err"]
    else:
        trimmed_data.columns = ["0", "t", "pos_t_err", "neg_t_err", "flux", "pos_flux_err", "neg_flux_err"]

    # Helper values
    density_factor = 1
    max_fluxes = np.max(trimmed_data["flux"])
    min_fluxes = np.min(trimmed_data["flux"])

    # Time values
    trim_t_val = trimmed_data["t"] if "t" in trimmed_data else trimmed_data["time_sec"]
    max_ts = np.max(trim_t_val)
    min_ts = np.min(trim_t_val)

    # Convert flux + time to log-scale
    log_max_fluxes = np.log10(max_fluxes)
    log_min_fluxes = np.log10(min_fluxes)
    log_max_ts = np.log10(max_ts)
    log_min_ts = np.log10(min_ts)

    # Extract symmetric errorbars in linear scale
    positive_ts_err = trimmed_data["pos_t_err"]
    negative_ts_err = trimmed_data["neg_t_err"]
    positive_fluxes_err = trimmed_data["pos_flux_err"]
    negative_fluxes_err = trimmed_data["neg_flux_err"]

    ts = trim_t_val.to_numpy()
    fluxes = trimmed_data["flux"].to_numpy()

    # Convert values to log-scale
    log_ts = np.log10(ts)
    log_fluxes = np.log10(fluxes)

    # Build flux errorbars in linear scale
    pos_fluxes = fluxes + positive_fluxes_err
    neg_fluxes = fluxes + negative_fluxes_err

    fluxes_err_sym = (pos_fluxes - neg_fluxes) / 2

    # Generate reconstruction time points using geometric spacing
    recon_t = np.geomspace(np.min(ts), np.max(ts), density_factor * len(ts))
    log_recon_t = np.log10(recon_t).reshape(-1, 1)

    # Compute time + flux errors
    ts_error = (positive_ts_err - negative_ts_err) / 2
    fluxes_error = (positive_fluxes_err - negative_fluxes_err) / 2

    pos_log_fluxes = np.log10(pos_fluxes)
    neg_log_fluxes = np.log10(neg_fluxes)

    # PLOT ORIGINAL GRB
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.errorbar(log_ts, log_fluxes, linestyle='none', yerr=[log_fluxes - neg_log_fluxes, pos_log_fluxes - log_fluxes], marker='o', capsize=5, label="Trimmed Data")
    plt.title(GRB_Name)
    print("\n-----ORIGINAL GRB-----\n")
    plt.show()

    # ------------------------------------------
    # NORMALIZATION FOR STABLE DGP TRAINING
    # ------------------------------------------
    log_ts_mean = np.mean(log_ts, keepdims=True)
    log_ts_std = np.std(log_ts, keepdims=True)
    log_ts_norm = (log_ts - log_ts_mean) / log_ts_std

    log_fluxes_mean = np.mean(log_fluxes, keepdims=True)
    log_fluxes_std = np.std(log_fluxes, keepdims=True)
    log_fluxes_norm = (log_fluxes - log_fluxes_mean) / log_fluxes_std

    # Convert to torch tensors
    train_x = torch.tensor(log_ts_norm, dtype=torch.float32)
    train_y = torch.tensor(log_fluxes_norm, dtype=torch.float32)
    log_fluxes_error = torch.tensor(fluxes_error / fluxes, dtype=torch.float32)

    # ---------------------------------------------------
    # SINGLE GP LAYER USED INSIDE THE DEEP GP
    # ---------------------------------------------------
    class SingleLayerGP(gpytorch.models.ApproximateGP):
        def __init__(self, input_dim, inducing_points):
            inducing_points = train_x[:inducing_points]

            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )

            super().__init__(variational_strategy)

            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(lengthscale=2.0, lengthscale_bounds=(0.05, 10.0))
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # ---------------------------------------------------
    # THE DEEP GP MODEL = TWO GP LAYERS STACKED
    # ---------------------------------------------------
    class DeepGP(torch.nn.Module):
        def __init__(self, num_inducing):
            super().__init__()

            # Two-level GP hierarchy
            self.hidden_layer = SingleLayerGP(train_x.shape[-1], inducing_points=35)
            self.final_layer = SingleLayerGP(train_x.shape[-1], inducing_points=25)

            # Gaussian likelihood with constrained noise
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

            self.likelihood.noise_covar.register_constraint("raw_noise", Interval(1e-4, 0.05))
            self.likelihood.noise_covar.initialize(noise=0.01)

        def forward(self, x):
            x = self.hidden_layer(x)
            x = self.final_layer(x.mean)
            return x

    # ---------------------------------------------------
    # TRAINING FUNCTION FOR DEEP GP
    # ---------------------------------------------------
    from torch.optim.lr_scheduler import StepLR

    def train_dgp():
        print("\n-----TRAINING DEEP GP-----\n")

        model = DeepGP(num_inducing=50)
        likelihood = model.likelihood

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.5)

        mll = gpytorch.mlls.VariationalELBO(likelihood, model.final_layer, num_data=train_y.size(0))

        num_iterations = 500

        for i in range(num_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (i + 1) % 20 == 0:
                print(f"Iteration {i + 1}/{num_iterations} - Loss: {loss.item():.4f}")

        model.eval()
        likelihood.eval()

        # ---------------------------------------------------
        # GAP-AWARE RECONSTRUCTION
        # ---------------------------------------------------
        with torch.no_grad():
            gaps = np.diff(log_ts)

            # Minimum allowed gap size in log-time
            min_gap = 0.05
            recon_log_t = [log_ts[0]]
            total_span = log_ts[-1] - log_ts[0]

            # Density allocation rules depending on dataset size
            if len(ts) > 500:
                fraction = 0.05
            elif len(ts) > 250:
                fraction = 0.1
            elif len(ts) > 100:
                fraction = 0.3
            else:
                fraction = 0.4

            n_points = max(20, int(fraction * len(ts)))

            # Allocate new points depending on gap size
            for i in range(len(ts) - 1):
                gap_size = log_ts[i + 1] - log_ts[i]

                if gap_size > min_gap:
                    interval_points = max(2, int(n_points * gap_size / total_span))
                    interval = np.linspace(log_ts[i], log_ts[i+1], interval_points, endpoint=True)
                    recon_log_t.extend(interval[1:])

            # Prepare test_x
            recon_log_t = np.array(recon_log_t)
            recon_t = 10 ** np.array(recon_log_t)
            recon_t = np.unique(recon_t)

            log_recon_t = np.log10(recon_t).reshape(-1, 1)
            log_recon_t = np.sort(log_recon_t, axis=0)

            test_x = torch.tensor(log_recon_t, dtype=torch.float32)
            test_x = (test_x - log_ts.mean()) / log_ts.std()

            observed_pred = likelihood(model(test_x))
            std_prediction = observed_pred.stddev.numpy()
            mean_prediction = observed_pred.mean.numpy()
            lower, upper = observed_pred.confidence_region()

        # ---------------------------------------------------
        # DENORMALIZE PREDICTIONS
        # ---------------------------------------------------
        mean_prediction_denorm = (mean_prediction * log_fluxes_std) + log_fluxes_mean
        lower_denorm = (lower.numpy() * log_fluxes_std) + log_fluxes_mean
        upper_denorm = (upper.numpy() * log_fluxes_std) + log_fluxes_mean

        # Monte Carlo sampling for flux reconstruction noise
        points = []
        for j in range(len(mean_prediction)):
            fitted_dist = st.norm(loc=mean_prediction[j], scale=std_prediction[j])
            point = np.random.choice(fitted_dist.rvs(size=50), size=1) - mean_prediction[j]
            points.append(point)

        points = np.array(points).ravel()
        new_points = mean_prediction + points

        log_reconstructed_flux = (new_points * log_fluxes_std) + log_fluxes_mean

        # Denormalize train_x and train_y
        test_x_denorm = test_x * log_ts.std() + log_ts.mean()
        train_x_denorm = (train_x.numpy() * log_ts_std) + log_ts_mean
        train_y_denorm = (train_y.numpy() * log_fluxes_std) + log_fluxes_mean

        # Reconstruct synthetic error distributions
        fluxes_error = (positive_fluxes_err - negative_fluxes_err) / 2
        logfluxerrs = fluxes_error / (fluxes * np.log(10))

        errparameters = st.norm.fit(logfluxerrs)
        err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])
        recon_errorbar = err_dist.rvs(size=len(log_recon_t))

        # ---------------------------------------------------
        # PLOT RECONSTRUCTED GRB
        # ---------------------------------------------------
        plt.figure(figsize=(8, 6))
        plt.errorbar(train_x_denorm, train_y_denorm, zorder=4,
                      yerr=[log_fluxes - neg_log_fluxes, pos_log_fluxes - log_fluxes], linestyle="")
        plt.errorbar(test_x_denorm, log_reconstructed_flux, linestyle='none',
                      yerr=np.abs(recon_errorbar), marker='o', capsize=5, color='yellow', zorder=3,
                      label="Reconstructed Points")
        plt.scatter(train_x_denorm, train_y_denorm, zorder=5, label='Observed Points')
        plt.plot(test_x_denorm, mean_prediction_denorm, label='Mean Prediction', zorder=2)
        plt.fill_between(test_x_denorm.flatten(), lower_denorm, upper_denorm,
                         alpha=0.5, color='orange', label='95% Confidence Interval', zorder=1)

        plt.legend(loc='lower left')
        plt.xlabel('log$_{10}$(Time) (s)', fontsize="15")
        plt.ylabel("log$_{10}$(Flux) ($erg$ ${cm^{-2}}$$s^{-1}$)", fontsize="15")
        plt.title(f'Deep GP on {GRB_Name}', fontsize="18")
        plt.savefig(f"/content/drive/MyDrive/{GRB_Name}.png", dpi=300)

        print("\n-----RECONSTRUCTED GRB-----\n")
        plt.show()

        # Time error modeling
        log_ts_error = ts_error / (ts * np.log(10))
        errparameters = st.norm.fit(log_ts_error)
        err_dist_time = st.norm(loc=errparameters[0], scale=errparameters[1])
        recon_logtimeerr = err_dist_time.rvs(size=len(log_recon_t))

        df = trimmed_data.copy(deep=True)

        # Save reconstructed LC into CSV
        for k in range(len(log_recon_t)):
            new_row = {
                "t": 10 ** log_recon_t[k],
                "pos_t_err": 10 ** recon_logtimeerr[k],
                "neg_t_err": 10 ** recon_logtimeerr[k],
                "flux": 10 ** log_reconstructed_flux[k],
                "pos_flux_err": 10 ** log_reconstructed_flux[k] * np.log(10) * recon_errorbar[k],
                "neg_flux_err": 10 ** log_reconstructed_flux[k] * np.log(10) * recon_errorbar[k]
            }

            new_row_df = pd.DataFrame(new_row)
            df = pd.concat([df, new_row_df], ignore_index=True)

        df.to_csv(f"{folder_path}/CSV_data/{GRB_Name}.csv", index=False)

    # Call the training function
    train_dgp()

    # ---------------------------------------------------------------
    #                   5-FOLD CROSS VALIDATION
    # ---------------------------------------------------------------
    print("\n-----5K FOLD VALIDATION BEGIN-----\n")

    from sklearn.model_selection import KFold
    import torch.nn.functional as F

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = f"{folder_path}/MSE_data.csv"
    write_header = not os.path.exists(file_path)

    test_mse_scores = []
    train_mse_scores = []

    # Perform 5-fold KFold training and evaluation
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_x, train_y)):
        print(f"Training on fold {fold + 1}/5...")

        train_x_fold, test_x_fold = train_x[train_idx].to(device), train_x[test_idx].to(device)
        train_y_fold, test_y_fold = train_y[train_idx].to(device), train_y[test_idx].to(device)

        model = DeepGP(num_inducing=50).to(device)
        likelihood = model.likelihood.to(device)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model.final_layer, num_data=train_y_fold.size(0))

        num_iterations = 200
        for i in range(num_iterations):
            optimizer.zero_grad()
            output = model(train_x_fold)

            train_pred = output.mean

            # Denormalize predictions and targets for MSE computation
            train_pred_denorm = (train_pred.detach().cpu().numpy() * log_fluxes_std + log_fluxes_mean)
            train_y_denorm = (train_y_fold.detach().cpu().numpy() * log_fluxes_std + log_fluxes_mean)

            train_pred_tensor = torch.tensor(train_pred_denorm, dtype=torch.float32)
            train_y_tensor = torch.tensor(train_y_denorm, dtype=torch.float32)

            loss = -mll(output, train_y_fold)
            loss.backward()
            optimizer.step()

            # Compute train MSE for this iteration
            train_mse_value = F.mse_loss(train_pred_tensor, train_y_tensor).item()
            train_mse_scores.append(train_mse_value)

            if (i + 1) % 50 == 0:
                print(f"Fold {fold + 1} - Iteration {i + 1}/{num_iterations} - Loss: {loss.item():.4f}")

        # Evaluate model
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            observed_pred = likelihood(model(test_x_fold))
            mean_prediction = observed_pred.mean

            test_pred_denorm = (mean_prediction.detach().cpu().numpy() * log_fluxes_std + log_fluxes_mean)
            test_y_fold_denorm = (test_y_fold.detach().cpu().numpy() * log_fluxes_std + log_fluxes_mean)

            test_pred_tensor = torch.tensor(test_pred_denorm, dtype=torch.float32)
            test_y_tensor = torch.tensor(test_y_fold_denorm, dtype=torch.float32)

            test_mse_value = F.mse_loss(test_pred_tensor, test_y_tensor).item()
            test_mse_scores.append(test_mse_value)

        print(f"Fold {fold + 1} - Test MSE: {test_mse_value:.4f}")

    # Compute mean metrics
    mean_train_mse = np.mean(train_mse_scores)
    mean_test_mse = np.mean(test_mse_scores)

    # Save MSE results
    data = {
        "GRB_Name": [GRB_Name],
        "Final_Train_MSE": [mean_train_mse],
        "Final_Validation_MSE": [mean_test_mse]
    }

    df = pd.DataFrame(data)
    df.to_csv(file_path, mode='a', header=write_header, index=False)

    print("\n-----SAVING TRAIN AND TEST MSE-----\n")
    print(f"Mean Train MSE: {mean_train_mse:4f}")
    print(f"Mean Test MSE: {mean_test_mse:.4f}")


#Add GRB names and run
GRBs=["GRB060206"]

for grb_name in GRBs:
  run_model(grb_name)