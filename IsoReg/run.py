import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as st
from sklearn.isotonic import IsotonicRegression
from lmfit import minimize, Parameters

from functions import *

header_names = ['t', 'pos_t_err', 'neg_t_err', 'flux', 'pos_flux_err', 'neg_flux_err']
Iso_plots = "/Users/zuza/Downloads/GRB_IsotonicRegression_plots"
Iso_stats = "/Users/zuza/Downloads/GRB_IsotonicRegression_stats"

GRB_parameters = pd.read_csv("/Users/zuza/Downloads/545_GRBs_parameters.csv", header=0, index_col=0)

# THE RESULT TABLE
# Since the process of generating the plots need a lot of memory the plots was
# generated in 100/150 GRB is one group.
# Here is presented the last loop of generating for 400-545 GRBs (with 10% noise) which
# was saved in IsoReg_recon_10p_5.csv file

results_10p = np.zeros((len(GRB_parameters.index[400:]), 21))
i = 0

for grb_id in GRB_parameters.index[400:]:
    GRB_Name = grb_id
    print(grb_id)

    trimmed_data = pd.read_csv("/Users/zuza/Downloads/GRBs_trimmed/"+GRB_Name+"_trimmed.csv", verbose=False, skiprows=1, skip_blank_lines=True, sep=',', dtype=float, header=None, names=header_names)

    # DATA
    density_factor = 1
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

    #SYMMETRIC ERROR VALUE
    fluxes_err_sym= (pos_fluxes - neg_fluxes )/2

    # GENERATES TIME VALUES AT EQUAL INTERVALS IN RANGE OF TS IN LINEAR SCALE
    # THIS IS TO BE USED FOR GENERATING THE TIMES AT WHICH WE RECONSTRUCT THE LC
    # IT IS EQUAL TO THE NUMBER OF DATA POINTS AS THE ORIGINAL LIGHT CURVE
    recon_t = np.geomspace(np.min(ts), np.max(ts), density_factor*len(ts))

    #ABOVE VALUE IN LOG SCALE
    log_recon_t = np.log10(recon_t)
    log_recon_t = log_recon_t.reshape(-1,1)

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
    #CALCULATING ERRORBAR IN LINEAR SCALE
    ts_error = (positive_ts_err - negative_ts_err )/2
    fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2

    #CALCULATING ERRORBAR IN LOG SCALE

    pos_log_fluxes = np.log10(pos_fluxes)
    neg_log_fluxes = np.log10(neg_fluxes)

    #GETTING ERROR DISTRIBUTION
    fluxes_error = (positive_fluxes_err - negative_fluxes_err)/2
    logfluxerrs = fluxes_error/(fluxes*np.log(10))

    errparameters = st.norm.fit(logfluxerrs) #GAUSSIAN FITTING ON ERROR-BAR DISTRIBUTION
    err_dist = st.norm(loc=errparameters[0], scale=errparameters[1])

    recon_errorbar=abs(err_dist.rvs(size=len(log_recon_t)))

    # ISOTONIC REGRESSION STARTS HERE:
    # FIRST FIT (To find which points are outliers and should have weight)

    X_train, y_train = log_ts, log_fluxes
    Ir = IsotonicRegression(y_min=log_min_fluxes, y_max=log_max_fluxes,out_of_bounds='clip')
    Ir.fit(X_train, y_train)
    mean_prediction = Ir.predict(log_recon_t)

    # FIND WEIGHTS TO OUTLIERS

    log_residuals = abs(log_fluxes - Ir.predict(X_train))

    threshold = np.percentile(log_residuals, 95)
    suspicious_mask = log_residuals > threshold

    weights = np.ones_like(y_train, dtype=float)
    weights[suspicious_mask] = 0.1

    # MAKE ISOTONIC REGRESSION WITH WEIGHTS

    Ir2 = IsotonicRegression(y_min=log_min_fluxes, y_max=log_max_fluxes,
                            increasing=False, out_of_bounds='clip')
    Ir2.fit(X_train.reshape(-1, 1), y_train, sample_weight=weights)
    recon_pred = Ir2.predict(log_recon_t)

    
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




    # PLOT
    plt.rcParams["figure.dpi"]=1000
    plt.rcParams["axes.edgecolor"]="black"
    plt.rcParams["axes.linewidth"]=2.0
    plt.errorbar(log_ts, log_fluxes, yerr=[log_fluxes-neg_log_fluxes,pos_log_fluxes-log_fluxes], label=r"$\log_{10}\,flux$", linestyle="",zorder=4)
    plt.errorbar(log_recon_t, jiggled_points, linestyle='none', yerr=np.abs(recon_errorbar), marker='o', capsize=5, color='yellow',zorder=3, label = "Reconstructed Points")

    plt.scatter(log_ts,log_fluxes,label='Observations',zorder=5)
    plt.plot(log_recon_t,recon_pred,label='Mean predictions',zorder=2)
    #plt.fill_between(log_recon_t, ci_95_lower, ci_95_upper,color='orange', alpha=0.5, label='95% Confidence Interval',zorder=1)
    plt.fill_between(log_recon_t, ci_95_lower, ci_95_upper,color='orange', alpha=0.5, label='95% Confidence Interval',zorder=1)
    plt.legend(loc='lower left')
    plt.xlabel('log Time', fontsize="19")
    plt.ylabel("log Flux", fontsize="19")
    _ = plt.title(f"IsoReg on {GRB_Name} (10% noise level)", fontsize="19")
    plt.savefig(f"{Iso_plots}/IsoReg_{GRB_Name}_recon_10p.pdf", bbox_inches='tight', facecolor='w')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    # ERRORS ANALYZE

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


    # ERROR PARAMETERS (LOG)
    log_F_a_err = (log_F_a_max - log_F_a_min)/2
    log_T_a_err=(log_T_a_max - log_T_a_min)/2
    alpha_err = (alpha_max - alpha_min)/2


    log_F_a_err_frac=log_F_a_err/log_F_a #
    alpha_err_frac= alpha_err/alpha #
    log_T_a_err_frac=log_T_a_err/log_T_a #


    print("Error Fraction for original log Fa: "+str(log_F_a_err_frac))
    print("Error Fraction for original alpha: "+str(alpha_err_frac))
    print("Error Fraction for original log Ta: "+str(log_T_a_err_frac))

    xdata = np.append(log_recon_t, log_ts)                   ## XDATA AND YDATA ARE NOW THE NEW TIME AND FLUX DATAPOINTS (ORIGINAL POINTS + RECONSTRUCTED POINTS).
    ydata = np.append(new_points, log_fluxes)

    params=Parameters()    #DEFINITION OF WILLINGALE PARAMETERS AS A REQUIREMENT FOR THE USE OF LMFIT LIBRARY.
    params.add('log_F_a', value=log_F_a)
    params.add('alpha', value=alpha)
    params.add('log_T_a', value=log_T_a)

    result=minimize(get_residual, params, args=(xdata, ydata))
    result.params

    # print(fit_report(result))

    log_F_a_refit = result.params['log_F_a'].value
    alpha_refit = result.params['alpha'].value
    log_T_a_refit = result.params['log_T_a'].value

    # NEW ERRORS ASSOCIATED WITH THE WILLINGALE PARAMETERS
    log_T_a_err_refit = result.params['log_T_a'].stderr
    log_F_a_err_refit = result.params['log_F_a'].stderr
    alpha_err_refit = result.params['alpha'].stderr

    # NEW ERROR FRACTIONS
    log_F_a_err_refit_frac= log_F_a_err_refit/log_F_a_refit
    log_T_a_err_refit_frac= log_T_a_err_refit/log_T_a_refit
    alpha_err_refit_frac= alpha_err_refit/alpha_refit

    print("Error Fraction for reconstructed log Fa: "+str(log_F_a_err_refit_frac))
    print("Error Fraction for reconstructed alpha: "+str(alpha_err_refit_frac))
    print("Error Fraction for reconstructed log Ta: "+str(log_T_a_err_refit_frac))

    log_F_a_err_per_decrease=((abs(log_F_a_err_refit_frac)-abs(log_F_a_err_frac))/(abs(log_F_a_err_frac)))*100   #positive percentage indicates increase in error fraction.
    log_T_a_err_per_decrease=((abs(log_T_a_err_refit_frac)-abs(log_T_a_err_frac))/(abs(log_T_a_err_frac)))*100   #negative percentage indicates decrease in error fraction.
    alpha_err_per_decrease=((abs(alpha_err_refit_frac)-abs(alpha_err_frac))/(abs(alpha_err_frac)))*100

    print("Percentage decrease in Error Fraction of log Fa: "+str(log_F_a_err_per_decrease))
    print("Percentage decrease in Error Fraction of alpha: "+str(alpha_err_per_decrease))
    print("Percentage decrease in Error Fraction of log Ta: "+str(log_T_a_err_per_decrease))

    # THE i-th row in final table results
    results_10p[i, :] = [log_T_a, log_F_a, alpha,
                        log_T_a_err, log_F_a_err, alpha_err,
                        log_T_a_refit, log_F_a_refit, alpha_refit,
                        log_T_a_err_refit, log_F_a_err_refit, alpha_err_refit,
                        log_F_a_err_frac, log_T_a_err_frac, alpha_err_frac,
                        log_F_a_err_refit_frac, log_T_a_err_refit_frac, alpha_err_refit_frac,
                        log_F_a_err_per_decrease, log_T_a_err_per_decrease, alpha_err_per_decrease]
    i += 1


    result_10p_Iso_pd = pd.DataFrame(results_10p, columns=[
        'logTa', 'logFa', 'alpha',
        'logTa_err', 'logFa_err', 'alpha_err',
        'logTa_new', 'logFa_new', 'alpha_new',
        'logTa_err_new', 'logFa_err_new', 'alpha_err_new',
        'logTa_err_frac', 'logFa_err_frac', 'alpha_err_frac',
        'logTa_err_frac_recon', 'logFa_err_frac_recon', 'alpha_err_frac_recon',
        'logTa_err_per_dec', 'logFa_err_per_dec', 'alpha_err_per_dec'])

    result_10p_Iso_pd.insert(loc=0, column='GRBID', value=GRB_parameters.index[400:])
    print(result_10p_Iso_pd)

    result_10p_Iso_pd.to_csv(f"{Iso_stats}/IsoReg_recon_10p_5.csv")
