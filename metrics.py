from prophet.python import fbprophet
from prophet.python.fbprophet import models
from prophet.python.fbprophet import plot
from prophet.python.fbprophet import diagnostics

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pkg_resources
import os
from pathlib import Path
import numpy as np
import scipy
from copy import deepcopy
import tqdm

fit_kwargs = {'adapt_delta': 0.9, 'max_treedepth': 11, 'adapt_kappa': 0.75}


def lppd(model, s=None):
    
    return np.log(np.exp(model.params['log_lik']).mean(axis=0)).sum()


def aic(model):
    k = model.params['beta'].shape[1] + model.params['delta'].shape[1] + 2
    return -2*lppd(model) + 2*k


def log_lik_bayes_theta(model):
    mu = model.predict()['yhat']/model.y_scale
    y = model.history['y']/model.y_scale
    sigma = model.params['sigma_obs'].mean()
    return scipy.stats.norm.logpdf(y, loc=mu, scale=sigma).sum()


def dic(model):  # Vois olla nopeempi, nyt log_lik_bayes laskettu kahdesti
    return -2*log_lik_bayes_theta(model) + 2*p_dic(model)


def dic_alt(model):
    return -2*log_lik_bayes_theta(model) + 2*p_dic_alt(model)


def waic(model):
    #ll = model.params['log_lik']
    #pwaic2 = 1/(model.mcmc_samples - 1)*((ll - ll.mean(axis=0))**2).sum(axis=0).sum()
    return -2*lppd(model) + 2*p_waic_2(model)

def loo_cv(model):
    k = model.history.shape[0]
    return k_fold_loo_cv(model, k)

def k_fold_loo_cv(model, k=10):  # Oletetaan, että residuaalit iid
    og_lppd = lppd(model)
    data = model.history.sample(frac=1).reset_index(drop=True)  # shuffle
    breakpoints = np.linspace(0, data.shape[0], k+1, endpoint=True).astype(int)

    #scipy.stats.norm.pdf(p, loc=df['y'].values, scale=m1.params['sigma_obs'][:, np.newaxis])  # TODO: muista skaalaus
    def train_test_split(i):  # OK
        test = data[breakpoints[i]:breakpoints[i+1]].sort_values(by='t')#.reset_index().drop(columns='index')
        train = data.drop(test.index, axis=0).sort_values(by='t').reset_index(drop=True)
        test = test.reset_index(drop=True)
        return (train, test) 

    res = 0
    lppds = []
    params = []
    for i in range(k):
    #for i in tqdm.tqdm(range(k)):
        #print(f'CV {i}')
        train, test = train_test_split(i)
        clean_model = prophet_copy(model)
        fit = clean_model.fit(train, control=fit_kwargs)  # Voiko olla ongelma, että t scale muuttuu?
        params.append(fit.params.copy())
        p = predict_with_samples(fit, test)  
        lik = scipy.stats.norm.pdf(p/fit.y_scale, loc=test['y'].values/fit.y_scale, scale=fit.params['sigma_obs'][:, np.newaxis]) # OK
        res += np.log(lik.prod(axis=1).mean())  # OK

        #p_i = predict_with_samples(fit, model.history)
        #b_lik = scipy.stats.norm.pdf(p_i/fit.y_scale, loc=model.history['y'].values/fit.y_scale, scale=fit.params['sigma_obs'][:, np.newaxis])
        #lppd_i = np.log(b_lik.prod(axis=1).mean())
        #mean_lppd_i += np.log(b_lik.prod(axis=1).mean())/k  # Taitaa olla väärin
        lppd_i = 0
        for j in range(k):
            #print(f'lppd {j}')
            train_i, test_i = train_test_split(j)
            p_i = predict_with_samples(fit, test_i)
            lik_i = scipy.stats.norm.pdf(p_i/fit.y_scale, loc=test_i['y'].values/fit.y_scale, scale=fit.params['sigma_obs'][:, np.newaxis])
            lppd_i += np.log(lik_i.prod(axis=1).mean())
        lppds.append(lppd_i)

    mean_lppd_i = sum(lppds) / k

    bias = og_lppd - mean_lppd_i
    lppd_bias_corrected = res + bias
    eff_params = mean_lppd_i - res
    return {'lppd_loo_cv': res, 'lppd_cloo_cv': lppd_bias_corrected, 'bias': bias, 'p_cloo': eff_params, 'fitted_params':params}

def sliced_params(s, stan_fit):
    ex = stan_fit.extract(permuted=False)[:, :, :-1]
    fn_split = pd.Series(stan_fit.flatnames).apply(lambda x: x.split('[')[0])
    var_names = fn_split.unique()
    params = {}
    for name in var_names:
        reshape_mask = fn_split == name
        reshape_dim = reshape_mask.sum()
        params[name] = ex[:s, :, (fn_split == name).values].reshape((s*4, reshape_dim), order='F')
        np.random.shuffle(params[name])
        if name in ['k', 'm', 'sigma_obs']:  # Squeezables
            params[name] = np.squeeze(params[name])
    return params

# TODO: pistä tää kuntoon
def k_fold_loo_cv_sweepable(model, points, max_samples, k=10):  # Oletetaan, että residuaalit iid
    warmup = model.stan_backend.stan_fit.stan_args[0]['warmup']
    og_lppd = lppd(model)
    data = model.history.sample(frac=1).reset_index(drop=True)  # shuffle
    breakpoints = np.linspace(0, data.shape[0], k+1, endpoint=True).astype(int)
    def train_test_split(i):  # OK
        test = data[breakpoints[i]:breakpoints[i+1]].sort_values(by='t')
        train = data.drop(test.index, axis=0).sort_values(by='t').reset_index(drop=True)
        test = test.reset_index(drop=True)
        return (train, test) 

    fits = pd.DataFrame(index=range(k), columns=['fit', 'train', 'test'], dtype=object)
    for i in range(k):
        print(f'Fitting {i}/{k}')
        train, test = train_test_split(i)
        fits.at[i, 'train'] = train.copy()
        fits.at[i, 'test'] = test.copy()

        clean_model = prophet_copy(model)
        fit = clean_model.fit(train, control=fit_kwargs, warmup=warmup)  # Voiko olla ongelma, että t scale muuttuu?
        fits.loc[i, 'fit'] = fit

    results = pd.DataFrame()
    for s in np.linspace(max_samples//points, max_samples, points).astype(int):
        print(f'Calculating for {s}/{max_samples} samples')
        res_s = 0
        lppds_s = []
        for i in range(k):            
            fit_s = fits.loc[i, 'fit']
            test_s = fits.loc[i, 'test']
            #fits.loc[i, 'og_params'] = fit_s.params.copy()
            fit_s.params = sliced_params(s, fit_s.stan_backend.stan_fit)
            p = predict_with_samples(fit_s, test_s)  
            lik = scipy.stats.norm.pdf(p/fit_s.y_scale, loc=test_s['y'].values/fit_s.y_scale, scale=fit_s.params['sigma_obs'][:, np.newaxis]) # OK
            res_s += np.log(lik.prod(axis=1).mean())  # OK
            lppd_i = 0
            for j in range(k):
                train_i, test_i = train_test_split(j)
                p_i = predict_with_samples(fit_s, test_i)
                lik_i = scipy.stats.norm.pdf(p_i/fit_s.y_scale, loc=test_i['y'].values/fit_s.y_scale, scale=fit_s.params['sigma_obs'][:, np.newaxis])
                lppd_i += np.log(lik_i.prod(axis=1).mean())
            lppds_s.append(lppd_i)

        mean_lppd_i = sum(lppds_s) / k
        bias = og_lppd - mean_lppd_i
        lppd_bias_corrected = res_s + bias
        eff_params = mean_lppd_i - res_s
        results = results.append({'lppd_loo_cv': res_s, 'lppd_cloo_cv': lppd_bias_corrected, 'bias': bias, 'p_cloo': eff_params, 'samples':s}, ignore_index=True)
    return results
    

def prophet_copy(m):
    if m.history is None:
        raise Exception('Mallin täytyy olla eka fitattu originaalilla datalla. CV tulee laskea siis vikana')

    if m.specified_changepoints:
        changepoints = m.changepoints
        #if cutoff is not None:
        #    # Filter change points '<= cutoff'
        #    changepoints = changepoints[changepoints <= cutoff]
    else:
        changepoints = None

    # Auto seasonalities are set to False because they are already set in
    # m.seasonalities.
    m2 = m.__class__(
        growth=m.growth,
        n_changepoints=m.n_changepoints,
        changepoint_range=m.changepoint_range,
        changepoints=changepoints,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=m.holidays,
        seasonality_mode=m.seasonality_mode,
        seasonality_prior_scale=m.seasonality_prior_scale,
        changepoint_prior_scale=m.changepoint_prior_scale,
        holidays_prior_scale=m.holidays_prior_scale,
        mcmc_samples=m.mcmc_samples,
        interval_width=m.interval_width,
        uncertainty_samples=m.uncertainty_samples,
        stan_backend=m.stan_backend.get_type()
    )
    #m2.changepoints_t = deepcopy(m.changepoints_t)
    m2.extra_regressors = deepcopy(m.extra_regressors)
    m2.seasonalities = deepcopy(m.seasonalities)
    m2.country_holidays = deepcopy(m.country_holidays)
    return m2    
    
    
def predict_with_samples(model, df):
    """
    In: model, prediction df
    Out: s*n matrix of predictions
    """
    # Fix scaling issues between different datasets
    if 'floor' not in df:
        df['floor'] = 0
    df['t'] = (df['ds'] - model.start) / model.t_scale
    df['y_scaled'] = (df['y'] - df['floor']) / model.y_scale
    
    trend = predict_trend_with_samples(model, df)
    sf = predict_seasonal_components_with_samples(model, df)
    return trend * (1 + sf['multiplicative_terms']) + sf['additive_terms']
    


def predict_seasonal_components_with_samples(model, df): # TODO: make_all antaa train setin kokoisen framen ulos
    # Väärin, tutki miks ei toimi
    
    seasonal_features, _, component_cols, _ = (
        model.make_all_seasonality_features(df)
    )
    X = seasonal_features.values
    data = {}
    for component in component_cols.columns:
        beta_c = model.params['beta'] * component_cols[component].values
        comp = (X @ beta_c.T).T
        if component in model.component_modes['additive']:
            comp *= model.y_scale
        data[component] = comp
    return data
    
def predict_trend_with_samples(model, df):  # TODO: implement also for logistic and flat trends
    
    
    #k = np.nanmean(self.params['k'])
    #m = np.nanmean(self.params['m'])
    #deltas = np.nanmean(self.params['delta'], axis=0)

    #t = np.array(df['t'])
    #if self.growth == 'linear':
    #    trend = self.piecewise_linear(t, deltas, k, m, self.changepoints_t)
    #elif self.growth == 'logistic':
    #    cap = df['cap_scaled']
    #    trend = self.piecewise_logistic(
    #        t, cap, deltas, k, m, self.changepoints_t)
    #elif self.growth == 'flat':
    #    # constant trend
    #    trend = self.flat_trend(t, m)
        
    t_cp = model.changepoints_t
    n_cp = t_cp.size
    n = df.shape[0]
    A = np.zeros([n, n_cp])
    for i in range(n):
        A[i, :] = t_cp <= df.iloc[i, :]['t']  # TODO: katso että kumpikin on timestamp tai float
    
    delta = model.params['delta']
    k = model.params['k']
    m = model.params['m']
    
    trend = df['t'].values * (k + A @ delta.T).T + (m + A @ (-t_cp * delta).T).T
    trend *= model.y_scale

    return trend


def p_dic(model):
    return 2*(log_lik_bayes_theta(model) - model.params['log_lik'].sum(axis=1).mean())
              
def p_waic_2(model):
    ll = model.params['log_lik']
    return ((ll - ll.mean(axis=0))**2).var(axis=0, ddof=1).sum()
    #return 1/(model.mcmc_samples - 1)*((ll - ll.mean(axis=0))**2).sum(axis=0).sum()  # Tuplacheckattava

def p_dic_alt(model):  # Pitäis olla oikein
    return model.params['log_lik'].sum(axis=1).var()*2
               
    

def sweep_samples(model, data, warmup=None, metrics = {'aic':aic, 'dic':dic, 'waic':waic}, points = 30):
#warmup=500
#data = datasets['A']
#model = fbprophet.Prophet(mcmc_samples=1000, seasonality_mode='additive')
#metrics = {'aic':aic, 'dic':dic, 'waic':waic}

    if warmup is None:
        warmup = model.mcmc_samples//2
    max_samples = model.mcmc_samples - warmup
    fit = model.fit(data, control=fit_kwargs, warmup=warmup)
    og_params = model.params.copy()

    res = pd.DataFrame(columns=metrics.keys())

    def sliced_params(s):
        stan_fit = fit.stan_backend.stan_fit
        ex = stan_fit.extract(permuted=False)[:, :, :-1]
        fn_split = pd.Series(stan_fit.flatnames).apply(lambda x: x.split('[')[0])
        var_names = fn_split.unique()
        params = {}
        for name in var_names:
            reshape_mask = fn_split == name
            reshape_dim = reshape_mask.sum()
            params[name] = ex[:s, :, (fn_split == name).values].reshape((s*4, reshape_dim), order='F')
            np.random.shuffle(params[name])
        return params

    
    for s in np.linspace(max_samples//points, max_samples, points).astype(int):
        row = pd.Series()
        row['samples'] = s
        fit.params = sliced_params(s)
        for metric_name, metric in metrics.items():
            row[metric_name] = metric(fit)
        res = res.append(row, ignore_index=True)
    return res

def defalut_cross_val(model, horizon='15w', initial='104w'):
    cv_res = diagnostics.cross_validation(model, horizon=horizon, initial=initial)
    agg = diagnostics.performance_metrics(cv_res, rolling_window=1)
    return {'aggregated':agg, 'preds':cv_res}


def compare_models(ms, data, no_fit=False, verbose=True):
    """Lista malleista ja oikean mallinen df (sis. y, ds, x) -> metriikat malleille.
    Mallit ei saa olla fitattuja."""
    res = {}
    metrics = {
        'AIC': aic, 
        'DIC': dic, 
        'WAIC2': waic, 
        'p_dic': p_dic, 
        'p_dic_alt': p_dic_alt, 
        'p_waic2': p_waic_2, 
        '10-fold_cv': lambda x: k_fold_loo_cv(x, k=10),
        'default_cv': defalut_cross_val
        }
    
    if no_fit:
        fits = ms
    else:
        fits = [m.fit(data, control=fit_kwargs) for m in ms]
    for metric_name, metric in metrics.items():
        res[metric_name] = {}
        if verbose:
            print(f'Evaluating {metric_name}')
        for f in fits:
            value = metric(f)
            res[metric_name][f] = value
            if verbose:
                if 'cv' not in metric_name:
                    print(value)
                else:
                    print('Done')
        if verbose:
            print('')
    return res