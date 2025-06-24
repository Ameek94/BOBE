from jaxbo.bo import BOBE
import numpy as np
# from jaxbo.bo_utils import plot_final_samples
from jaxbo.loglike import cobaya_loglike
from jaxbo.bo_utils import input_standardize, input_unstandardize
from getdist import plots, MCSamples, loadMCSamples
import time

cobaya_input_file = './cosmo_input/Planck_lite_BAO_SN_CPL.yaml'

likelihood = cobaya_loglike(cobaya_input_file, confidence_for_unbounded=0.9999995,
        minus_inf=-1e4, noise_std=0.0,name='CPL_lite')

start = time.time()
sampler = BOBE(n_cobaya_init=8, n_sobol_init = 32, 
        miniters=500, maxiters=1500,max_gp_size=1200,
        loglikelihood=likelihood,
        fit_step = 15, update_mc_step = 5, ns_step = 50,
        num_hmc_warmup = 512,num_hmc_samples = 512, mc_points_size = 64,
        resume=True,resume_file='./CPL_lite.npz',
        use_svm=True,svm_use_size=400,svm_threshold=150,svm_gp_threshold=5000,
        logz_threshold=5.,mc_points_method='NUTS',
        lengthscale_priors='DSLP', minus_inf=-1e5,
        return_getdist_samples=True)

gp, ns_samples, logz_dict = sampler.run()
end = time.time()
print(f"Total time taken = {end-start:.4f} seconds")


def plot_final_samples(gp,samples_dict,param_list,param_labels,plot_params=None,param_bounds=None,
                       reference_samples = None,
                       reference_file = None, reference_ignore_rows=0.,reference_label='MCMC',
                       scatter_points=False,markers=None,output_file='output'):
    """
    Plot the final samples from the Bayesian optimization process.

    Arguments
    ----------
    gp : GP object
        The Gaussian process object used for the optimization.
    ns_samples : dict
        The samples from the nested sampling or MCMC process.
    param_list : list
        The list of parameter names.
    param_labels : list
        The list of parameter labels for plotting.
    plot_params : list, optional
        The list of parameters to plot. If None, all parameters will be plotted.
    param_bounds : np.ndarray, optional
        The bounds of the parameters. If None, assumed to be [0,1] for all parameters.
    reference_samples : MCSamples, optional
        The reference getdist MCsamples from the MCMC/Nested Sampling to comparea gainst. 
        If None, will be loaded from the reference_file.
    reference_file : str, optional
        The getdist file root containing the reference samples. If None, will be loaded from the reference_samples.
        If both are None, no reference samples will be plotted.
    reference_ignore_rows : float, optional
        The fraction of rows to ignore in the reference file. Default is 0.0.
    reference_label : str, optional
        The label for the reference samples. Default is 'MCMC'.
    scatter_points : bool, optional
        If True, scatter the training points on the plot. Default is False.
    output_file : str, optional
        The output file name for the plot. Default is 'output'.
    """

    if plot_params is None:
        plot_params = param_list
    ranges = dict(zip(param_list,param_bounds.T))

    samples = samples_dict['x']

    if param_bounds is None:
        param_bounds = np.array([[0,1]]*len(param_list)).T
    samples = input_unstandardize(samples,param_bounds)
    weights = samples_dict['weights']
    gd_samples = MCSamples(samples=samples, names=param_list, labels=param_labels, 
                           ranges=ranges, weights=weights)
    

    plot_samples = [gd_samples]

    if reference_file is not None:
        ref_samples = loadMCSamples(reference_file,settings={'ignore_rows': reference_ignore_rows})
        plot_samples.append(ref_samples)
    elif reference_samples is not None:
        plot_samples.append(reference_samples)

    labels = ['GP',reference_label]

    for label,s in zip(labels,plot_samples):
        print(f"\nParameter limits from {label}")
        for key in plot_params:
            print(s.getInlineLatex(key,limit=1))
    
    ndim = len(plot_params)

    g = plots.get_subplot_plotter(subplot_size=2.5,subplot_size_ratio=1)
    g.settings.legend_fontsize = 18
    g.settings.axes_fontsize=16
    g.settings.axes_labelsize=18
    g.settings.title_limit_fontsize = 14   
    g.triangle_plot(plot_samples, params = plot_params,filled=[True,False],
                    contour_colors=['#006FED', 'black'],contour_lws=[1,1.5],
                    legend_labels=['GP',f'{reference_label}']
                    ,markers=markers,marker_args={'lw': 1, 'ls': ':'},title_limit=1) 
    ax = g.subplots[1,0]
    ax.axvline(x=-1.,ls='--',color='black',lw=1)
    ax.axhline(y=0.,ls='--',color='black',lw=1)
    if scatter_points:
        points = input_unstandardize(gp.train_x,param_bounds)
        for i in range(ndim):
            # ax = g.subplots[i,i]
            for j in range(i+1,ndim):
                ax = g.subplots[j,i]
                ax.scatter(points[:,i],points[:,j],alpha=0.5,color='forestgreen',s=8)

    g.export(output_file+'_samples.pdf')

# plot_final_samples(gp, ns_samples,param_list=sampler.param_list,param_bounds=sampler.param_bounds,
#                    param_labels=sampler.param_labels,output_file=likelihood.name,
#                    reference_file='./cosmo_input/chains/Planck_lite_BAO_SN_mcmc',reference_ignore_rows=0.3,
#                    reference_label='MCMC',scatter_points=True)

plot_samples = [ns_samples]

ref_samples = loadMCSamples('./cosmo_input/chains/Planck_lite_BAO_SN_mcmc',settings={'ignore_rows': 0.3})
plot_samples.append(ref_samples)


plot_parameters = ['w','wa']
# g = plots.get_subplot_plotter(subplot_size=2.5,subplot_size_ratio=1)
g = plots.get_single_plotter(width_inch=6, ratio=4/5)
g.settings.legend_fontsize = 15
g.settings.axes_fontsize=16
g.settings.axes_labelsize=18
g.settings.title_limit_fontsize = 14   
g.plot_2d(plot_samples,param1='w',param2='wa',filled=[True,False],colors=['#006FED', 'black'])
g.add_x_marker(-1.,color='k',ls='--',lw=1)
g.add_y_marker(0.,color='k',ls='--',lw=1)
g.add_legend(['GP', 'MCMC'],legend_loc='upper right')
# g.triangle_plot(plot_samples, params = plot_parameters,filled=[True,False],
#                     contour_colors=['#006FED', 'black'],contour_lws=[1,1.5],
#                     legend_labels=['GP','MCMC']
#                     ,markers={'w': -1., 'wa': 0.},marker_args={'lw': 1, 'ls': ':'},title_limit=1) 
g.export('CPL_w0wa_samples.pdf')


# plot_final_samples(gp, ns_samples,param_list=sampler.param_list,param_bounds=sampler.param_bounds,
#                    param_labels=sampler.param_labels,output_file='CPL_w0wa',plot_params=plot_parameters,
#                    reference_file='./cosmo_input/chains/Planck_lite_BAO_SN_mcmc',reference_ignore_rows=0.3,
#                    reference_label='MCMC',scatter_points=False)


# 2025-04-22 19:33:32,222 INFO:[BO]:  LogZ info: upper=-1228.2622, mean=-1231.9608, lower=-1232.2119, dlogz sampler=0.1881
# 2025-04-22 23:18:12,526 INFO:[BO]:  Final LogZ: upper=-1231.5452, mean=-1231.8537, lower=-1232.1034, dlogz sampler=0.1164
