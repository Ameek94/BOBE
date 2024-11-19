import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import glob
import contextlib
from PIL import Image


def png2gif(curr_step, loglike_name, ndim):
    log.info(f"Generating GIF from plot frames: GIFs/{ndim}D/{loglike_name}_{curr_step}steps_{ndim}D.gif")
    fp_in = []
    for i in range(1, curr_step):
        fp_in.append(f"GIFs/{ndim}D/{loglike_name}_step_{i}.png")
    fp_out = f"GIFs/{loglike_name}_{curr_step}steps_{ndim}D.gif"
    
    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:
    
        # lazily load images
        img, *imgs = [Image.open(f) for f in fp_in]
        
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=300, loop=1)


def acq_check_metric_plot(BOBE, curr_step):
    ax = BOBE.ax
    acq_goal = BOBE.acq_goal
    acq_check_plot = BOBE.plot_data['acq_check']
    x = np.linspace(0, curr_step, curr_step)
    #ax[0, 2].loglog(x, acq_check_plot, label="Acq Check", marker="x")
    ax[0, 2].plot(x, acq_check_plot, label="Acq Check", marker="x")
    ax[0, 2].axhline(y = acq_goal, xmin=0, xmax = curr_step, linestyle="dotted", label="Acq Goal")
    ax[0, 2].set_title("Convergence Metrics")
    
    return ax


def FBGP_hyperparameter_plot(BOBE, curr_step):
    ax = BOBE.ax
    FBGP_outputscale = BOBE.plot_data['outputscale']
    FBGP_lengthscales = BOBE.plot_data['lengthscales']
    ndim = np.array(FBGP_lengthscales).shape[-1]
    ### Outputscale ###
    ax[1, 0].plot(np.linspace(0, curr_step, curr_step), FBGP_outputscale, marker="o")
    ax[1, 0].text(0.1, 0.99, f"Max: {np.max(FBGP_outputscale[-1])} Min: {np.min(FBGP_outputscale[-1])}", ha='left', va='top', transform=ax[1, 0].transAxes)
    ax[1, 0].set_xlabel("Step Number")
    ax[1, 0].set_ylabel("Outputscale")
    ax[1, 0].set_title("FBGP Outputscale")
    
    if ndim == 2:
        ### Lengthscales ###
        for l in range(0, ndim):
            ax[1, l+1].plot(np.linspace(1, curr_step, curr_step), np.array(FBGP_lengthscales)[..., l], marker="o")
            ax[1, l+1].text(0.1, 0.90, f"Max: {np.array(FBGP_lengthscales)[..., l][-1].max()} Min: {np.array(FBGP_lengthscales)[:, l][-1].min()}", ha='left', va='top', transform=ax[1, l+1].transAxes)
            ax[1, l+1].set_xlabel("Step Number")
            ax[1, l+1].set_ylabel(f"Lengthscale x{l+1}")
            ax[1, l+1].set_title(f"FBGP Lengthscale x{l+1}")

    if ndim > 2:
        for l in range(0,  ndim):
            #We want to alternate between each plot until we run out of lengthscales
            if l%2 == 0: #Even for [1, 1]
                lengthscale = np.mean(np.array(FBGP_lengthscales)[..., l], axis=1)
                ax[1, 1].plot(np.linspace(1, curr_step, curr_step),lengthscale, marker="o", label=f"Lengthscale x{l+1}")
                #We want to move the text label down each time we plot again on the same plot we know there have been n/2 + 1 even numbers
                ax[1, 1].text(0.1, 0.99-(l/2)*(.05), f"Max: {np.array(FBGP_lengthscales)[..., l][-1].max()} Min: {np.array(FBGP_lengthscales)[:, l][-1].min()}", ha='left', va='top', transform=ax[1, 1].transAxes)
            if l%2 == 1: #Odd for [1, 2]
                lengthscale = np.mean(np.array(FBGP_lengthscales)[..., l], axis=1)
                ax[1, 2].plot(np.linspace(1, curr_step, curr_step),lengthscale, marker="o", label=f"Lengthscale x{l+1}")
                #We want to move the text label down each time we plot again on the same plot we know there have been floor(n/2 + 1) odd numbers
                ax[1, 2].text(0.1, 0.99-(np.floor(l/2))*(.05), f"Max: {np.array(FBGP_lengthscales)[..., l][-1].max()} Min: {np.array(FBGP_lengthscales)[:, l][-1].min()}", ha='left', va='top', transform=ax[1, 2].transAxes)
        ax[1, 1].set_title('FBGP Lengthscales')
        ax[1, 1].sharey(ax[1, 2])
        ax[1, 1].legend(loc='lower left')
        ax[1, 2].set_title('FBGP Lengthscales')
        ax[1, 2].legend(loc='lower left')
                        
    return ax

def integral_metrics_plot(BOBE, curr_step, integral_true_accuracy = None):
    ax = BOBE.ax
    acq_goal = BOBE.acq_goal
    integral_accuracy = BOBE.integral_accuracy
    if integral_true_accuracy != None:
        ax[2, 0].plot(np.linspace(0, curr_step, curr_step), np.array(integral_true_accuracy)[:, 0], marker="o", label="Nested Sampler Integral - Analytic Integral")
    
    ax[2, 0].plot(np.linspace(0, curr_step, curr_step), [(a - b)/2 for a, b in zip(integral_accuracy['upper'], integral_accuracy['lower'])], marker="o", label="(BOBE Upper - BOBE Lower)/2")
    ax[2, 0].plot(np.linspace(0, curr_step, curr_step), integral_accuracy['dlogz sampler'], marker="o", label="Intrinsic Error in Nested Sampler")
    ax[2, 0].axhline(y = acq_goal, xmin=0, xmax = curr_step, linestyle="dotted", label="Precision Goal")
    ax[2, 0].set_yscale('log')
    ax[2, 0].legend()
    ax[2, 0].sharex(ax[1, 1])
    ax[2, 0].set_title("Comparison to Analytic Integral")

    return ax

def FBGP_mll_plot(BOBE, curr_step):
    ax = BOBE.ax
    computed_mlls = BOBE.plot_data['mll']
    
    ax[2, 1].plot(np.linspace(1, curr_step, curr_step), np.array(computed_mlls).reshape(curr_step, -1), marker="o")
    ax[2, 1].set_xlabel("Step Number")
    ax[2, 1].set_ylabel("Exact Marginal Log Likelihood")
    ax[2, 1].set_title("Exact Marginal Log Likelihoood")

    return ax

def integral_accuracy_plot(BOBE, curr_step, logz_truth=None, prior_fac=None):
    ax = BOBE.ax
    acq_goal = BOBE.acq_goal
    log_z_mean = BOBE.integral_accuracy['mean']
    log_z_upper = BOBE.integral_accuracy['upper']
    log_z_lower = BOBE.integral_accuracy['lower']
    
    ax[2, 2].plot(np.linspace(0, curr_step, curr_step), log_z_mean, label="Mean", marker="x")
    ax[2, 2].plot(np.linspace(0, curr_step, curr_step), log_z_upper, linestyle="dotted", label="Upper", marker="x")
    ax[2, 2].plot(np.linspace(0, curr_step, curr_step), log_z_lower, linestyle="dotted", label="Lower", marker="x")
    if logz_truth != None and prior_fac != None:
        ax[2, 2].axhline(y = logz_truth - prior_fac, xmin=0, xmax = curr_step, linestyle="dotted", label="Real Value")
        ax[2, 2].axhline(y = (logz_truth - prior_fac) + acq_goal, xmin=0, xmax = curr_step, linestyle="dotted", label="Precision Goal Upper")
        ax[2, 2].axhline(y = (logz_truth - prior_fac) - acq_goal, xmin=0, xmax = curr_step, linestyle="dotted", label="Precision Goal Lower")
    ax[2, 2].legend()
    ax[2, 2].sharex(ax[1, 1])
    ax[2, 2].set_title("Integral Precision")

    return ax


def timing_plot(BOBE, curr_step):
    ax = BOBE.ax
    timing = BOBE.timing
    
    ax[3, 1].plot(np.linspace(1, curr_step, curr_step), timing['NS'], linestyle="dotted", marker="x", label="Nested Sampling")
    ax[3, 1].plot(np.linspace(1, curr_step, curr_step), timing['ACQ'], linestyle="dotted", marker="x", label="Acqusition Function")
    ax[3, 1].plot(np.linspace(1, curr_step, curr_step), timing['Likelihood'], linestyle="dotted", marker="x", label="Likelihood Evaluation")
    ax[3, 1].plot(np.linspace(1, curr_step, curr_step), timing['GP'], linestyle="dotted", marker="x", label="GP Training")
    ax[3, 1].set_title("Timing")
    ax[3, 1].legend()

    return ax