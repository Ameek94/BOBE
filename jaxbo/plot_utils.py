import numpy as np
import matplotlib.pyplot as plt
import ast

def summary_plot(sampler, save_file=None, avg_lengthscales=False):
    '''
    input_files: Array[str]
        1D Array of shape [Data File 1, Data File 2]
    save_file: str
        filename to save plot as. If None no file will be saved
    save_dir: str
        directory to save plot in
    '''
    data_dict = {'logz': [], 'hp': [], 'acq': []}
            
    data_dict['hp'] = {'lengthscales': sampler.hyperparameter_data['lengthscales'], 
                       'outputscale': sampler.hyperparameter_data['outputscale'],
                       'mll': sampler.hyperparameter_data['mll']
                      }
    data_dict['acq'] = np.array(sampler.acq_data['pt_and_val'])[:, -1]
    logz_data = {k: [d[k] for d in sampler.logz_data] for k in sampler.logz_data[0]}
    data_dict['logz'] = logz_data
    
    if len(data_dict['logz']['mean'][:-1]) != 0:
        logz_step = (int(len(data_dict['hp']['lengthscales'])/len(data_dict['logz']['mean'][:-1])))
    else:
        logz_step = (int(len(data_dict['hp']['lengthscales'])/len(data_dict['logz']['mean'])))

    print(logz_step)
    
    fig, ax = plt.subplots(7, 1, figsize=(15, 10), sharex=True)
    #LogZ Plots
    ax[0].plot(np.linspace(0, len(data_dict['logz']['mean']), len(data_dict['logz']['mean']))*logz_step, data_dict['logz']['mean'])
    ax[0].set_xlabel('Iteration #')
    #ax[0][num_files].plot(np.linspace(0, len(data_dict['logz'][i]['mean']), len(data_dict['logz'][i]['mean']))*logz_step[i], data_dict['logz'][i]['mean'])
    ax[0].set_ylabel('LogZ') 
    #ax[0].set_yscale('log')
    #∆Logz Plots
    print(f"{0}-> {len(data_dict['logz']['mean']*logz_step)} in {len(data_dict['logz']['mean'])} step(s)")
    ax[1].plot(np.linspace(0, 
                           len(data_dict['logz']['mean']*logz_step), 
                           len(data_dict['logz']['mean'])), 
               np.array(data_dict['logz']['upper']) - np.array(data_dict['logz']['lower']))
    ax[1].axhline(sampler.logz_threshold, ls='--', c='r')
    ax[1].set_xlabel('Iteration #')
    ax[1].set_ylabel('∆LogZ') 
    ax[1].set_yscale('log')
    #HP Plots
    #Lengthscales
    if avg_lengthscales:
        ax[2].plot(np.linspace(0, len(data_dict['hp']['lengthscales']), len(data_dict['hp']['lengthscales'])), np.mean(data_dict['hp']['lengthscales'], axis=1))
    else:
        ax[2].plot(np.linspace(0, len(data_dict['hp']['lengthscales']), len(data_dict['hp']['lengthscales'])), data_dict['hp']['lengthscales'])
    ax[2].set_xlabel('Iteration #')
    ax[2].set_ylabel('Lengthscale')
    ax[2].set_yscale('log')
    #Outputscales
    ax[3].plot(np.linspace(0, len(data_dict['hp']['outputscale']), len(data_dict['hp']['outputscale'])), data_dict['hp']['outputscale'])
    ax[3].set_xlabel('Iteration #')
    ax[3].set_ylabel('Outputscale')
    ax[3].set_xlabel('Iteration #')
    ax[3].set_yscale('log')
    #MLL
    ax[4].plot(np.linspace(0, len(data_dict['hp']['mll']), len(data_dict['hp']['mll'])), data_dict['hp']['mll'])
    ax[4].set_xlabel('Iteration #')
    ax[4].set_ylabel('MLL')
    #ax[4][i].set_yscale('log')
    #Acq Value
    ax[5].plot(np.linspace(0, len(data_dict['acq']), len(data_dict['acq'])), data_dict['acq'])
    ax[5].set_xlabel('Iteration #')
    ax[5].set_ylabel('Acqusition Fn Value')
    #Surprise Factor
    ax[6].plot(np.linspace(0, len(data_dict['acq']), len(data_dict['acq'])), sampler.sample_point_data['del_val']/np.sqrt(sampler.sample_point_data['variance']))
    ax[6].set_xlabel('Iteration #')
    ax[6].set_ylabel(r'$\frac{F(x) - GP(x)}{GP_{\sigma}}$')
    ax[6].set_yscale('log')
    if save_file is not None:
        plt.savefig(f"{save_file}")
    plt.tight_layout()
    plt.show()

def plot_timing(sampler, total_time, output_file='timing_output'):
    plt.figure()
    [plt.plot(v, label=k) for k, v in sampler.timing.items()]
    plt.xlabel('Step Number')
    plt.ylabel('Time (s)')
    plt.yscale('log')
    plt.title(f"Run Time: {total_time:.4f}s, # Samples: {sampler.gp.train_y.size}")
    plt.legend()
    plt.savefig(output_file+'.png')
    plt.show()
    plt.close()
    

def plot_logz(sampler, ref_logz_mean=None, ref_logz_err=None, output_file='logz_output', save_dir=None):
    logzs_dict = {k: [d[k] for d in sampler.logz_data] for k in sampler.logz_data[0]}
    fig, ax = plt.subplots(2, 1)
    x = np.linspace(1, len(sampler.logz_data), len(sampler.logz_data))
    ax[0].plot(x, logzs_dict['mean'], label='mean', ls='--', color='black')
    ax[0].fill_between(x, logzs_dict['upper'], logzs_dict['lower'], alpha=0.5, color='blue', label='∆LogZ')
    ax[1].plot(x, np.array(logzs_dict['upper']) - np.array(logzs_dict['lower']))
    if ref_logz_mean is not None and ref_logz_err is not None:
        ax[0].axhline(xmin=x[0], xmax=x[-1], y=ref_logz_mean, label='Reference')
        ax[0].fill_between(x, ref_logz_mean+ref_logz_err, ref_logz_mean-ref_logz_err, alpha=0.5, color='orange')
        ax[0].set_title(f"BOBE LogZ: {logzs_dict['mean'][-1]:.3f} ± {logzs_dict['upper'][-1] - logzs_dict['lower'][-1]:.3f} \n Reference LogZ: {ref_logz_mean:.3f} ± {ref_logz_err:.3f} ")
    else:
        ax[0].set_title(f"BOBE LogZ: {logzs_dict['mean'][-1]:.3f} ± {logzs_dict['upper'][-1] - logzs_dict['lower'][-1]:.3f}") 
    ax[1].set_yscale('log')
    plt.savefig(output_file+'.png')
    plt.show()
    #plt.close()