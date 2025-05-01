import time
import sys
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from math import sqrt

num_fixed_nuisance_params = int(sys.argv[1])

input_dict = {
    "theory": {
        "camb": {
            "path": "global",
            "extra_args": {
                "halofit_version": "mead",
                "bbn_predictor": "PArthENoPE_880.2_standard.dat",
                "lens_potential_accuracy": 1,
                "num_massive_neutrinos": 1,
                "nnu": 3.044,
                "dark_energy_model": "fluid",
            },
        },
    },
    "likelihood": {
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_NPIPE_highl_CamSpec.TTTEEE": None,
        "planck_2018_lensing.native": None,
        "bao.desi_2024_bao_all": None,
    },
    "params": {
        # Baseline cosmological parameters
        "w": -1.0,
        "omk": 0.0,
        "omch2": {
            "latex": r"\Omega_\mathrm{c} h^2",
            "prior": {"min": 0.11, "max": 0.13},
            "ref": {"dist": "norm", "loc": 1.186384753e-01, "scale": 0.0005},
            "proposal": 0.0005,
        },
        "logA": {
            "prior": {"min": 2.98, "max": 3.10},
            "ref": {"dist": "norm", "loc": 3.040776555e00, "scale": 0.001},
            "proposal": 0.001,
            "latex": r"\log(10^{10} A_\mathrm{s})",
            # "drop": True,
        },
        "As": {
            "value": "lambda logA: 1e-10*np.exp(logA)",
            "latex": r"A_\mathrm{s}",
        },
        "ns": {
            "prior": {"min": 0.94, "max": 0.99},
            "ref": {"dist": "norm", "loc": 9.652909209e-01, "scale": 0.002},
            "proposal": 0.002,
            "latex": r"n_\mathrm{s}",
        },
        "H0": {
            "latex": r"H_0",
            "prior": {"min": 64, "max": 72},
            "ref": {"dist": "norm", "loc": 6.771234910e+01, "scale": 0.05},
            "proposal": 0.05,
        },
        "ombh2": {
            "prior": {"min": 0.021, "max": 0.023},
            "ref": {"dist": "norm", "loc": 2.224954976e-02, "scale": 0.0001},
            "proposal": 0.0001,
            "latex": r"\Omega_\mathrm{b} h^2",
        },
        "tau": {
            "latex": r"\tau_\mathrm{reio}",
            "prior": {"min": 0.03, "max": 0.08},
            "ref": {"dist": "norm", "loc": 5.546101493e-02, "scale": 0.003},
            "proposal": 0.003,
        },
        "A_planck": {
            "prior": {"dist": "norm", "loc": 1, "scale": 0.0025},
            "ref": {"dist": "norm", "loc": 1, "scale": 0.002},
            "proposal": 0.0005,
            "latex": r"y_\mathrm{cal}",
            "renames": "calPlanck",
        },
        "amp_143": {
            "prior": {"dist": "uniform", "min": 0, "max": 50},
            "ref": {"dist": "norm", "loc": 1.778787048e+01, "scale": 1},
            "latex": r"A^{\rm power}_{143}",
            "proposal": 1,
        },
        "amp_217": {
            "prior": {"dist": "uniform", "min": 0, "max": 50},
            "ref": {"dist": "norm", "loc": 1.163383768e+01, "scale": 1},
            "latex": r"A^{\rm power}_{217}",
            "proposal": 1,
        },
        "amp_143x217": {
            "prior": {"dist": "uniform", "min": 0, "max": 50},
            "ref": {"dist": "norm", "loc": 8.542669523e+00, "scale": 1},
            "latex": r"A^{\rm power}_{143\times217}",
            "proposal": 1,
        },
        "n_143": {
            "prior": {"dist": "uniform", "min": 0, "max": 5},
            "ref": {"dist": "norm", "loc": 1.058733547e+00, "scale": 0.2},
            "latex": r"\gamma^{\rm power}_{143}",
            "proposal": 0.2,
        },
        "n_217": {
            "prior": {"dist": "uniform", "min": 0, "max": 5},
            "ref": {"dist": "norm", "loc": 1.546673841e+00, "scale": 0.2},
            "latex": r"\gamma^{\rm power}_{217}",
            "proposal": 0.2,
        },
        "n_143x217": {
            "prior": {"dist": "uniform", "min": 0, "max": 5},
            "ref": {"dist": "norm", "loc": 1.729442742e+00, "scale": 0.2},
            "latex": r"\gamma^{\rm power}_{143\times217}",
            "proposal": 0.2,
        },
        "calTE": {
            "prior": {"dist": "norm", "loc": 1, "scale": 0.01},
            "ref": {"dist": "norm", "loc": 1, "scale": 0.01},
            "proposal": 0.01,
            "latex": r"c_{TE}",
        },
        "calEE": {
            "prior": {"dist": "norm", "loc": 1, "scale": 0.01},
            "ref": {"dist": "norm", "loc": 1, "scale": 0.01},
            "proposal": 0.01,
            "latex": r"c_{EE}",
        },
        "mnu": 0.06,
    },
    "sampler": {
        "mcmc": {
            "drag": False,
            "oversample_power": 0.4,
            "proposal_scale": 1.9,
            "covmat": './chains/Planck_lite_LCDM.covmat', #"./chains/Planck_DESI_LCDM.covmat",
            "Rminus1_stop": 0.01,
            "Rminus1_cl_stop": 0.2,
        },
    },
    "output": f"chains/Planck_DESI_LCDM_{num_fixed_nuisance_params}",
}

nuisance_params = ['A_planck','amp_143','amp_217','amp_143x217','n_143','n_217','n_143x217','calTE','calEE'] 
nuisance_params_dict = params_dict = {
    "A_planck": {
        "value": 1.000262839e+00,
        "latex": r"y_\mathrm{cal}"
    },
    "amp_143": {
        "value": 1.778787048e+01,
        "latex": r"A^{\rm power}_{143}"
    },
    "amp_217": {
        "value": 1.163383768e+01,
        "latex": r"A^{\rm power}_{217}"
    },
    "amp_143x217": {
        "value": 8.542669523e+00,
        "latex": r"A^{\rm power}_{143\times217}"
    },
    "n_143": {
        "value": 1.058733547e+00,
        "latex": r"\gamma^{\rm power}_{143}"
    },
    "n_217": {
        "value": 1.546673841e+00,
        "latex": r"\gamma^{\rm power}_{217}"
    },
    "n_143x217": {
        "value": 1.729442742e+00,
        "latex": r"\gamma^{\rm power}_{143\times217}"
    },
    "calTE": {
        "value": 9.973337615e-01,
        "latex": r"c_{TE}"
    },
    "calEE": {
        "value": 9.981013175e-01,
        "latex": r"c_{EE}"
    },
}

if num_fixed_nuisance_params > 0:
    for i in range(num_fixed_nuisance_params):
        val = nuisance_params_dict[nuisance_params[i]]['value']
        input_dict['params'][nuisance_params[i]] = val

start = time.time()
from cobaya import run
updated_info, sampler = run(input_dict,force=True)
end = time.time()
print(f"Total time taken = {end-start:.4f} seconds")
