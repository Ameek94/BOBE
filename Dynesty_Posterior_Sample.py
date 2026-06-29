import os
import numpy as np
from dynesty import DynamicNestedSampler
from dynesty.pool import Pool as DynestyPool
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file

cobaya_input_file = 'examples/cosmo_input/LCDM_lite.yaml'
_MODEL = None

def build_bounds_and_info(cobaya_input_file):
    info = yaml_load_file(cobaya_input_file)
    model0 = get_model(info)
    
    param_list = list(model0.parameterization.sampled_params())
    ndim = len(param_list)
    
    param_labels = [model0.parameterization.labels()[k] for k in param_list]
    
    bounds = []
    for p in param_list:
        prior = info["params"][p]["prior"]
        bounds.append((prior["min"], prior["max"]))
    bounds = np.asarray(bounds)

    return param_list, param_labels, bounds

def prior_transform(u, bounds):
    u = np.asarray(u)
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])

def loglike(theta, cobaya_input_file):
    global _MODEL
    if _MODEL is None:
        print(f"[pid {os.getpid()}] building Cobaya model")
        info = yaml_load_file(cobaya_input_file)
        _MODEL = get_model(info)
        print(f"[pid {os.getpid()}] Cobaya model ready")
    theta = np.asarray(theta)
    return float(_MODEL.loglike(theta, return_derived=False))

if __name__ == '__main__':
    param_list, param_labels, bounds = build_bounds_and_info(cobaya_input_file)
    ndim = len(param_list)
    print(param_list)
    print(param_labels)
    
    nproc = 5
    with DynestyPool(
        nproc,
        loglike,
        prior_transform,
        logl_args=(cobaya_input_file,),
        ptform_args=(bounds,),
    ) as pool:
        dns_sampler = DynamicNestedSampler(
            pool.loglike,
            pool.prior_transform,
            ndim=ndim,
            sample="rwalk",
            nlive=1000,
            pool=pool,
            queue_size=nproc,
            use_pool={
                "prior_transform": True,
                "loglikelihood": True,
                "propose_point": True,
                "update_bound": False,
            },
        )
        dns_sampler.run_nested(print_progress=True, 
                               dlogz_init=1e-3,
                               wt_kwargs = {"pfrac": 0.0},
                               stop_kwargs = {"pfrac": 0.0, "evid_thresh": 0.01})
        res = dns_sampler.results

    np.savez(
        'examples/rotation_tests/lcdm_lite_dynesty',
        samples=res.samples,
        logwt=res.logwt,
        logl=res.logl,
        logz=res.logz,
        logzerr=res.logzerr,
        param_list=np.array(param_list, dtype=object),
        param_labels=np.array(param_labels, dtype=object)
        
    )