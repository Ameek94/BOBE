# Initialization methods for the samplers and optimizers using inputs from yaml
#  Default settings in default.yaml

def _gp_init(self):
    # noise, kernel, NUTS, mc_points size
    pass

def _acq_init(self):
    # which acq, needs mc_points or best_val
    pass

def _bo_init(self):
    # skip steps for fit, nested sampler, acq_goal, upper-lower goal...
    pass

def _params_init(self):
    # param names, bounds, labels
    pass