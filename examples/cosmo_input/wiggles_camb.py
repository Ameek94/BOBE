import numpy as np
from cobaya.theories.camb.camb import CAMB, CAMBOutputs

class wiggles_camb(CAMB):
    """
    CAMB theory wrapper with a custom primordial power spectrum

    Expects the YAML to provide:
        - Standard LCDM parameters, including As and ns
        - feature paraemters: A_feat, log10omega and phi

    Expects extra_args to include optional structure settings such as:
        - feature_type
        - k_pivot
        - pk_kmin
        - pk_kmax
        - pk_N_min
        - pk_rtol
    """

    def initialize(self):
        extra = dict(getattr(self, "extra_args", {}) or {})

        self.feature_type = extra.pop("feature_type", "none")
        self.k_pivot = extra.pop("k_pivot", 0.05)
        # Controls for CAMB's primordial spline construction
        self.pk_kmin = extra.pop("pk_kmin", 1e-6)
        self.pk_kmax = extra.pop("pk_kmax", 100.0)
        self.pk_N_min = extra.pop("pk_N_min", 4000)
        self.pk_rtol = extra.pop("pk_rtol", 1e-10)

        self.use_non_linear_ratio = False
        
        self.extra_args = extra

        super().initialize()

    def get_can_support_params(self):
        params = super().get_can_support_params()
        return params + ["log10A_feat", "log10lambda_feat", "phi"]

    def get_requirements(self):
        reqs = list(super().get_requirements())
        extra = [
            "H0",
            "ombh2",
            "omch2",
            "tau",
            "As",
            "ns",
            "log10A_feat",
            "log10lambda_feat",
            "phi",
        ]
        for p in extra:
            if p not in reqs:
                reqs.append(p)
        return reqs
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        try:
            # Build CAMBparams first, before any transfer functions are computed
            feature_keys = {"log10A_feat", "log10lambda_feat", "phi"}
            camb_input_params = {
                k: v for k , v in params_values_dict.items()
                if k not in feature_keys
            }
            camb_params = self.set(camb_input_params, state)
            if not camb_params:
                return False
            
            As = params_values_dict["As"]
            ns = params_values_dict["ns"]
            log10A_feat = params_values_dict["log10A_feat"]
            log10lambda_feat = params_values_dict["log10lambda_feat"]
            phi = params_values_dict["phi"]
            A_feat = 10.0 ** log10A_feat
            log10omega = np.log10(2.0 * np.pi) - log10lambda_feat
            phi_rad = phi

            # Attach the custom PPS BEFORE CAMB computes results
            camb_params.set_initial_power_function(
                    self.primordial_pk,
                    args=(As, ns, A_feat, log10omega, phi_rad),
                    kmin=self.pk_kmin,
                    kmax=self.pk_kmax,
                    N_min=self.pk_N_min,
                    rtol=self.pk_rtol,
                )
            camb_params.InitPower.effective_ns_for_nonlinear = ns

            # Full CAMB calculation from the modified params
            results = self.camb.get_transfer_functions(
                camb_params,
                only_time_sources=self.needs_perts,
            )
            results.power_spectra_from_transfer()

            #results = self.camb.get_results(camb_params)        

            # Fill requested theory products
            if self.collectors or "sigma8" in self.derived_extra:
                for product, collector in self.collectors.items():
                        if collector:
                            state[product] = collector.method(
                                results, *collector.args, **collector.kwargs
                            )
                            if collector.post:
                                state[product] = collector.post(*state[product])
                        else:
                            state[product] = results.copy()
        except self.camb.baseconfig.CAMBError as e:
            if self.stop_at_error:
                self.log.error(
                    "Computation error (see traceback below)! "
                    "Parameters sent to CAMB: %r and %r.\n"
                    "To ignore this kind of error, make 'stop_at_error: False'.",
                    dict(state["params"]),
                    dict(self.extra_args),
                )
                raise
            else:
                self.log.debug(
                    "Computation of cosmological product failed. "
                    "Assigning 0 likelihood and moving on. "
                    "The output of the CAMB error was %s",
                    e,
                )
                return False
            
        intermediates = CAMBOutputs(
            camb_params, results, results.get_derived_params() if results else None
        )
        if want_derived:
                state["derived"] = self._get_derived_output(intermediates)
        
        state["derived_extra"] = {
             p: self._get_derived(p, intermediates) for p in self.derived_extra
        }

    def primordial_pk(self, k, As, ns, A_feat, log10omega, phi):
        """
        Primordial scalar power spectrum with log or lin oscillatory features

        Parameters
        ----------
        k : float or array-like
            Wavenumber(s) in Mpc^-1
        As: float
            Scalar amplitude at the pivot scale k0
        ns: float
            Scalar spectral index
        A_feat: float
            Oscillation amplitude
        log10omega: float
            Base-10 logarithm of the oscillation frequency.
        phi: float
            Oscillation phase in radians
        feature_type: {"log, "linear", "none"}, optional
            Type of oscillatory feature.
            - "log": cos(omega * log(k / k0) + phi)
            - "linear": cos(omega * (k / k0) + phi)
            - "none": no oscillatory modulation
        k0: float, optional
            Pivot scale in Mpc^-1. Default is 0.05

        Returns
        -------
        float or np.ndarray
            Primordial power spectrum evaluated at k.
        """
        k = np.asarray(k, dtype=np.float64)
        k0 = self.k_pivot
        omega = 10.0 ** log10omega

        pk = As * (k / k0) ** (ns - 1.0)
        if self.feature_type == 'log':
            modulation = 1.0 + A_feat * np.cos(omega * np.log(k / k0) + phi)
        elif self.feature_type == 'linear':
            modulation = 1.0 + A_feat * np.cos(omega * (k / k0) + phi)
        elif self.feature_type == 'none':
            modulation = 1.0
        else:
            raise ValueError(f"Unkown feature type: {self.feature_type}")

        pk *= modulation
        if np.any(~np.isfinite(pk)):
            raise ValueError("PPS became non-finite")
        if np.any(pk <= 0.0):
            raise ValueError("PPS became non-positive. Check A_feat and other feature parameters")
        
        return pk.item() if pk.ndim == 0 else pk




