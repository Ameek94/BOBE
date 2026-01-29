"""
Kernel implementations for Gaussian Process models.

All kernels inherit from the base Kernel class and implement the covariance() method.
JAX JIT compilation is handled at higher levels (acquisition functions, optimization).
"""

from abc import ABC, abstractmethod
from math import sqrt
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Constants for Matérn kernel
sqrt2 = sqrt(2.)
sqrt3 = sqrt(3.)
sqrt5 = sqrt(5.)


class Kernel(ABC):
    """
    Abstract base class for all kernels in BOBE.
    
    Attributes
    ----------
    lengthscales : jnp.ndarray
        Lengthscale parameters for each dimension, shape (D,)
    kernel_variance : float
        Overall variance/amplitude of the kernel
    noise : float
        Observation noise level
    """
    
    def __init__(self, lengthscales, kernel_variance, noise=1e-8):
        """
        Initialize kernel with hyperparameters.
        
        Parameters
        ----------
        lengthscales : jnp.ndarray
            Lengthscale for each input dimension
        kernel_variance : float
            Kernel variance/amplitude parameter
        noise : float, optional
            Noise level added to diagonal. Default is 1e-8.
        """
        self.lengthscales = jnp.array(lengthscales)
        self.kernel_variance = kernel_variance
        self.noise = noise
    
    def sq_dist(self, xa, xb):
        """
        Compute squared Euclidean distance between two sets of points.
        
        This utility method is used by many kernel implementations.
        
        Parameters
        ----------
        xa : jnp.ndarray
            First set of points, shape (n1, D)
        xb : jnp.ndarray
            Second set of points, shape (n2, D)
            
        Returns
        -------
        sq_dist : jnp.ndarray
            Squared distances, shape (n1, n2)
        """
        return jnp.sum(jnp.square(xa[:, None, :] - xb[None, :, :]), axis=-1)
    
    @abstractmethod
    def covariance(self, xa, xb, include_noise=True):
        """
        Compute covariance matrix between two sets of points.
        
        Parameters
        ----------
        xa : jnp.ndarray
            First set of points, shape (n1, D)
        xb : jnp.ndarray
            Second set of points, shape (n2, D)
        include_noise : bool, optional
            Whether to add noise to diagonal (only when xa is xb). Default is True.
            
        Returns
        -------
        K : jnp.ndarray
            Covariance matrix of shape (n1, n2)
        """
        pass
    
    def diagonal(self, x, include_noise=True):
        """
        Compute only the diagonal of the kernel matrix K(x,x).
        
        For stationary kernels, the diagonal is constant: kernel_variance (+ noise).
        Override this method if your kernel has a non-constant diagonal.
        
        Parameters
        ----------
        x : jnp.ndarray
            Points at which to compute diagonal, shape (n, D)
        include_noise : bool, optional
            Whether to include noise in diagonal. Default is True.
            
        Returns
        -------
        diag : jnp.ndarray
            Diagonal values, shape (n,)
        """
        diag = self.kernel_variance * jnp.ones(x.shape[0])
        if include_noise:
            diag += self.noise
        return diag
    
    def update_hyperparams(self, lengthscales=None, kernel_variance=None, noise=None):
        """
        Update kernel hyperparameters.
        
        Parameters
        ----------
        lengthscales : jnp.ndarray, optional
            New lengthscale values
        kernel_variance : float, optional
            New kernel variance
        noise : float, optional
            New noise level
        """
        if lengthscales is not None:
            self.lengthscales = jnp.array(lengthscales)
        if kernel_variance is not None:
            self.kernel_variance = kernel_variance
        if noise is not None:
            self.noise = noise
    
    def __call__(self, xa, xb, include_noise=True):
        """Convenience method - same as covariance()"""
        return self.covariance(xa, xb, include_noise=include_noise)


class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) / Squared Exponential kernel.
    
    k(x, x') = σ² * exp(-0.5 * ||x - x'||²/ℓ²)
    
    where σ² is kernel_variance and ℓ is lengthscale.
    """
    
    def covariance(self, xa, xb, include_noise=True):
        """
        Compute RBF covariance matrix.
        
        Parameters
        ----------
        xa : jnp.ndarray
            First set of input points, shape (n1, d).
        xb : jnp.ndarray
            Second set of input points, shape (n2, d).
        include_noise : bool, optional
            Whether to include noise on diagonal. Default is True.
            
        Returns
        -------
        jnp.ndarray
            Kernel matrix of shape (n1, n2).
        """
        # Scale inputs by lengthscales
        xa_scaled = xa / self.lengthscales
        xb_scaled = xb / self.lengthscales
        
        # Compute squared distances
        sq_dist = self.sq_dist(xa_scaled, xb_scaled)
        
        # Apply RBF kernel
        K = self.kernel_variance * jnp.exp(-0.5 * sq_dist)
        
        # Add noise to diagonal if needed
        if include_noise:
            K += self.noise * jnp.eye(K.shape[0])
        
        return K


class MaternKernel(Kernel):
    """
    Matérn-5/2 kernel.
    
    k(x, x') = σ² * (1 + √5*d + 5*d²/3) * exp(-√5*d)
    
    where d = ||x - x'||/ℓ, σ² is kernel_variance, and ℓ is lengthscale.
    """
    
    def covariance(self, xa, xb, include_noise=True):
        """
        Compute Matérn-5/2 covariance matrix.
        
        Parameters
        ----------
        xa : jnp.ndarray
            First set of input points, shape (n1, d).
        xb : jnp.ndarray
            Second set of input points, shape (n2, d).
        include_noise : bool, optional
            Whether to include noise on diagonal. Default is True.
            
        Returns
        -------
        jnp.ndarray
            Kernel matrix of shape (n1, n2).
        """
        # Scale inputs by lengthscales
        xa_scaled = xa / self.lengthscales
        xb_scaled = xb / self.lengthscales
        
        # Compute squared distances
        dsq = self.sq_dist(xa_scaled, xb_scaled)
        
        # Safe sqrt to avoid division by zero
        d = jnp.sqrt(jnp.where(dsq < 1e-30, 1e-30, dsq))
        
        # Matérn-5/2 formula
        exp_term = jnp.exp(-sqrt5 * d)
        poly_term = 1. + d * (sqrt5 + d * 5. / 3.)
        K = self.kernel_variance * poly_term * exp_term
        
        # Add noise to diagonal if needed
        if include_noise:
            K += self.noise * jnp.eye(K.shape[0])
        
        return K


class SphericalKernelBase(Kernel):
    """
    Shared helpers for spherical-projection dot-product kernels.
    """

    def __init__(self, lengthscales, kernel_variance=1.0, noise=1e-8):
        super().__init__(lengthscales=lengthscales, kernel_variance=kernel_variance, noise=noise)

        #Paper default is a = sqrt(D)
        self.a = jnp.sqrt(self.lengthscales.shape[0]).astype(float)

    def _softmax_simplex(self, logits):
        """
        Map unconstrained logits -> simplex weights (sum=1, all positive).
        """
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        ex = jnp.exp(logits)
        return ex / jnp.sum(ex, axis=-1, keepdims=True)
    
    def b_simplex(self):
        return self._softmax_simplex(self.b_logits)

    def _sphere_features(self, u):
        """
        Inverse stereographic projection P(z) where:
        x := 2u - 1                          (map unit cube -> [-1,1]^D)
        z := x / (a * lengthscales)
        P(z) := [2z, ||z||^2 - 1] / (||z||^2 + 1)

        u : (n, D) in [0,1]^D
        returns : (n, D+1), rows are unit-norm (up to numerical error)
        """
        # Map unit cube -> centred box (paper convention)
        x = 2.0 * u - 1.0

        # Robust positive scale (only prevents NaNs; does not materially change values)
        ls = jnp.clip(self.lengthscales, 1e-30, jnp.inf)   # super tiny floor
        a  = jnp.clip(self.a,          1e-30, jnp.inf)

        z = x / ls
        z = z / a

        r2 = jnp.sum(z * z, axis=1, keepdims=True)
        inv = 1.0 / (r2 + 1.0)

        head = 2.0 * z
        tail = r2 - 1.0

        return jnp.concatenate([head, tail], axis=1) * inv

    def _sphere_dot(self, xa, xb):
        """
        Dot products on the sphere:
          S_ij = P(xa_i)^T P(xb_j)

        xa : (n1, D), xb : (n2, D)
        returns : (n1, n2)
        """
        Pa = self._sphere_features(xa)   # (n1, D+1)
        Pb = self._sphere_features(xb)   # (n2, D+1)
        return Pa @ Pb.T

    def update_hyperparams(self, lengthscales=None, kernel_variance=None, noise=None, a=None):
        """
        Update base spherical kernel hyperparameters.
        """
        super().update_hyperparams(lengthscales=lengthscales, kernel_variance=kernel_variance, noise=noise)
        if a is not None:
            self.a = a



class SphericalLinearKernel(SphericalKernelBase):
    """
    Spherical linear kernel:

      k(x, x') = kernel_variance * [ b0 + b1 * <P(z), P(z')> ]

    where (b0,b1) are constrained to the simplex via softmax(b_logits).
    For strict paper faithfulness, use kernel_variance=1.0 and do not optimise it.
    """

    def __init__(self, lengthscales, kernel_variance=1.0, noise=1e-8,
                 fixed_b=False, fixed_b_logits=None, fixed_a=False, fixed_a_value=None):
        super().__init__(lengthscales=lengthscales, kernel_variance=kernel_variance, noise=noise)

        self.fixed_b = fixed_b

        if fixed_b:
            if fixed_b_logits is None:
                fixed_b_logits = jnp.array([-10.0, 10.0])
            self.fixed_b_logits = jnp.array(fixed_b_logits)
            self.b_logits = self.fixed_b_logits
        else:
            self.fixed_b_logits = None 
            self.b_logits = jnp.zeros((2,))

    def covariance(self, xa, xb, include_noise=False):
        """
        Compute spherical linear covariance matrix.

        xa : (n1, D)
        xb : (n2, D)
        """
        logits = self.fixed_b_logits if self.fixed_b else self.b_logits
        b = self.b_simplex().reshape(-1) #self._softmax_simplex(logits)   # (2,)
        s = self._sphere_dot(xa, xb)
        K = self.kernel_variance * (b[0] + b[1] * s)

        if include_noise and xa.shape[0] == xb.shape[0]:
            K += self.noise * jnp.eye(K.shape[0])

        return K

    def diagonal(self, x, include_noise=True):
        """
        Since <P(x),P(x)> = 1 and b is on simplex, k(x,x)=kernel_variance.
        """
        diag = self.kernel_variance * jnp.ones(x.shape[0])
        if include_noise:
            diag += self.noise
        return diag

    def update_hyperparams(self, lengthscales=None, kernel_variance=None, noise=None, a=None, b_logits=None):
        super().update_hyperparams(lengthscales=lengthscales, kernel_variance=kernel_variance, noise=noise, a=a)
        if not self.fixed_b and b_logits is not None:
            self.b_logits = jnp.array(b_logits)

class SphericalPolynomialKernel(SphericalKernelBase):
    """
    Spherical polynomial kernel:

      k(x, x') = kernel_variance * sum_{i=0}^m b_i * (<P(z), P(z')>)^i

    where b is constrained to the simplex via softmax(b_logits).
    For strict paper faithfulness, use kernel_variance=1.0 and do not optimise it.
    """

    def __init__(self, lengthscales, kernel_variance=1.0, noise=1e-8):
        super().__init__(lengthscales=lengthscales, kernel_variance=kernel_variance, noise=noise)

        self.m = 3

        self.b_logits = jnp.zeros((self.m + 1,))

    def covariance(self, xa, xb, include_noise=False):
        """
        Compute spherical polynomial covariance matrix.

        xa : (n1, D)
        xb : (n2, D)
        """
        b = self._softmax_simplex(self.b_logits)   # (m+1,)
        s = self._sphere_dot(xa, xb)

        out = jnp.zeros_like(s)
        s_pow = jnp.ones_like(s)
        for i in range(self.m + 1):
            out = out + b[i] * s_pow
            s_pow = s_pow * s

        K = self.kernel_variance * out

        if include_noise and xa.shape[0] == xb.shape[0]:
            K += self.noise * jnp.eye(K.shape[0])

        return K

    def diagonal(self, x, include_noise=True):
        """
        Since <P(x),P(x)>=1 and sum b_i = 1, k(x,x)=kernel_variance.
        """
        diag = self.kernel_variance * jnp.ones(x.shape[0])
        if include_noise:
            diag += self.noise
        return diag

    def update_hyperparams(self, lengthscales=None, kernel_variance=None, noise=None, a=None, b_logits=None):
        super().update_hyperparams(lengthscales=lengthscales, kernel_variance=kernel_variance, noise=noise, a=a)
        if b_logits is not None:
            self.b_logits = jnp.array(b_logits)