<p align="center">
<img width="700" height="300" src="https://raw.githubusercontent.com/JaxGaussianProcesses/JaxKern/main/docs/_static/logo/logo.png" alt="JaxKern's logo">
</p>
<h2 align='center'>Kernels in Jax.</h2>

## NOTE
This is a fork of JaxKern, losely maintained to be compatible with up-to-date versions of Jax. The original JaxKern is not maintained anymore and has now been incorporated into [GPJax](https://github.com/JaxGaussianProcesses/GPJax).

Changes made: 
- Replaced use of `jax.random.KeyArray` with `jax.Array` as per [jax 0.4.16 (Sept 18, 2023)](https://jax.readthedocs.io/en/latest/changelog.html#jax-0-4-16-sept-18-2023). 
- removed dependency on [JaxUtils](https://github.com/JaxGaussianProcesses/JaxUtils) by incorporating JaxUtils PyTree implementation into this fork.

Todo:
- [ ] Remove package dependency on JaxUtils  

## Introduction

JaxKern is Python library for working with kernel functions in JAX. We currently support the following kernels:
* Stationary
    * Radial basis function (Squared exponential)
    * Matérn
    * Powered exponential
    * Rational quadratic
    * White noise
    * Periodic
* Non-stationary
    * Linear 
    * Polynomial
* Non-Euclidean
    * Graph kernels

In addition to this, we implement kernel approximations using the [Random Fourier feature](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) approach.

## Example

The following code snippet demonstrates how the first order Matérn kernel can be computed and, subsequently, approximated using random Fourier features.
```python
import jaxkern as jk
import jax.numpy as jnp
import jax.random as jr
key = jr.PRNGKey(123)

# Define the points on which we'll evaluate the kernel
X = jr.uniform(key, shape = (10, 1), minval=-3., maxval=3.)
Y = jr.uniform(key, shape = (20, 1), minval=-3., maxval=3.)

# Instantiate the kernel and its parameters
kernel = jk.Matern32()
params = kernel.init_params(key)

# Compute the 10x10 Gram matrix
Kxx = kernel.gram(params, X)

# Compute the 10x20 cross-covariance matrix
Kxy = kernel.cross_covariance(params, X, Y)

# Build a RFF approximation
approx = RFF(kernel, num_basis_fns = 5)
rff_params = approx.init_params(key)

# Build an approximation to the Gram matrix
Qff = approx.gram(rff_params, X)
```

## Code Structure

All kernels are supplied with a `gram` and `cross_covariance` method. When computing a Gram matrix, there is often some structure in the data (e.g., Markov) that can be exploited to yield a sparse matrix. To instruct JAX how to operate on this, the return type of `gram` is a Linear Operator from [JaxLinOp](https://github.com/JaxGaussianProcesses/JaxLinOp). 

Within [GPJax](https://github.com/JaxGaussianProcesses/GPJax), all kernel computations are handled using JaxKern.

