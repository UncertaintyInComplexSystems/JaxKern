from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from jaxtyping import Array, Float

from .base import AbstractKernel
from .computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from .utils import euclidean_distance, squared_distance


class RBF(AbstractKernel):
    """The Radial Basis Function (RBF) kernel."""

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Radial basis function kernel",
    ) -> None:
        super().__init__(
            compute_engine, active_dims, stationary, spectral, name
        )

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\\ell` and variance :math:`\\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( \\frac{\\lVert x - y \\rVert^2_2}{2 \\ell^2} \\Bigg)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["variance"] * jnp.exp(-0.5 * squared_distance(x, y))
        return K.squeeze()

    def init_params(self, key: KeyArray) -> Dict:
        params = {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }
        return jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x), params)


class Matern12(AbstractKernel):
    """The Matérn kernel with smoothness parameter fixed at 0.5."""

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Matérn 1/2 kernel",
    ) -> None:
        super().__init__(
            compute_engine, active_dims, stationary, spectral, name
        )

    def __call__(
        self,
        params: Dict,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\\ell` and variance :math:`\\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( -\\frac{\\lvert x-y \\rvert}{2\\ell^2}  \\Bigg)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call
        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["variance"] * jnp.exp(-euclidean_distance(x, y))
        return K.squeeze()

    def init_params(self, key: KeyArray) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


class Matern32(AbstractKernel):
    """The Matérn kernel with smoothness parameter fixed at 1.5."""

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Matern 3/2",
    ) -> None:
        super().__init__(
            compute_engine, active_dims, stationary, spectral, name
        )

    def __call__(
        self,
        params: Dict,
        x: Float[Array, "1 D"],
        y: Float[Array, "1 D"],
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\\ell` and variance :math:`\\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg(1+ \\frac{\\sqrt{3}\\lvert x-y \\rvert}{\\ell^2}  \\Bigg)\\exp\\Bigg(-\\frac{\\sqrt{3}\\lvert x-y\\rvert}{\\ell^2} \\Bigg)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        tau = euclidean_distance(x, y)
        K = (
            params["variance"]
            * (1.0 + jnp.sqrt(3.0) * tau)
            * jnp.exp(-jnp.sqrt(3.0) * tau)
        )
        return K.squeeze()

    def init_params(self, key: KeyArray) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


class Matern52(AbstractKernel):
    """The Matérn kernel with smoothness parameter fixed at 2.5."""

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Matern 5/2",
    ) -> None:
        super().__init__(
            compute_engine, active_dims, stationary, spectral, name
        )

    def __call__(
        self, params: Dict, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with
        lengthscale parameter :math:`\\ell` and variance :math:`\\sigma^2`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg(1+ \\frac{\\sqrt{5}\\lvert x-y \\rvert}{\\ell^2} + \\frac{5\\lvert x - y \\rvert^2}{3\\ell^2} \\Bigg)\\exp\\Bigg(-\\frac{\\sqrt{5}\\lvert x-y\\rvert}{\\ell^2} \\Bigg)

        Args:
            params (Dict): Parameter set for which the kernel should be evaluated on.
            x (Float[Array, "1 D"]): The left hand argument of the kernel function's call.
            y (Float[Array, "1 D"]): The right hand argument of the kernel function's call.

        Returns:
            Float[Array, "1"]: The value of :math:`k(x, y)`.
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        tau = euclidean_distance(x, y)
        K = (
            params["variance"]
            * (1.0 + jnp.sqrt(5.0) * tau + 5.0 / 3.0 * jnp.square(tau))
            * jnp.exp(-jnp.sqrt(5.0) * tau)
        )
        return K.squeeze()

    def init_params(self, key: KeyArray) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }


class PoweredExponential(AbstractKernel):
    """The powered exponential family of kernels.

    Key reference is Diggle and Ribeiro (2007) - "Model-based Geostatistics".

    """

    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Powered exponential",
    ) -> None:
        super().__init__(
            compute_engine, active_dims, stationary, spectral, name
        )

    def __call__(self, params: dict, x: jax.Array, y: jax.Array) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\\ell`, :math:`\\sigma` and power :math:`\\kappa`.

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( - \\Big( \\frac{\\lVert x - y \\rVert^2}{\\ell^2} \\Big)^\\kappa \\Bigg)

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.

        Returns:
            Array: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["variance"] * jnp.exp(
            -euclidean_distance(x, y) ** params["power"]
        )
        return K.squeeze()

    def init_params(self, key: KeyArray) -> Dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
            "power": jnp.array([1.0]),
        }


class RationalQuadratic(AbstractKernel):
    def __init__(
        self,
        compute_engine: AbstractKernelComputation = DenseKernelComputation,
        active_dims: Optional[List[int]] = None,
        stationary: Optional[bool] = False,
        spectral: Optional[bool] = False,
        name: Optional[str] = "Rational Quadratic",
    ) -> None:
        super().__init__(
            compute_engine, active_dims, stationary, spectral, name
        )

    def __call__(self, params: dict, x: jax.Array, y: jax.Array) -> Array:
        """Evaluate the kernel on a pair of inputs :math:`(x, y)` with length-scale parameter :math:`\\ell` and variance :math:`\\sigma`

        .. math::
            k(x, y) = \\sigma^2 \\exp \\Bigg( 1 + \\frac{\\lVert x - y \\rVert^2_2}{2 \\alpha \\ell^2} \\Bigg)

        Args:
            x (jax.Array): The left hand argument of the kernel function's call.
            y (jax.Array): The right hand argument of the kernel function's call
            params (dict): Parameter set for which the kernel should be evaluated on.
        Returns:
            Array: The value of :math:`k(x, y)`
        """
        x = self.slice_input(x) / params["lengthscale"]
        y = self.slice_input(y) / params["lengthscale"]
        K = params["variance"] * (
            1 + 0.5 * squared_distance(x, y) / params["alpha"]
        ) ** (-params["alpha"])
        return K.squeeze()

    def init_params(self, key: KeyArray) -> dict:
        return {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
            "alpha": jnp.array([1.0]),
        }


__all__ = [
    "RBF",
    "Matern12",
    "Matern32",
    "Matern52",
    "RationalQuadratic",
    "PoweredExponential",
]
