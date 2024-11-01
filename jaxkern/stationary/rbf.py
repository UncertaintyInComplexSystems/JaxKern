# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
# # from jax.random import KeyArray
from jaxtyping import Array, Float

from ..base import AbstractKernel
from ..computations import (
    DenseKernelComputation,
)
from .utils import squared_distance
import distrax as dx


class RBF(AbstractKernel):
    """The Radial Basis Function (RBF) kernel."""

    def __init__(
        self,
        active_dims: Optional[List[int]] = None,
        name: Optional[str] = "Radial basis function kernel",
    ) -> None:
        super().__init__(
            DenseKernelComputation,
            active_dims,
            spectral_density=dx.Normal(loc=0.0, scale=1.0),
            name=name,
        )
        self._stationary = True

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

    def init_params(self, key: Array) -> Dict:
        params = {
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }
        return jax.tree_util.tree_map(lambda x: jnp.atleast_1d(x), params)
