#
# SPDX-License-Identifier: Apache-2.0
#
# Class implementing the LBFGSB-PC algorithm in stir
#
# Authors:  Kris Thielemans
#
# Based on Georg Schramm's
# https://github.com/SyneRBI/PETRIC-MaGeZ/blob/a690205b2e3ec874e621ed2a32a802cd0bed4c1d/simulation_src/sim_stochastic_grad.py
# but with using diag(H.1) as preconditioner at the moment, as per Tsai's paper (see ref in the class doc)
#
# Copyright 2025 University College London

import numpy as np
import numpy.typing as npt
import stir
from scipy.optimize import fmin_l_bfgs_b
from typing import Callable, Optional, List

# import matplotlib.pyplot as plt


class LBFGSBPC:
    """Implementation of the LBFGSB-PC Algorithm

    See
    Tsai et al,
    Fast Quasi-Newton Algorithms for Penalized Reconstruction in Emission Tomography and Further Improvements via Preconditioning
    IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. 37, NO. 4, APRIL 2018
    DOI: 10.1109/TMI.2017.2786865

    WARNING: it maximises the objective function (as required by sirf.STIR.ObjectiveFunction).
    WARNING: the implementation uses asarray(), which means you need SIRF 3.9. You should be able to just replace it with as_array() otherwise.

    This implementation is NOT a CIL.Algorithm, but it behaves somewhat as one.
    """

    def __init__(
        self,
        objfun: stir.GeneralisedObjectiveFunction3DFloat,
        initial: stir.FloatVoxelsOnCartesianGrid,
        update_objective_interval: int = 0,
    ):
        self.trunc_filter = stir.TruncateToCylindricalFOVImageProcessor3DFloat()
        self.objfun = objfun
        self.initial = initial.clone()
        self.trunc_filter.apply(self.initial)
        self.shape = initial.shape()
        self.output = None
        self.update_objective_interval = update_objective_interval

        precon = initial.get_empty_copy()
        objfun.accumulate_Hessian_times_input(precon, initial, initial * 0 + 1)
        precon *= -1
        # self.Dinv_STIR = precon.maximum(1).power(-0.5)
        self.Dinv = np.power(np.maximum(precon.as_array(), 1), -0.5)
        self.Dinv_STIR = precon
        self.Dinv_STIR.fill(self.Dinv)
        self.trunc_filter.apply(self.Dinv_STIR)
        # plt.figure()
        # plt.imshow(self.Dinv_STIR.as_array()[self.shape[0] // 2, :, :])
        self.Dinv = self.Dinv_STIR.as_array().ravel()
        self.tmp_for_value = initial.get_empty_copy()
        self.tmp1_for_gradient = initial.get_empty_copy()
        self.tmp2_for_gradient = initial.get_empty_copy()

    def precond_objfun_value(self, z: npt.ArrayLike) -> float:
        self.tmp_for_value.fill(
            np.reshape(z.astype(np.float32) * self.Dinv, self.shape)
        )
        return -self.objfun.compute_value(self.tmp_for_value)

    def precond_objfun_gradient(self, z: npt.ArrayLike) -> np.ndarray:
        self.tmp1_for_gradient.fill(
            np.reshape(z.astype(np.float32) * self.Dinv, self.shape)
        )
        self.objfun.compute_gradient(self.tmp2_for_gradient, self.tmp1_for_gradient)
        return self.tmp2_for_gradient.as_array().ravel() * self.Dinv * -1

    def callback(self, x):
        if (
            self.update_objective_interval > 0
            and self.iter % self.update_objective_interval == 0
        ):
            self.loss.append(-self.precond_objfun_value(x))
            self.iterations.append(self.iter)
        self.iter += 1

    def process(
        self, iterations=None, callbacks: Optional[List[Callable]] = None, verbose=0
    ) -> None:
        r"""run upto :code:`iterations` with callbacks.

        Parameters
        -----------
        iterations: int, default is None
            Number of iterations to run.
        callbacks: list of callables, default is Defaults to self.callback
            List of callables which are passed the current Algorithm object each iteration. Defaults to :code:`[ProgressCallback(verbose)]`.
        verbose: 0=quiet, 1=info, 2=debug
            Passed to the default callback to determine the verbosity of the printed output.
        """
        if iterations is None:
            raise ValueError("`missing argument `iterations`")
        precond_init = self.initial / self.Dinv_STIR
        self.trunc_filter.apply(precond_init)
        precond_init = precond_init.as_array().ravel()
        bounds = precond_init.size * [(0, None)]
        self.iter = 0
        self.loss = []
        self.iterations = []
        # TODO not really required, but it differs from the first value reported by fmin_l_bfgs_b. Not sure why...
        self.callback(precond_init)
        self.iter = 0  # set back again
        res = fmin_l_bfgs_b(
            self.precond_objfun_value,
            precond_init,
            self.precond_objfun_gradient,
            maxiter=iterations,
            bounds=bounds,
            m=20,
            callback=self.callback,
            factr=0,
            pgtol=0,
        )
        # store result (use name "x" for CIL compatibility)
        self.x = self.tmp_for_value.get_empty_copy()
        self.x.fill(np.reshape(res[0].astype(np.float32) * self.Dinv, self.shape))

    def run(
        self, **kwargs
    ) -> None:  # CIL alias, would need to callback and verbose keywords etc
        self.process(**kwargs)

    def get_output(self) -> stir.FloatVoxelsOnCartesianGrid:
        return self.x
