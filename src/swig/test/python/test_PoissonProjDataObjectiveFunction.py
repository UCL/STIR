# Test file for STIR objective functions
# Use as follows:
# on command line
#     pytest test_PoissonProjDataObjectiveFunction.py


#    Copyright (C) 2025 University College London
#    Copyright (C) 2024 Science Technology Facilities Council
#    This file is part of STIR.
#
#    SPDX-License-Identifier: Apache-2.0
#
#    See STIR/LICENSE.txt for details

import os
import numpy as np
import unittest
import stir

stir.Verbosity.set(0)


# implement missing norm
def norm(stir_obj):
    return np.linalg.norm(stir_obj.as_array())


class TestSTIRObjectiveFunction(unittest.TestCase):

    def setUp(self):
        # location of script
        loc = os.path.dirname(__file__)
        # location of recon_test_pack
        data_path = os.path.join(loc, "..", "..", "..", "..", "recon_test_pack")

        image = stir.FloatVoxelsOnCartesianGrid.read_from_file(
            os.path.join(data_path, "test_image_5.hv")
        )
        templ = stir.ProjDataInMemory.read_from_file(
            os.path.join(data_path, "Utahscat600k_ca_seg4.hs")
        )

        pm = stir.ProjMatrixByBinUsingRayTracing()
        projector_pair = stir.ProjectorByBinPairUsingProjMatrixByBin(pm)
        projector_pair.set_up(templ.get_proj_data_info(), image)
        acquired_data = templ
        # acquired_data = stir.ProjDataInMemory(templ.get_exam_info(), templ.get_proj_data_info())
        # projector_pair.get_forward_projector().forward_project(acquired_data, image)
        obj_fun = stir.PoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat()
        obj_fun.set_projector_pair_sptr(projector_pair)
        obj_fun.set_input_data(acquired_data)
        obj_fun.set_additive_proj_data_sptr(
            acquired_data * 0 + np.mean(acquired_data.as_array())
        )
        prior = stir.QuadraticPrior3DFloat()
        prior.set_penalisation_factor(10)
        obj_fun.set_prior_sptr(prior)
        obj_fun.set_up(image)

        self.obj_fun = obj_fun
        self.prior = prior
        self.image = image

    # def tearDown(self):

    def test_value_with_prior(self):
        """check if value is loglik - prior"""
        beta = self.prior.get_penalisation_factor()
        x = self.image
        obj_v = self.obj_fun.compute_value(x)
        prior_v = self.prior.compute_value(x)
        self.prior.set_penalisation_factor(0)
        loglik_v = self.obj_fun.compute_value(x)
        assert np.allclose(obj_v, loglik_v - prior_v)
        self.prior.set_penalisation_factor(beta)

    def test_gradient_with_prior(self):
        """check if gradient is the same as loglik - prior"""
        beta = self.prior.get_penalisation_factor()
        x = self.image
        obj_g = x.get_empty_copy()
        prior_g = x.get_empty_copy()
        loglik_g = x.get_empty_copy()
        self.obj_fun.compute_gradient(obj_g, x)
        self.prior.compute_gradient(prior_g, x)
        self.prior.set_penalisation_factor(0)
        self.obj_fun.compute_gradient(loglik_g, x)
        assert norm(obj_g - (loglik_g - prior_g)) <= norm(obj_g) * 1e-3
        self.prior.set_penalisation_factor(beta)

    def test_Hessian_with_prior(self, eps=1e-2):
        """check if Hessian is the same as loglik - prior"""
        beta = self.prior.get_penalisation_factor()
        x = self.image
        dx = x.clone()
        dx *= eps
        dx += eps / 2
        obj_Hdx = x.get_empty_copy()
        prior_Hdx = x.get_empty_copy()
        loglik_Hdx = x.get_empty_copy()
        self.obj_fun.accumulate_sub_Hessian_times_input(obj_Hdx, x, dx, 0)
        self.prior.accumulate_Hessian_times_input(prior_Hdx, x, dx)
        self.prior.set_penalisation_factor(0)
        self.obj_fun.accumulate_sub_Hessian_times_input(loglik_Hdx, x, dx, 0)
        assert norm(obj_Hdx - (loglik_Hdx - prior_Hdx)) <= norm(obj_Hdx) * 1e-3
        self.prior.set_penalisation_factor(beta)

    def test_Hessian(self, subset=0, eps=5e-4):
        """Checks that grad(x + dx) - grad(x) is close to H(x)*dx"""
        x = self.image
        dx = x.clone()
        dx *= eps
        dx += eps / 2
        y = x + dx
        gx = x.get_empty_copy()
        gy = x.get_empty_copy()
        Hdx = x.get_empty_copy()
        self.obj_fun.compute_sub_gradient(gx, x, subset)
        self.obj_fun.compute_sub_gradient(gy, y, subset)
        dg = gy - gx

        self.obj_fun.accumulate_Hessian_times_input(Hdx, x, dx)
        q = norm(dg - Hdx) / norm(dg)
        print("norm of grad(x): %f" % norm(gx))
        print("norm of grad(x + dx): %f" % norm(gy))
        print("norm of grad(x + dx) - grad(x): %f" % norm(dg))
        print("norm of H(x)*dx: %f" % norm(Hdx))
        print("relative difference: %f" % q)
        assert q <= 0.01
