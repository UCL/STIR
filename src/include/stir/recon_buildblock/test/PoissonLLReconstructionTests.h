/*
    Copyright (C) 2020, University College London
    This file is part of STIR.
    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.
    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_test
  \brief Test class for iterative reconstructions using stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/test/ReconstructionTests.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"

START_NAMESPACE_STIR

/*!
  \ingroup recon_test
  \brief Base class for tests for iterative reconstruction that use
   PoissonLogLikelihoodWithLinearModelForMeanAndProjData
*/
template <class TargetT>
class PoissonLLReconstructionTests : public ReconstructionTests<TargetT> {
private:
  typedef ReconstructionTests<TargetT> base_type;

public:
  //! Constructor that can take some input data to run the test with
  explicit inline PoissonLLReconstructionTests(const std::string& proj_data_filename = "",
                                               const std::string& density_filename = "")
      : base_type(proj_data_filename, density_filename) {}

  //! creates Poisson log likelihood
  /*! sets \c _proj_data_sptr and uses \c _input_density_sptr for set_up.
   */
  virtual inline void construct_log_likelihood();

protected:
  shared_ptr<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>> _objective_function_sptr;
};

template <class TargetT>
void
PoissonLLReconstructionTests<TargetT>::construct_log_likelihood() {
  this->_objective_function_sptr.reset(new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>);
  PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>& objective_function =
      reinterpret_cast<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>&>(*this->_objective_function_sptr);
  objective_function.set_proj_data_sptr(this->_proj_data_sptr);
  shared_ptr<ProjMatrixByBin> proj_matrix_sptr(new ProjMatrixByBinUsingRayTracing());
  shared_ptr<ProjectorByBinPair> proj_pair_sptr(new ProjectorByBinPairUsingProjMatrixByBin(proj_matrix_sptr));
  objective_function.set_projector_pair_sptr(proj_pair_sptr);
  /*objective_function.set_normalisation_sptr(bin_norm_sptr);
  objective_function.set_additive_proj_data_sptr(add_proj_data_sptr);
  */
}

END_NAMESPACE_STIR
