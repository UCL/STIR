/*
    Copyright (C) 2020-2021, University College London
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
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
#include "stir/KeyParser.h"

START_NAMESPACE_STIR

/*!
  \ingroup recon_test
  \brief Base class for tests for iterative reconstruction that use
   PoissonLogLikelihoodWithLinearModelForMeanAndProjData
*/
template <class TargetT>
class PoissonLLReconstructionTests : public ReconstructionTests<TargetT>
{
private:
  typedef ReconstructionTests<TargetT> base_type;
public:
  //! Constructor that can take some input data to run the test with
  explicit inline
    PoissonLLReconstructionTests(const std::string& projector_pair_filename = "",
                                 const std::string &proj_data_filename = "",
                                 const std::string & density_filename = "")
    : base_type(proj_data_filename, density_filename)
    {
      this->construct_projector_pair(projector_pair_filename);
    }

  //! parses projector-pair file to initialise the projector pair
  /*! defaults to using the ray-tracing matrix */
  void construct_projector_pair(const std::string& filename = "");

  //! creates Poisson log likelihood
  /*! sets \c _proj_data_sptr and uses \c _input_density_sptr for set_up.
  */
  virtual inline void construct_log_likelihood();

protected:
  shared_ptr<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT> > _objective_function_sptr;
  shared_ptr<ProjectorByBinPair> _projector_pair_sptr;
};

template <class TargetT>
void
PoissonLLReconstructionTests<TargetT>::
construct_projector_pair(const std::string& filename)
{
  if (filename.empty())
    {
      shared_ptr<ProjMatrixByBin> proj_matrix_sptr(new ProjMatrixByBinUsingRayTracing());
      this->_projector_pair_sptr.reset(new ProjectorByBinPairUsingProjMatrixByBin(proj_matrix_sptr));
      return;
    }
      
  KeyParser parser;
  parser.add_start_key("projector pair parameters");
  parser.add_parsing_key("projector pair type", &this->_projector_pair_sptr);
  parser.add_stop_key("end projector pair parameters");
  parser.parse(filename.c_str());
  if (!this->_projector_pair_sptr)
    error("Error parsing projector pair file");
}
template <class TargetT>
void
PoissonLLReconstructionTests<TargetT>::
construct_log_likelihood()
{ 
  this->_objective_function_sptr.reset(new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>);
  PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>& objective_function =
    reinterpret_cast<  PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>& >(*this->_objective_function_sptr);
  objective_function.set_proj_data_sptr(this->_proj_data_sptr);
  if (!this->_projector_pair_sptr)
    error("Internal error: need to set the projector pair first");
  objective_function.set_projector_pair_sptr(this->_projector_pair_sptr);
  /*objective_function.set_normalisation_sptr(bin_norm_sptr);
  objective_function.set_additive_proj_data_sptr(add_proj_data_sptr);
  */
}

END_NAMESPACE_STIR
