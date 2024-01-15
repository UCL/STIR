/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2013, 2018, 2020, 2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::GeneralisedObjectiveFunction function hierarchy

  \author Kris Thielemans
  \author Robert Twyman
*/

%ignore *::get_exam_info_uptr_for_target;
%ignore *::get_projector_pair_sptr;
%ignore *::get_proj_data_sptr;
%ignore *::get_normalisation_sptr;
%rename (get_initial_data) *::get_initial_data_ptr;
%rename (construct_target_image) *::construct_target_image_ptr;
%rename (construct_target) *::construct_target_ptr;
%ignore *::get_prior_sptr;
%rename (get_prior) *::get_prior_ptr;
%rename (get_anatomical_prior) *::get_anatomical_prior_sptr;
%rename (get_kappa) *::get_kappa_sptr;

#define TargetT stir::DiscretisedDensity<3,float>
#define elemT float

%shared_ptr(stir::GeneralisedObjectiveFunction<TargetT >);
%shared_ptr(stir::PoissonLogLikelihoodWithLinearModelForMean<TargetT >);
%shared_ptr(stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT >);
%shared_ptr(stir::RegisteredParsingObject<stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >,
	    stir::GeneralisedObjectiveFunction<TargetT >,
	    stir::PoissonLogLikelihoodWithLinearModelForMean<TargetT > >);
%shared_ptr(stir::RegisteredParsingObject<stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT >,
	    stir::GeneralisedObjectiveFunction<TargetT >,
	    stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT > >);

%shared_ptr(stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >);
%shared_ptr(stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT >);

%shared_ptr(stir::SqrtHessianRowSum<TargetT >);

#undef elemT
#undef TargetT


%include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
%include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
%include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
%include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
%include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeData.h"
%include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin.h"
%include "stir/recon_buildblock/SqrtHessianRowSum.h"


#define TargetT stir::DiscretisedDensity<3,float>
#define elemT float

%template (GeneralisedObjectiveFunction3DFloat) stir::GeneralisedObjectiveFunction<TargetT >;
//%template () stir::GeneralisedObjectiveFunction<TargetT >;
%template (PoissonLogLikelihoodWithLinearModelForMean3DFloat) stir::PoissonLogLikelihoodWithLinearModelForMean<TargetT >;
%template (PoissonLogLikelihoodWithLinearModelForMeanAndListModeData3DFloat) stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT >;

// TODO do we really need this name?
// Without it we don't see the parsing functions in python...
// Note: we cannot start it with __ as then we we get a run-time error when we're not using the builtin option
%template(RPPoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat)  stir::RegisteredParsingObject<stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >,
  stir::GeneralisedObjectiveFunction<TargetT >,
  stir::PoissonLogLikelihoodWithLinearModelForMean<TargetT > >;

%template(RPPoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin3DFloat)  stir::RegisteredParsingObject<stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT >,
  stir::GeneralisedObjectiveFunction<TargetT >,
  stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT > >;

%template (PoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat) stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >;
%template (PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin3DFloat) stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT >;

%inline %{
  template <class T>
    stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<T> *
    ToPoissonLogLikelihoodWithLinearModelForMeanAndProjData(stir::GeneralisedObjectiveFunction<T> *b) {
    return dynamic_cast<stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData<T>*>(b);
  }
%}

%template(ToPoissonLogLikelihoodWithLinearModelForMeanAndProjData3DFloat) ToPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT >;

%template (SqrtHessianRowSum3DFloat) stir::SqrtHessianRowSum<TargetT >;

#undef elemT
#undef TargetT
