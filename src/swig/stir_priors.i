/*
    Copyright (C) 2018, 2020, 2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG

  \author Kris Thielemans
  \author Robert Twyman
*/



#define TargetT stir::DiscretisedDensity<3,float>
#define elemT float

%shared_ptr(stir::GeneralisedPrior<TargetT >);

%shared_ptr(stir::PriorWithParabolicSurrogate<TargetT >);

%shared_ptr(stir::RegisteredParsingObject< stir::QuadraticPrior<elemT>,
            stir::GeneralisedPrior<TargetT >,
            stir::PriorWithParabolicSurrogate<TargetT  > >);

%shared_ptr(stir::QuadraticPrior<elemT>);

%shared_ptr(stir::RegisteredParsingObject< stir::PLSPrior<elemT>,
            stir::GeneralisedPrior<TargetT >,
            stir::GeneralisedPrior<TargetT > >);
%shared_ptr(stir::PLSPrior<elemT>);

%shared_ptr(stir::RegisteredParsingObject< stir::RelativeDifferencePrior<elemT>,
         stir::GeneralisedPrior<TargetT >,
         stir::GeneralisedPrior<TargetT > >);
%shared_ptr(stir::RelativeDifferencePrior<elemT>);

%shared_ptr(stir::RegisteredParsingObject< stir::LogcoshPrior<elemT>,
        stir::GeneralisedPrior<TargetT >,
        stir::PriorWithParabolicSurrogate<TargetT  > >);
%shared_ptr(stir::LogcoshPrior<elemT>);

%shared_ptr(stir::QuadraticPotential<elemT>);

%shared_ptr(stir::GibbsPrior<elemT, stir::QuadraticPotential<elemT>>);
%shared_ptr(stir::RegisteredParsingObject< stir::GibbsQuadraticPrior<elemT>,
  stir::GeneralisedPrior<TargetT >,
  stir::GibbsPrior<elemT, stir::QuadraticPotential<elemT>> >);
%shared_ptr(stir::GibbsQuadraticPrior<elemT>);

%shared_ptr(stir::RelativeDifferencePotential<elemT>);

%shared_ptr(stir::GibbsPrior<elemT, stir::RelativeDifferencePotential<elemT>>);
%shared_ptr(stir::RegisteredParsingObject< stir::GibbsRelativeDifferencePrior<elemT>,
  stir::GeneralisedPrior<TargetT >,
  stir::GibbsPrior<elemT, stir::RelativeDifferencePotential<elemT>> >);
%shared_ptr(stir::GibbsRelativeDifferencePrior<elemT>);

#ifdef __CUDACC__
  %shared_ptr(stir::RegisteredParsingObject< stir::CudaRelativeDifferencePrior<elemT>,
          stir::GeneralisedPrior<TargetT >,
          stir::RelativeDifferencePrior<elemT> >);
  %shared_ptr(stir::CudaRelativeDifferencePrior<elemT>);

  %shared_ptr(stir::CudaGibbsPrior<elemT, stir::QuadraticPotential<elemT>>);
  %shared_ptr(stir::RegisteredParsingObject< stir::CudaGibbsQuadraticPrior<elemT>,
    stir::GeneralisedPrior<TargetT >,
    stir::CudaGibbsPrior<elemT, stir::QuadraticPotential<elemT>> >);
  %shared_ptr(stir::CudaGibbsQuadraticPrior<elemT>);

  %shared_ptr(stir::CudaGibbsPrior<elemT, stir::RelativeDifferencePotential<elemT>>);
  %shared_ptr(stir::RegisteredParsingObject< stir::CudaGibbsRelativeDifferencePrior<elemT>,
    stir::GeneralisedPrior<TargetT >,
    stir::CudaGibbsPrior<elemT, stir::RelativeDifferencePotential<elemT>> >);
  %shared_ptr(stir::CudaGibbsRelativeDifferencePrior<elemT>);
#endif




#undef elemT
#undef TargetT
%include "stir/recon_buildblock/GeneralisedPrior.h"
%include "stir/recon_buildblock/GibbsPrior.h"

#ifdef STIR_WITH_CUDA
  %include "stir/recon_buildblock/CUDA/CudaGibbsPrior.h"
  %include "stir/recon_buildblock/CUDA/CudaRelativeDifferencePrior.h"
#endif

%include "stir/recon_buildblock/GibbsQuadraticPrior.h"
%include "stir/recon_buildblock/GibbsRelativeDifferencePrior.h"

%include "stir/recon_buildblock/PriorWithParabolicSurrogate.h"
%include "stir/recon_buildblock/QuadraticPrior.h"
%include "stir/recon_buildblock/PLSPrior.h"
%include "stir/recon_buildblock/RelativeDifferencePrior.h"
%include "stir/recon_buildblock/LogcoshPrior.h"

#define TargetT stir::DiscretisedDensity<3,float>
#define elemT float

%template (GeneralisedPrior3DFloat) stir::GeneralisedPrior<TargetT >;
%template (PriorWithParabolicSurrogate3DFloat) stir::PriorWithParabolicSurrogate<TargetT >;

%template (RPQuadraticPrior3DFloat)
  stir::RegisteredParsingObject< stir::QuadraticPrior<elemT>,
      stir::GeneralisedPrior<TargetT >,
      stir::PriorWithParabolicSurrogate<TargetT  > >;
%template (QuadraticPrior3DFloat) stir::QuadraticPrior<elemT>;

%template (RPPLSPrior3DFloat)
  stir::RegisteredParsingObject< stir::PLSPrior<elemT>,
      stir::GeneralisedPrior<TargetT >,
      stir::GeneralisedPrior<TargetT > >;
%template (PLSPrior3DFloat) stir::PLSPrior<elemT>;

%template (RPRelativeDifferencePrior3DFloat)
    stir::RegisteredParsingObject< stir::RelativeDifferencePrior<elemT>,
       stir::GeneralisedPrior<TargetT >,
       stir::GeneralisedPrior<TargetT > >;
%template (RelativeDifferencePrior3DFloat) stir::RelativeDifferencePrior<elemT>;

%template (RPLogcoshPrior3DFloat)
stir::RegisteredParsingObject< stir::LogcoshPrior<elemT>,
        stir::GeneralisedPrior<TargetT >,
        stir::PriorWithParabolicSurrogate<TargetT  > >;
%template (LogcoshPrior3DFloat) stir::LogcoshPrior<elemT>;

%template (QuadraticPotential3DFloat) stir::QuadraticPotential<elemT>;

%nodefaultctor stir::GibbsPrior;
%ignore stir::GibbsPrior::GibbsPrior;
%template (GibbsPrior3DFloat_Q) stir::GibbsPrior<elemT, stir::QuadraticPotential<elemT>>;

%template (RPGibbsQuadraticPrior3DFloat)
stir::RegisteredParsingObject< stir::GibbsQuadraticPrior<elemT>,
stir::GeneralisedPrior<TargetT >,
stir::GibbsPrior<elemT, stir::QuadraticPotential<elemT>> >;
%template (GibbsQuadraticPrior3DFloat) stir::GibbsQuadraticPrior<elemT>;

%template (RelativeDifference3DFloat) stir::RelativeDifferencePotential<elemT>;

%template (GibbsPrior3DFloat_RD) stir::GibbsPrior<elemT, stir::RelativeDifferencePotential<elemT>>;

%template (RPGibbsRelativeDifferencePrior3DFloat)
stir::RegisteredParsingObject< stir::GibbsRelativeDifferencePrior<elemT>,
stir::GeneralisedPrior<TargetT >,
stir::GibbsPrior<elemT, stir::RelativeDifferencePotential<elemT>> >;
%template (GibbsRelativeDifferencePrior3DFloat) stir::GibbsRelativeDifferencePrior<elemT>;

#ifdef __CUDACC__
  %template (RPCudaRelativeDifferencePrior3DFloat)
    stir::RegisteredParsingObject< stir::CudaRelativeDifferencePrior<elemT>,
      stir::GeneralisedPrior<TargetT >,
      stir::RelativeDifferencePrior<elemT> >;
  %template (CudaRelativeDifferencePrior3DFloat) stir::CudaRelativeDifferencePrior<elemT>;

  %nodefaultctor stir::CudaGibbsPrior;
  %ignore stir::CudaGibbsPrior::CudaGibbsPrior;
  %template (CudaGibbsPrior3DFloat_RD) stir::CudaGibbsPrior<elemT, stir::RelativeDifferencePotential<elemT>>;

  %template (RPCudaGibbsRelativeDifferencePrior3DFloat)
  stir::RegisteredParsingObject< stir::CudaGibbsRelativeDifferencePrior<elemT>,
  stir::GeneralisedPrior<TargetT >,
  stir::CudaGibbsPrior<elemT, stir::RelativeDifferencePotential<elemT>> >;
  %template (CudaGibbsRelativeDifferencePrior3DFloat) stir::CudaGibbsRelativeDifferencePrior<elemT>;

  %nodefaultctor stir::CudaGibbsPrior;
  %ignore stir::CudaGibbsPrior::CudaGibbsPrior;
  %template (CudaGibbsPrior3DFloat_Q) stir::CudaGibbsPrior<elemT, stir::QuadraticPotential<elemT>>;

  %template (RPCudaGibbsQuadraticPrior3DFloat)
  stir::RegisteredParsingObject< stir::CudaGibbsQuadraticPrior<elemT>,
  stir::GeneralisedPrior<TargetT >,
  stir::CudaGibbsPrior<elemT, stir::QuadraticPotential<elemT>> >;
  %template (CudaGibbsQuadraticPrior3DFloat) stir::CudaGibbsQuadraticPrior<elemT>;
#endif



#undef elemT
#undef TargetT
