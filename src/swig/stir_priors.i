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

%shared_ptr(stir::GibbsPenalty<elemT, stir::QuadraticPotential<elemT>>);
%shared_ptr(stir::RegisteredParsingObject< stir::GibbsQuadraticPenalty<elemT>,
  stir::GeneralisedPrior<TargetT >,
  stir::GibbsPenalty<elemT, stir::QuadraticPotential<elemT>> >);
%shared_ptr(stir::GibbsQuadraticPenalty<elemT>);

%shared_ptr(stir::RelativeDifferencePotential<elemT>);

%shared_ptr(stir::GibbsPenalty<elemT, stir::RelativeDifferencePotential<elemT>>);
%shared_ptr(stir::RegisteredParsingObject< stir::GibbsRelativeDifferencePenalty<elemT>,
  stir::GeneralisedPrior<TargetT >,
  stir::GibbsPenalty<elemT, stir::RelativeDifferencePotential<elemT>> >);
%shared_ptr(stir::GibbsRelativeDifferencePenalty<elemT>);

#ifdef STIR_WITH_CUDA
  %shared_ptr(stir::RegisteredParsingObject< stir::CudaRelativeDifferencePrior<elemT>,
          stir::GeneralisedPrior<TargetT >,
          stir::RelativeDifferencePrior<elemT> >);
  %shared_ptr(stir::CudaRelativeDifferencePrior<elemT>);

  %shared_ptr(stir::CudaGibbsPenalty<elemT, stir::QuadraticPotential<elemT>>);
  %shared_ptr(stir::RegisteredParsingObject< stir::CudaGibbsQuadraticPenalty<elemT>,
    stir::GeneralisedPrior<TargetT >,
    stir::CudaGibbsPenalty<elemT, stir::QuadraticPotential<elemT>> >);
  %shared_ptr(stir::CudaGibbsQuadraticPenalty<elemT>);

  %shared_ptr(stir::CudaGibbsPenalty<elemT, stir::RelativeDifferencePotential<elemT>>);
  %shared_ptr(stir::RegisteredParsingObject< stir::CudaGibbsRelativeDifferencePenalty<elemT>,
    stir::GeneralisedPrior<TargetT >,
    stir::CudaGibbsPenalty<elemT, stir::RelativeDifferencePotential<elemT>> >);
  %shared_ptr(stir::CudaGibbsRelativeDifferencePenalty<elemT>);
#endif




#undef elemT
#undef TargetT
%include "stir/recon_buildblock/GeneralisedPrior.h"
%include "stir/recon_buildblock/GibbsPenalty.h"

#ifdef STIR_WITH_CUDA
  %include "stir/recon_buildblock/CUDA/CudaGibbsPenalty.h"
  %include "stir/recon_buildblock/CUDA/CudaRelativeDifferencePrior.h"
#endif

%include "stir/recon_buildblock/GibbsQuadraticPenalty.h"
%include "stir/recon_buildblock/GibbsRelativeDifferencePenalty.h"

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

%nodefaultctor stir::GibbsPenalty;
%ignore stir::GibbsPenalty::GibbsPenalty;
%template (GibbsPenalty3DFloat_Q) stir::GibbsPenalty<elemT, stir::QuadraticPotential<elemT>>;

%template (RPGibbsQuadraticPenalty3DFloat)
stir::RegisteredParsingObject< stir::GibbsQuadraticPenalty<elemT>,
stir::GeneralisedPrior<TargetT >,
stir::GibbsPenalty<elemT, stir::QuadraticPotential<elemT>> >;
%template (GibbsQuadraticPenalty3DFloat) stir::GibbsQuadraticPenalty<elemT>;

%template (RelativeDifference3DFloat) stir::RelativeDifferencePotential<elemT>;

%nodefaultctor stir::GibbsPenalty;
%ignore stir::GibbsPenalty::GibbsPenalty;
%template (GibbsPenalty3DFloat_RD) stir::GibbsPenalty<elemT, stir::RelativeDifferencePotential<elemT>>;

%template (RPGibbsRelativeDifferencePenalty3DFloat)
stir::RegisteredParsingObject< stir::GibbsRelativeDifferencePenalty<elemT>,
stir::GeneralisedPrior<TargetT >,
stir::GibbsPenalty<elemT, stir::RelativeDifferencePotential<elemT>> >;
%template (GibbsRelativeDifferencePenalty3DFloat) stir::GibbsRelativeDifferencePenalty<elemT>;

#ifdef STIR_WITH_CUDA
  %template (RPCudaRelativeDifferencePrior3DFloat)
    stir::RegisteredParsingObject< stir::CudaRelativeDifferencePrior<elemT>,
      stir::GeneralisedPrior<TargetT >,
      stir::RelativeDifferencePrior<elemT> >;
  %template (CudaRelativeDifferencePrior3DFloat) stir::CudaRelativeDifferencePrior<elemT>;

  %nodefaultctor stir::CudaGibbsPenalty;
  %ignore stir::CudaGibbsPenalty::CudaGibbsPenalty;
  %template (CudaGibbsPenalty3DFloat_RD) stir::CudaGibbsPenalty<elemT, stir::RelativeDifferencePotential<elemT>>;

  %template (RPCudaGibbsRelativeDifferencePenalty3DFloat)
  stir::RegisteredParsingObject< stir::CudaGibbsRelativeDifferencePenalty<elemT>,
  stir::GeneralisedPrior<TargetT >,
  stir::CudaGibbsPenalty<elemT, stir::RelativeDifferencePotential<elemT>> >;
  %template (CudaGibbsRelativeDifferencePenalty3DFloat) stir::CudaGibbsRelativeDifferencePenalty<elemT>;

  %template (CudaGibbsPenalty3DFloat_Q) stir::CudaGibbsPenalty<elemT, stir::QuadraticPotential<elemT>>;

  %template (RPCudaGibbsQuadraticPenalty3DFloat)
  stir::RegisteredParsingObject< stir::CudaGibbsQuadraticPenalty<elemT>,
  stir::GeneralisedPrior<TargetT >,
  stir::CudaGibbsPenalty<elemT, stir::QuadraticPotential<elemT>> >;
  %template (CudaGibbsQuadraticPenalty3DFloat) stir::CudaGibbsQuadraticPenalty<elemT>;
#endif



#undef elemT
#undef TargetT
