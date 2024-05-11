/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2014, 2022 University College London
    Copyright (C) 2022 Positrigo
    Copyright (C) 2022 Katholieke Universiteit London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: projectors

  \author Kris Thielemans
  \author Markus Jehl
  \author Georg Schramm
*/

%rename (get_forward_projector) *::get_forward_projector_sptr;
%rename (get_back_projector) *::get_back_projector_sptr;
%rename (get_proj_matrix) *::get_proj_matrix;
%rename (get_attenuation_image) *::get_attenuation_image_sptr;

%shared_ptr(stir::ForwardProjectorByBin);
%shared_ptr(stir::RegisteredParsingObject<stir::ForwardProjectorByBinUsingProjMatrixByBin,
    stir::ForwardProjectorByBin>);
%shared_ptr(stir::ForwardProjectorByBinUsingProjMatrixByBin);

%shared_ptr(stir::BackProjectorByBin);
%shared_ptr(stir::RegisteredParsingObject<stir::BackProjectorByBinUsingProjMatrixByBin,
    stir::BackProjectorByBin>);
%shared_ptr(stir::BackProjectorByBinUsingProjMatrixByBin);

%shared_ptr(stir::ProjMatrixByBin);
%shared_ptr(stir::RegisteredParsingObject<stir::ProjMatrixByBinUsingRayTracing,
    stir::ProjMatrixByBin, stir::ProjMatrixByBin>);
%shared_ptr(stir::ProjMatrixByBinUsingRayTracing);

%shared_ptr(stir::RegisteredParsingObject<
	      stir::ProjMatrixByBinSPECTUB,
              stir::ProjMatrixByBin,
              stir::ProjMatrixByBin
	    >);
%shared_ptr(stir::ProjMatrixByBinSPECTUB);

%shared_ptr(stir::RegisteredParsingObject<
	      stir::ProjMatrixByBinPinholeSPECTUB,
              stir::ProjMatrixByBin,
              stir::ProjMatrixByBin
	    >);
%shared_ptr(stir::ProjMatrixByBinPinholeSPECTUB);

%include "stir/recon_buildblock/ForwardProjectorByBin.h"
%include "stir/recon_buildblock/BackProjectorByBin.h"
%include "stir/recon_buildblock/ProjMatrixByBin.h"

%template (internalRPProjMatrixByBinUsingRayTracing) stir::RegisteredParsingObject<
	  stir::ProjMatrixByBinUsingRayTracing, stir::ProjMatrixByBin, stir::ProjMatrixByBin>;

%include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"

%template (internalRPProjMatrixByBinSPECTUB) stir::RegisteredParsingObject<
	      stir::ProjMatrixByBinSPECTUB,
              stir::ProjMatrixByBin,
              stir::ProjMatrixByBin
  >;

%include "stir/recon_buildblock/ProjMatrixByBinPinholeSPECTUB.h"

%template (internalRPProjMatrixByBinPinholeSPECTUB) stir::RegisteredParsingObject<
	      stir::ProjMatrixByBinPinholeSPECTUB,
              stir::ProjMatrixByBin,
              stir::ProjMatrixByBin
  >;

%include "stir/recon_buildblock/ProjMatrixByBinPinholeSPECTUB.h"

%template (internalRPForwardProjectorByBinUsingProjMatrixByBin)  
  stir::RegisteredParsingObject<stir::ForwardProjectorByBinUsingProjMatrixByBin,
     stir::ForwardProjectorByBin>;
%include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"

%template (internalRPBackProjectorByBinUsingProjMatrixByBin)  
  stir::RegisteredParsingObject<stir::BackProjectorByBinUsingProjMatrixByBin,
     stir::BackProjectorByBin>;
%include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"

%shared_ptr(stir::ProjectorByBinPair);
// explicitly ignore constructor, because SWIG tries to instantiate the abstract class otherwise
%ignore stir::ProjectorByBinPair::ProjectorByBinPair();
%shared_ptr(stir::RegisteredParsingObject<stir::ProjectorByBinPairUsingProjMatrixByBin,
    stir::ProjectorByBinPair, stir::ProjectorByBinPair>);
%shared_ptr(stir::ProjectorByBinPairUsingProjMatrixByBin)

%include "stir/recon_buildblock/ProjectorByBinPair.h"

%template(internalRPProjectorByBinPairUsingProjMatrixByBin) stir::RegisteredParsingObject<
    stir::ProjectorByBinPairUsingProjMatrixByBin, stir::ProjectorByBinPair, stir::ProjectorByBinPair>;
%include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"

%shared_ptr(stir::RegisteredParsingObject<stir::ProjectorByBinPairUsingSeparateProjectors,
    stir::ProjectorByBinPair, stir::ProjectorByBinPair>);
%shared_ptr(stir::ProjectorByBinPairUsingSeparateProjectors)

%template(internalRPProjectorByBinPairUsingSeparateProjectors) stir::RegisteredParsingObject<
    stir::ProjectorByBinPairUsingSeparateProjectors, stir::ProjectorByBinPair, stir::ProjectorByBinPair>;
%include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"

#ifdef HAVE_parallelproj
%shared_ptr(stir::RegisteredParsingObject<stir::ForwardProjectorByBinParallelproj,
    stir::ForwardProjectorByBin>);
%shared_ptr(stir::ForwardProjectorByBinParallelproj);
%shared_ptr(stir::RegisteredParsingObject<stir::BackProjectorByBinParallelproj,
    stir::BackProjectorByBin>);
%shared_ptr(stir::BackProjectorByBinParallelproj);

%template (internalRPForwardProjectorByBinParallelproj)  
  stir::RegisteredParsingObject<stir::ForwardProjectorByBinParallelproj,
     stir::ForwardProjectorByBin>;
%include "stir/recon_buildblock/Parallelproj_projector/ForwardProjectorByBinParallelproj.h"

%template (internalRPBackProjectorByBinParallelproj)  
  stir::RegisteredParsingObject<stir::BackProjectorByBinParallelproj,
     stir::BackProjectorByBin>;
%include "stir/recon_buildblock/Parallelproj_projector/BackProjectorByBinParallelproj.h"

%shared_ptr(stir::RegisteredParsingObject<stir::ProjectorByBinPairUsingParallelproj,
    stir::ProjectorByBinPair, stir::ProjectorByBinPair>);
%shared_ptr(stir::ProjectorByBinPairUsingParallelproj);
%template(internalRPProjectorByBinPairUsingParallelproj)
  stir::RegisteredParsingObject<stir::ProjectorByBinPairUsingParallelproj,
    stir::ProjectorByBinPair, stir::ProjectorByBinPair>;
%include "stir/recon_buildblock/Parallelproj_projector/ProjectorByBinPairUsingParallelproj.h"
#endif
