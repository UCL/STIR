/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2013, 2014, 2022 University College London
    Copyright (C) 2022 Positrigo
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::Reconstruction hierarchy

  \author Kris Thielemans
  \author Markus Jehl
*/

#define TargetT stir::DiscretisedDensity<3,float>
#define elemT float

%shared_ptr(stir::Reconstruction<TargetT >);
%shared_ptr(stir::IterativeReconstruction<TargetT >);

%shared_ptr(stir::RegisteredParsingObject<
	      stir::OSMAPOSLReconstruction <TargetT > ,
	      stir::Reconstruction < TargetT >,
	      stir::IterativeReconstruction < TargetT >
            >)
%shared_ptr(stir::RegisteredParsingObject<
	      stir::OSSPSReconstruction <TargetT > ,
	      stir::Reconstruction < TargetT >,
	      stir::IterativeReconstruction < TargetT >
            >)

%shared_ptr(stir::OSMAPOSLReconstruction<TargetT >);
%shared_ptr(stir::OSSPSReconstruction<TargetT >);

%shared_ptr(stir::AnalyticReconstruction);

%shared_ptr(stir::RegisteredParsingObject<
        stir::FBP2DReconstruction,
        stir::Reconstruction < TargetT >,
        stir::AnalyticReconstruction
            >);
%shared_ptr(stir::FBP2DReconstruction);

%shared_ptr(stir::RegisteredParsingObject<
        stir::FBP3DRPReconstruction,
        stir::Reconstruction < TargetT > ,
        stir::AnalyticReconstruction
            >);
%shared_ptr(stir::FBP3DRPReconstruction);

#undef elemT
#undef TargetT


%include "stir/recon_buildblock/Reconstruction.h"
 // there's a get_objective_function, so we'll ignore the sptr version
%ignore *::get_objective_function_sptr;
%include "stir/recon_buildblock/IterativeReconstruction.h"
%include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
%include "stir/OSSPS/OSSPSReconstruction.h"

%include "stir/recon_buildblock/AnalyticReconstruction.h"
%include "stir/analytic/FBP2D/FBP2DReconstruction.h"
%include "stir/analytic/FBP3DRP/FBP3DRPReconstruction.h"


#define TargetT stir::DiscretisedDensity<3,float>
#define elemT float

%template (Reconstruction3DFloat) stir::Reconstruction<TargetT >;
//%template () stir::Reconstruction<TargetT >;
%template (IterativeReconstruction3DFloat) stir::IterativeReconstruction<TargetT >;
//%template () stir::IterativeReconstruction<TargetT >;

%template (RPOSMAPOSLReconstruction3DFloat) stir::RegisteredParsingObject<
	      stir::OSMAPOSLReconstruction <TargetT > ,
	      stir::Reconstruction < TargetT >,
	      stir::IterativeReconstruction < TargetT >
              >;
%template (RPOSSPSReconstruction) stir::RegisteredParsingObject<
	      stir::OSSPSReconstruction <TargetT > ,
	      stir::Reconstruction < TargetT >,
	      stir::IterativeReconstruction < TargetT >
            >;

%template (OSMAPOSLReconstruction3DFloat) stir::OSMAPOSLReconstruction<TargetT >;
%template (OSSPSReconstruction3DFloat) stir::OSSPSReconstruction<TargetT >;

// Unfortunately, the below two templates currently break the SWIG interface
// %template (RPFBP2DReconstruction3DFloat) stir::RegisteredParsingObject<
//         stir::FBP2DReconstruction,
//         stir::Reconstruction < TargetT >,
//         stir::AnalyticReconstruction
//             >;

// %template (RPFBP3DReconstruction3DFloat) stir::RegisteredParsingObject<
//         stir::FBP3DRPReconstruction,
//         stir::Reconstruction < TargetT > ,
//         stir::AnalyticReconstruction
//             >;

#undef elemT
#undef TargetT
