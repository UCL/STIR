/*
    Copyright (C) 2002-2011, Hammersmith Imanet Ltd
    Copyright (C) 2012, Kris Thielemans
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2013, University College London
    
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
  \ingroup IO

  \brief File that registers all stir::RegisterObject children in IO

  \author Kris Thielemans
  \author Berta Marti Fuster
*/

#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/IO/InterfileOutputFileFormat.h"
#include "stir/IO/ITKOutputFileFormat.h"
#include "stir/IO/InterfileDynamicDiscretisedDensityOutputFileFormat.h"
#include "stir/IO/InterfileParametricDensityOutputFileFormat.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/ECAT6OutputFileFormat.h"
#include "stir/IO/ECAT7OutputFileFormat.h"
#include "stir/IO/ECAT7DynamicDiscretisedDensityOutputFileFormat.h"
#include "stir/IO/ECAT7ParametricDensityOutputFileFormat.h"
#include "stir/IO/ECAT7DynamicDiscretisedDensityInputFileFormat.h"
#endif


#if 1
#include "stir/IO/InputFileFormatRegistry.h"
#include "stir/IO/InterfileImageInputFileFormat.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/ECAT6ImageInputFileFormat.h"
#include "stir/IO/ECAT7ImageInputFileFormat.h"
#include "stir/IO/ECAT966ListmodeInputFileFormat.h"
#include "stir/IO/ECAT962ListmodeInputFileFormat.h"
#endif
#endif

#ifdef HAVE_ITK
#include "stir/IO/ITKOutputFileFormat.h"
#include "stir/IO/ITKImageInputFileFormat.h"
#endif

START_NAMESPACE_STIR

static InterfileOutputFileFormat::RegisterIt dummy1;
#ifdef HAVE_ITK
static ITKOutputFileFormat::RegisterIt dummyITK1;
#endif
static InterfileDynamicDiscretisedDensityOutputFileFormat::RegisterIt dummydynIntfIn;
static InterfileParametricDensityOutputFileFormat<ParametricVoxelsOnCartesianGridBaseType>::RegisterIt dummyparIntfIn;

#ifdef HAVE_LLN_MATRIX
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT6
static ECAT6OutputFileFormat::RegisterIt dummy2;
END_NAMESPACE_ECAT6
START_NAMESPACE_ECAT7
static ECAT7OutputFileFormat::RegisterIt dummy3;
static ECAT7DynamicDiscretisedDensityOutputFileFormat::RegisterIt dummydynecat7In;
static ECAT7ParametricDensityOutputFileFormat<ParametricVoxelsOnCartesianGridBaseType>::RegisterIt dummyparecat7In;
END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
#endif


static RegisterInputFileFormat<InterfileImageInputFileFormat> idummy0(0);
#ifdef HAVE_LLN_MATRIX
static RegisterInputFileFormat<ecat::ecat7::ECAT7ImageInputFileFormat> idummy2(4);

// ECAT6 very low priority it doesn't have a signature 
static RegisterInputFileFormat<ecat::ecat6::ECAT6ImageInputFileFormat> idummy4(100000);

static RegisterInputFileFormat<ecat::ecat7::ECAT7DynamicDiscretisedDensityInputFileFormat> dynidummy(0);
#endif
#ifdef HAVE_ITK
// we'll put it at low priority such that it is tried (almost) last, i.e. after STIR specific input routines
// This is because we translate the ITK info currently incompletely.
static RegisterInputFileFormat<ITKImageInputFileFormat> idummy8(10000);
#endif


/*************************** listmode data **********************/
#ifdef HAVE_LLN_MATRIX
static RegisterInputFileFormat<ecat::ecat7::ECAT966ListmodeInputFileFormat> LMdummyECAT966(4);
static RegisterInputFileFormat<ecat::ecat7::ECAT962ListmodeInputFileFormat> LMdummyECAT962(5);
#endif
END_NAMESPACE_STIR
