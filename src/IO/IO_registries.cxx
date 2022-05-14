/*
    Copyright (C) 2002-2011, Hammersmith Imanet Ltd
    Copyright (C) 2012, Kris Thielemans
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2013, University College London
    
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
#include "stir/IO/InterfileDynamicDiscretisedDensityInputFileFormat.h"
#include "stir/IO/InterfileParametricDiscretisedDensityInputFileFormat.h"
#include "stir/IO/InterfileParametricDiscretisedDensityOutputFileFormat.h"
#include "stir/IO/MultiDynamicDiscretisedDensityInputFileFormat.h"
#include "stir/IO/MultiDynamicDiscretisedDensityOutputFileFormat.h"
#include "stir/IO/MultiParametricDiscretisedDensityInputFileFormat.h"
#include "stir/IO/MultiParametricDiscretisedDensityOutputFileFormat.h"
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
#include "stir/IO/ECAT8_32bitListmodeInputFileFormat.h"

#ifdef HAVE_HDF5
#include "stir/IO/GEHDF5ListmodeInputFileFormat.h"
#endif

//! Addition for SAFIR listmode input file format
#include "stir/IO/SAFIRCListmodeInputFileFormat.h"

//! Addition for ROOT support - Nikos Efthimiou
#ifdef HAVE_CERN_ROOT
#include "stir/IO/ROOTListmodeInputFileFormat.h"
#include "stir/IO/InputStreamFromROOTFileForCylindricalPET.h"
#include "stir/IO/InputStreamFromROOTFileForECATPET.h"
#endif

#ifdef HAVE_UPENN
#include "stir/IO/PENNListmodeInputFileFormat.h"
#include "stir/IO/InputStreamWithRecordsFromUPENNbin.h"
#include "stir/IO/InputStreamWithRecordsFromUPENNtxt.h"
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
static InterfileDynamicDiscretisedDensityOutputFileFormat::RegisterIt dummydynIntfOut;
static InterfileParametricDiscretisedDensityOutputFileFormat<ParametricVoxelsOnCartesianGridBaseType>::RegisterIt dummyparIntfOut;
static MultiDynamicDiscretisedDensityOutputFileFormat::RegisterIt dummydynMultiOut;
static MultiParametricDiscretisedDensityOutputFileFormat<ParametricVoxelsOnCartesianGridBaseType>::RegisterIt dummyparMultiOut;

//! Support for SAFIR listmode file format
static RegisterInputFileFormat<SAFIRCListmodeInputFileFormat> LMdummySAFIR(4);


//!
//! \brief LMdummyROOT
//! \author Nikos Efthimiou
//! \details ROOT support
#ifdef HAVE_CERN_ROOT
static RegisterInputFileFormat<ROOTListmodeInputFileFormat> LMdummyROOT(6);
static InputStreamFromROOTFileForCylindricalPET::RegisterIt dummy60606;
static InputStreamFromROOTFileForECATPET::RegisterIt dummy606062;
#endif



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
static RegisterInputFileFormat<ITKImageInputFileFormat<DiscretisedDensity<3,float> > > idummy6(10000);
static RegisterInputFileFormat<ITKImageInputFileFormat<DiscretisedDensity<3,CartesianCoordinate3D<float> > > > idummy7(10000);
#endif
static RegisterInputFileFormat<InterfileDynamicDiscretisedDensityInputFileFormat> dyndummy_intf(1);
static RegisterInputFileFormat<InterfileParametricDiscretisedDensityInputFileFormat> paradummy_intf(1);
static RegisterInputFileFormat<MultiDynamicDiscretisedDensityInputFileFormat> dynim_dummy_multi(1);
static RegisterInputFileFormat<MultiParametricDiscretisedDensityInputFileFormat> parim_dummy_multi(1);


/*************************** listmode data **********************/
#ifdef HAVE_LLN_MATRIX
static RegisterInputFileFormat<ecat::ecat7::ECAT966ListmodeInputFileFormat> LMdummyECAT966(4);
static RegisterInputFileFormat<ecat::ecat7::ECAT962ListmodeInputFileFormat> LMdummyECAT962(5);
#endif
static RegisterInputFileFormat<ecat::ECAT8_32bitListmodeInputFileFormat> LMdummyECAT8(6);
#ifdef HAVE_HDF5
static RegisterInputFileFormat<GE::RDF_HDF5::GEHDF5ListmodeInputFileFormat> LMdummyGEHDF5(7);
#endif

#ifdef HAVE_UPENN
static RegisterInputFileFormat<PENNListmodeInputFileFormat> LMdummyPENN(8);
static InputStreamWithRecordsFromUPENNbin::RegisterIt dummy68606;
static InputStreamWithRecordsFromUPENNtxt::RegisterIt dummy686062;
//static RegisterInputFileFormat<PENNbinListmodeInputFileFormat> LMdummyPENNbin(9);
//static RegisterInputFileFormat<PENNImageInputFileFormat> idummy1(2);
#endif

END_NAMESPACE_STIR
