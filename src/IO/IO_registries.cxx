//
// $Id$
//
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
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
  
  $Date$
  $Revision$
*/

#include "stir/IO/InterfileOutputFileFormat.h"
#include "stir/IO/ECAT6OutputFileFormat.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/ECAT7OutputFileFormat.h"
#endif


#if 1
#include "stir/IO/InputFileFormatRegistry.h"
#include "stir/IO/InterfileImageInputFileFormat.h"
#include "stir/IO/ECAT6ImageInputFileFormat.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/ECAT7ImageInputFileFormat.h"
#endif
#endif

START_NAMESPACE_STIR

static InterfileOutputFileFormat::RegisterIt dummy1;
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT6
static ECAT6OutputFileFormat::RegisterIt dummy2;
END_NAMESPACE_ECAT6
#ifdef HAVE_LLN_MATRIX
START_NAMESPACE_ECAT7
static ECAT7OutputFileFormat::RegisterIt dummy3;
END_NAMESPACE_ECAT7
#endif
END_NAMESPACE_ECAT


#if 1
static RegisterInputFileFormat<InterfileImageInputFileFormat> idummy0(0);
#ifdef HAVE_LLN_MATRIX
static RegisterInputFileFormat<ecat::ecat7::ECAT7ImageInputFileFormat> idummy2(4);
#endif
static RegisterInputFileFormat<ecat::ecat6::ECAT6ImageInputFileFormat> idummy4(100000); // very low priority it doesn't have a signature 
#else
// TODO
InputFileFormatRegistry<DiscretisedDensity<3,float> >  registry;
  //  *InputFileFormatRegistry<DiscretisedDensity<3,float> >::default_sptr();
  registry.add_to_registry(new InterfileImageInputFileFormat, 0);
#ifdef HAVE_LLN_MATRIX
  registry.add_to_registry(new ecat::ecat7::ECAT7ImageInputFileFormat,4);
#endif
  registry.add_to_registry(new ecat::ecat6::ECAT6ImageInputFileFormat,100000); // last as it doesn't have a signature 
#endif
END_NAMESPACE_STIR
