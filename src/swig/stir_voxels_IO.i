/*
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2020 - 2022 University College London
    Copyright (C) 2022 Positrigo
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: IO for stir:DiscretisedDensity

  \author Kris Thielemans
  \author Markus Jehl
*/

#ifdef STIRSWIG_SHARED_PTR
#define DataT stir::DiscretisedDensity<3,float>
%shared_ptr(stir::OutputFileFormat<stir::DiscretisedDensity<3,float> >);
%shared_ptr(stir::RegisteredObject< stir::OutputFileFormat< stir::DiscretisedDensity< 3,float > > >);
%shared_ptr(stir::RegisteredParsingObject< stir::InterfileOutputFileFormat, stir::OutputFileFormat<DataT >, stir::OutputFileFormat<DataT > >);
%shared_ptr(stir::InterfileOutputFileFormat);
#ifdef HAVE_LLN_MATRIX
%shared_ptr(stir::RegisteredParsingObject<stir::ecat::ecat7::ECAT7OutputFileFormat, stir::OutputFileFormat<DataT >, stir::OutputFileFormat<DataT > >);
%shared_ptr(stir::ecat::ecat7::ECAT7OutputFileFormat);
#endif

#ifdef HAVE_ITK
%shared_ptr(stir::RegisteredParsingObject< stir::ITKOutputFileFormat, stir::OutputFileFormat<DataT >, stir::OutputFileFormat<DataT > >);
%shared_ptr(stir::ITKOutputFileFormat);
#endif

#undef DataT
#endif

%include "stir/IO/OutputFileFormat.h"

#define DataT stir::DiscretisedDensity<3,float>
%template(Float3DDiscretisedDensityOutputFileFormat) stir::OutputFileFormat<DataT >;
  //cannot do that as pure virtual functions
  //%template(ROOutputFileFormat3DFloat) RegisteredObject< OutputFileFormat< DiscretisedDensity< 3,float > > >;
%template(RPInterfileOutputFileFormat) stir::RegisteredParsingObject<stir::InterfileOutputFileFormat, stir::OutputFileFormat<DataT >, stir::OutputFileFormat<DataT > >;

#ifdef HAVE_ITK
%template(RPITKOutputFileFormat) stir::RegisteredParsingObject<stir::ITKOutputFileFormat, stir::OutputFileFormat<DataT >, stir::OutputFileFormat<DataT > >;
#endif

%include "stir/IO/InterfileOutputFileFormat.h"
#ifdef HAVE_LLN_MATRIX
%include "stir/IO/ECAT7OutputFileFormat.h"
#endif

#ifdef HAVE_ITK
%include "stir/IO/ITKOutputFileFormat.h"
#endif

#undef DataT
