//
// $Id$
//
/*
  Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
  \brief  initialisation of the stir::OutputFileFormat::_default_sptr member
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/

#include "local/stir/modelling/ParametricDiscretisedDensity.h"  
#include "local/stir/DynamicDiscretisedDensity.h" 
#ifdef HAVE_LLN_MATRIX
#include "local/stir/IO/ECAT7ParametricDensityOutputFileFormat.h" 
#include "local/stir/IO/ECAT7DynamicDiscretisedDensityOutputFileFormat.h" 
#else
#include "local/stir/modelling/KineticParameters.h"  
#include "local/stir/IO/InterfileParametricDensityOutputFileFormat.h" 
#include "local/stir/IO/InterfileDynamicDiscretisedDensityOutputFileFormat.h" 
#endif

START_NAMESPACE_STIR
#if 0
  template <>
  shared_ptr<OutputFileFormat<ParametricDiscretisedDensity<3,KineticParameters<2,float> > > > 
  OutputFileFormat<ParametricDiscretisedDensity<3,KineticParameters<2,float> > >::_default_sptr = 
  new InterfileParametricDensityOutputFileFormat<3,KineticParameters<2,float> >;
#else
  template <>
  shared_ptr<OutputFileFormat<ParametricVoxelsOnCartesianGrid > >
  OutputFileFormat<ParametricVoxelsOnCartesianGrid>::_default_sptr = 
#ifdef HAVE_LLN_MATRIX
  new ecat::ecat7::ECAT7ParametricDensityOutputFileFormat<ParametricVoxelsOnCartesianGridBaseType>;
#else
  new InterfileParametricDensityOutputFileFormat<ParametricVoxelsOnCartesianGridBaseType >;
#endif
#endif
#if 0
  template <>
  shared_ptr<OutputFileFormat<DynamicDiscretisedDensity > >
  OutputFileFormat<DynamicDiscretisedDensity>::
  _default_sptr = 
#ifdef HAVE_LLN_MATRIX
  new ecat::ecat7::ECAT7DynamicDiscretisedDensityOutputFileFormat;
#else
  new InterfileDynamicDiscretisedDensityOutputFileFormat;
#endif
#endif

END_NAMESPACE_STIR

