//
//
/*
    Copyright (C) 2005- 2009-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
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
      
*/

#include "stir/IO/InterfileOutputFileFormat.h"
#include "stir/DiscretisedDensity.h"

#include "stir/modelling/ParametricDiscretisedDensity.h"  
#include "stir/DynamicDiscretisedDensity.h" 
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/ECAT7ParametricDensityOutputFileFormat.h" 
#include "stir/IO/ECAT7DynamicDiscretisedDensityOutputFileFormat.h" 
#else
#include "stir/modelling/KineticParameters.h"  
#include "stir/IO/InterfileParametricDensityOutputFileFormat.h" 
#include "stir/IO/InterfileDynamicDiscretisedDensityOutputFileFormat.h" 
#endif

START_NAMESPACE_STIR

template <>
shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > >
OutputFileFormat<DiscretisedDensity<3,float> >::_default_sptr(new InterfileOutputFileFormat);


#if 0
  template <>
  shared_ptr<OutputFileFormat<ParametricDiscretisedDensity<3,KineticParameters<2,float> > > > 
  OutputFileFormat<ParametricDiscretisedDensity<3,KineticParameters<2,float> > >::_default_sptr = 
  new InterfileParametricDensityOutputFileFormat<3,KineticParameters<2,float> >;
#else
  template <>
  shared_ptr<OutputFileFormat<ParametricVoxelsOnCartesianGrid > >
  OutputFileFormat<ParametricVoxelsOnCartesianGrid>::_default_sptr(
#ifdef HAVE_LLN_MATRIX
  new ecat::ecat7::ECAT7ParametricDensityOutputFileFormat<ParametricVoxelsOnCartesianGridBaseType>
#else
  new InterfileParametricDensityOutputFileFormat<ParametricVoxelsOnCartesianGridBaseType >
#endif
								   );
#endif
#if 1
  template <>
  shared_ptr<OutputFileFormat<DynamicDiscretisedDensity > >
  OutputFileFormat<DynamicDiscretisedDensity>::
  _default_sptr(
#ifdef HAVE_LLN_MATRIX
  new ecat::ecat7::ECAT7DynamicDiscretisedDensityOutputFileFormat
#else
  new InterfileDynamicDiscretisedDensityOutputFileFormat
#endif
		);
#endif

END_NAMESPACE_STIR




