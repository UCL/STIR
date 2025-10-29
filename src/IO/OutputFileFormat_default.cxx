//
//
/*
    Copyright (C) 2005- 2009-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0

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

#include "stir/DynamicDiscretisedDensity.h"
#ifndef MINI_STIR
#  include "stir/modelling/ParametricDiscretisedDensity.h"
#  ifdef HAVE_LLN_MATRIX
#    include "stir/IO/ECAT7ParametricDensityOutputFileFormat.h"
#    include "stir/IO/ECAT7DynamicDiscretisedDensityOutputFileFormat.h"
#  else
#    include "stir/modelling/KineticParameters.h"
#    include "stir/IO/InterfileParametricDiscretisedDensityOutputFileFormat.h"
#    include "stir/IO/InterfileDynamicDiscretisedDensityOutputFileFormat.h"
#    include "stir/IO/MultiDynamicDiscretisedDensityOutputFileFormat.h"
#  endif
#endif

START_NAMESPACE_STIR

template <>
shared_ptr<OutputFileFormat<DiscretisedDensity<3, float>>>
    OutputFileFormat<DiscretisedDensity<3, float>>::_default_sptr(new InterfileOutputFileFormat);

#ifndef MINI_STIR
#  if 0
  template <>
  shared_ptr<OutputFileFormat<ParametricDiscretisedDensity<3,KineticParameters<2,float> > > > 
  OutputFileFormat<ParametricDiscretisedDensity<3,KineticParameters<2,float> > >::_default_sptr = 
  new InterfileParametricDiscretisedDensityOutputFileFormat<3,KineticParameters<2,float> >;
#  else
template <>
shared_ptr<OutputFileFormat<ParametricVoxelsOnCartesianGrid>> OutputFileFormat<ParametricVoxelsOnCartesianGrid>::_default_sptr(
#    ifdef HAVE_LLN_MATRIX
    new ecat::ecat7::ECAT7ParametricDensityOutputFileFormat<ParametricVoxelsOnCartesianGridBaseType>
#    else
    new InterfileParametricDiscretisedDensityOutputFileFormat<ParametricVoxelsOnCartesianGridBaseType>
#    endif
);
#  endif
#  if 1
template <>
shared_ptr<OutputFileFormat<DynamicDiscretisedDensity>> OutputFileFormat<DynamicDiscretisedDensity>::_default_sptr(
#    ifdef HAVE_LLN_MATRIX
    new ecat::ecat7::ECAT7DynamicDiscretisedDensityOutputFileFormat
#    else
    new InterfileDynamicDiscretisedDensityOutputFileFormat
#    endif
);
#  endif
#endif

END_NAMESPACE_STIR
