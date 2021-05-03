//
//
/*
    Copyright (C) 2008- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup IO
  \brief Instantiations for class stir::InputFileFormatRegistry

  \author Kris Thielemans

*/

#include "stir/IO/InputFileFormatRegistry.txx"

#include "stir/DiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"  
#include "stir/DynamicDiscretisedDensity.h" 
#include "stir/listmode/ListModeData.h"

START_NAMESPACE_STIR

// instantiations
template class InputFileFormatRegistry<DiscretisedDensity<3,float> >;
template class InputFileFormatRegistry<ParametricVoxelsOnCartesianGrid >;
template class InputFileFormatRegistry<DynamicDiscretisedDensity>;
template class InputFileFormatRegistry<ListModeData>;
template class InputFileFormatRegistry<DiscretisedDensity<3,CartesianCoordinate3D<float> > >;

END_NAMESPACE_STIR
