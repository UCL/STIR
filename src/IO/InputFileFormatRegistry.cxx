//
//
/*
    Copyright (C) 2008- 2011, Hammersmith Imanet Ltd
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
  \brief Instantiations for class stir::InputFileFormatRegistry

  \author Kris Thielemans

*/

#include "stir/IO/InputFileFormatRegistry.txx"

#include "stir/DiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"  
#include "stir/DynamicDiscretisedDensity.h" 
#include "stir/listmode/CListModeData.h"

START_NAMESPACE_STIR

// instantiations
template class InputFileFormatRegistry<DiscretisedDensity<3,float> >;
template class InputFileFormatRegistry<ParametricVoxelsOnCartesianGrid >;
template class InputFileFormatRegistry<DynamicDiscretisedDensity>;
template class InputFileFormatRegistry<CListModeData>;

END_NAMESPACE_STIR
