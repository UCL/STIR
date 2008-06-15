//
// $Id$
//
/*
    Copyright (C) 2008- $Date$, Hammersmith Imanet Ltd
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

  $Date$
  $Revision$
*/

#include "stir/IO/InputFileFormatRegistry.txx"

// instantiations
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

template class InputFileFormatRegistry<DiscretisedDensity<3,float> >;

END_NAMESPACE_STIR
