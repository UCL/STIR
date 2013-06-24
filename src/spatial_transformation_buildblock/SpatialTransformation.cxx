//
/*
    Copyright (C) 2009 - 2013, King's College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.3 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
 */
 /*!
  \file
  \ingroup spatial_transformation
  \brief Implementations of inline functions of class stir::SpatialTransformation

  \author Charalampos Tsoumpas

  This is the most basic class for including Motion Fields. 

  $Date$
  $Revision$
*/


#include "stir/spatial_transformation/SpatialTransformation.h"


START_NAMESPACE_STIR

const char * const 
SpatialTransformation::registered_name = "Motion Field Type";

SpatialTransformation::SpatialTransformation()    //!< default constructor
{ }

SpatialTransformation::~SpatialTransformation()   //!< default destructor
{ }

END_NAMESPACE_STIR
