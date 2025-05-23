//
/*
    Copyright (C) 2009 - 2013, King's College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
 */
 /*!
  \file
  \ingroup spatial_transformation
  \brief Implementations of inline functions of class stir::SpatialTransformation

  \author Charalampos Tsoumpas

  This is the most basic class for including Motion Fields. 

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
