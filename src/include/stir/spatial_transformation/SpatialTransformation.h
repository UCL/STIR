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
  \brief Definition of class stir::SpatialTransformation
  \author Charalampos Tsoumpas
 
  $Date$
  $Revision$
*/

#ifndef __stir_spatial_transformation_SpatialTransformation_H__
#define __stir_spatial_transformation_SpatialTransformation_H__

#include "stir/RegisteredObject.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

/*! 
  \brief base class for any type of motion fields
  \ingroup spatial_transformation
  At present very basic. It just provides the parsing mechanism.
*/
class SpatialTransformation: public RegisteredObject<SpatialTransformation> 
{ 
 public:
  static const char * const registered_name ; 
  //! default constructor
  SpatialTransformation();

  //! default destructor
  virtual ~SpatialTransformation();

  virtual Succeeded set_up() = 0;
};

END_NAMESPACE_STIR

#endif //__stir_spatial_transformation_SpatialTransformation_H__
