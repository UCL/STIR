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
  \brief Definition of class stir::SpatialTransformation
  \author Charalampos Tsoumpas
 
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
