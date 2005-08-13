//
// $Id$
//
/*
    Copyright (C) 2005- $Date$ , Hammersmith Imanet Ltd
    For internal GE use only
*/
/*!
  \file
  \ingroup motion

  \brief Declaration of class stir::ObjectTransformation

  \author Kris Thielemans
  $Date$
  $Revision$
*/

#ifndef __stir_motion_ObjectTransformation_H__
#define __stir_motion_ObjectTransformation_H__


#include "stir/BasicCoordinate.h"
#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"

START_NAMESPACE_STIR

/*! \ingroup  motion
  \brief Base-class for performing (potentially non-rigid) object transformations 
*/
template <int num_dimensions, class elemT>
class ObjectTransformation :
  public RegisteredObject<ObjectTransformation<num_dimensions, elemT> >,
  public ParsingObject
{
public:

  virtual ~ObjectTransformation() {}
  //! Transform point 
  /* \todo should be CartesianCoordinate<num_dimensions,elemT>, but we don't have that class yet*/
  virtual BasicCoordinate<num_dimensions,elemT> 
    transform_point(const BasicCoordinate<num_dimensions,elemT>& point) const = 0;

  //! Returns the determinant of the Jacobian matrix
  /*! This is related to the volume-element change due to the transformation. */
  virtual float
    jacobian(const BasicCoordinate<num_dimensions,elemT>& point) const = 0;
};

END_NAMESPACE_STIR

#endif
