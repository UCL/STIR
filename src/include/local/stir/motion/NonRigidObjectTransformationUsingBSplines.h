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

  \brief Declaration of class stir::NonRigidObjectTransformationUsingBSplines

  \author Kris Thielemans
  $Date$
  $Revision$
*/

#ifndef __stir_motion_NonRigidObjectTransformationUsingBSplines_H__
#define __stir_motion_NonRigidObjectTransformationUsingBSplines_H__


#include "local/stir/motion/ObjectTransformation.h"
#include "stir/CartesianCoordinate3D.h"
#include "local/stir/BSplines.h"
#include "local/stir/BSplinesRegularGrid.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"
//#include "stir/DiscretisedDensityOnCartesianGrid.h"

START_NAMESPACE_STIR
class Succeeded;

template <int num_dimensions, class elemT>
  class DeformationFieldOnCartesianGrid : 
 public BasicCoordinate<num_dimensions, Array<num_dimensions, elemT> >
//  public DiscretisedDensityOnCartesianGrid<num_dimensions, BasicCoordinate<num_dimensions, elemT> >
{
 public:
  DeformationFieldOnCartesianGrid() {}
  
};

/*! \ingroup  motion
  \brief Class to perform non-rigid object transformations in arbitrary dimensions

*/
template <int num_dimensions, class elemT>
class NonRigidObjectTransformationUsingBSplines
  : 
  public 
  RegisteredParsingObject<NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>,
                          ObjectTransformation<num_dimensions,elemT>,
                          ObjectTransformation<num_dimensions,elemT> >
{
public:
  static const char * const registered_name;

  NonRigidObjectTransformationUsingBSplines();

  //! Transform point 
  virtual
    BasicCoordinate<num_dimensions,elemT>
    transform_point(const BasicCoordinate<num_dimensions,elemT>& point) const;

  float jacobian(const BasicCoordinate<num_dimensions,elemT>& point) const;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  virtual void set_key_values();
private:
  BasicCoordinate<num_dimensions, BSpline::BSplinesRegularGrid<num_dimensions,elemT,elemT> > interpolator;
  BasicCoordinate<num_dimensions,elemT> _grid_spacing;
  BasicCoordinate<num_dimensions,elemT> _origin;
  BSpline::BSplineType _bspline_type;

  // use for parsing only
  shared_ptr< DeformationFieldOnCartesianGrid<num_dimensions,elemT> > deformation_field_sptr;
  int _bspline_order;
};
#if 0
//! Output to (text) stream
/*! \ingroup motion
*/
std::ostream&
operator<<(std::ostream& out,
	   const NonRigidObjectTransformationUsingBSplines& rigid_object_transformation);
//! Input from (text) stream
/*! \ingroup motion
*/
std::istream&
operator>>(std::istream& ,
	   NonRigidObjectTransformationUsingBSplines& rigid_object_transformation);
#endif


END_NAMESPACE_STIR

#endif
