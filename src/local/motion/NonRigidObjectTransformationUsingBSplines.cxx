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

  \brief Implementation of class stir::NonRigidObjectTransformationUsingBSplines

  \author  Kris Thielemans
  $Date$
  $Revision$
*/

#include "local/stir/motion/NonRigidObjectTransformationUsingBSplines.h"
#include "stir/stream.h"//xxx
#include "stir/numerics/determinant.h"
#include "stir/IndexRange2D.h"
#include <iostream>
START_NAMESPACE_STIR

template <>
const char * const 
NonRigidObjectTransformationUsingBSplines<3,float>::registered_name = "BSplines transformation"; 

template <int num_dimensions, class elemT>
void
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
set_defaults()
{}


template <int num_dimensions, class elemT>
void 
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
initialise_keymap()
{
  this->deformation_field_sptr = new DeformationFieldOnCartesianGrid<num_dimensions,elemT>;
  this->parser.add_key("grid spacing", &this->_grid_spacing);
  this->parser.add_key("origin", &this->_origin);
  this->parser.add_key("deformation field", this->deformation_field_sptr.get());

  this->parser.add_start_key("BSplines Transformation Parameters");
  this->parser.add_stop_key("End BSplines Transformation Parameters");
}

template <int num_dimensions, class elemT>
void 
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
set_key_values()
{
  for (int i=1; i<=num_dimensions; ++i)
    (*this->deformation_field_sptr)[i] =
      this->interpolator[i].get_coefficients();
}
    
template <int num_dimensions, class elemT>
bool 
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
post_processing()
{
  for (int i=1; i<=num_dimensions; ++i)
    this->interpolator[i] = 
      BSpline::BSplinesRegularGrid<num_dimensions,elemT,elemT>((*this->deformation_field_sptr)[i]);
  // deallocate data for deformation field
  // at present, have to do this by assigning an object as opposed to 0
  // in case we want to parse twice
  // WARNING: do not reassign a new pointer, as the keymap stores a pointer to the deformation_field object
  *(this->deformation_field_sptr)  = DeformationFieldOnCartesianGrid<num_dimensions,elemT>();

  return false;
}


template <int num_dimensions, class elemT>
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
NonRigidObjectTransformationUsingBSplines()
{
  this->set_defaults();
}


template <int num_dimensions, class elemT>
BasicCoordinate<num_dimensions,elemT>
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
transform_point(const BasicCoordinate<num_dimensions,elemT>& point) const
{
  // note: current Bspline needs double here
  const BasicCoordinate<num_dimensions,double> point_in_grid_coords =
    BasicCoordinate<num_dimensions,double>((point - this->_origin)/this->_grid_spacing);
  BasicCoordinate<num_dimensions,elemT> result;
  for (int i=1; i<=num_dimensions; ++i)
    result[i]= this->interpolator[i](point_in_grid_coords);
  return result + point;
}

template <int num_dimensions, class elemT>
float
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
jacobian(const BasicCoordinate<num_dimensions,elemT>& point) const
{
  // note: current Bspline needs double here
  const BasicCoordinate<num_dimensions,double> point_in_grid_coords =
    BasicCoordinate<num_dimensions,double>((point - this->_origin)/this->_grid_spacing);
  Array<2,float> jacobian_matrix(IndexRange2D(1,num_dimensions,1,num_dimensions));
  for (int i=1; i<=num_dimensions; ++i)
    {
      BasicCoordinate<num_dimensions,elemT> gradient =
	this->interpolator[i].gradient(point_in_grid_coords);
    std::copy(gradient.begin(), gradient.end(), jacobian_matrix[i].begin());
    }
  return 
    determinant(jacobian_matrix);
}
  


////////////////////// instantiations
template class NonRigidObjectTransformationUsingBSplines<3,float>;
END_NAMESPACE_STIR
