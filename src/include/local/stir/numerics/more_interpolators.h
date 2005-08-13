//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
    For internal GE use only.
*/
#ifndef __stir_numerics_more_interpolators_H__
#define __stir_numerics_more_interpolators_H__
/*!
  \file
  \ingroup numerics
  \brief Functions to interpolate data

  All preliminary.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/BasicCoordinate.h"
#include "stir/Array.h"


START_NAMESPACE_STIR

/*! \ingroup numerics
  \brief Pull \a value from the input array using nearest neigbour interpolation.

  Adds \a value to the grid point nearest to \a  point_in_output_coords
*/
template <class elemT, class positionT>
elemT
pull_nearest_neighbour_interpolate(const Array<3, elemT>& in, 
				   const BasicCoordinate<3, positionT>& point_in_input_coords);

/*! \ingroup numerics
  \brief Push \a value into the output array using nearest neigbour interpolation.

  Adds \a value to the grid point nearest to \a  point_in_output_coords
*/
template <int num_dimensions, class elemT, class positionT, class valueT>
void
push_nearest_neighbour_interpolate(Array<num_dimensions, elemT>& out, 
				   const BasicCoordinate<num_dimensions, positionT>& point_in_output_coords,
				   valueT value);


/*! \ingroup numerics
  \brief Returns an interpolated value according to \a point_in_input_coords.
*/
template <class elemT, class positionT>
elemT
pull_linear_interpolate(const Array<3, elemT>& in, 
			const BasicCoordinate<3, positionT>& point_in_input_coords);

/*! \ingroup numerics
  \brief Push \a value into the output array using the transpose of linear interpolation.
*/
template <class elemT, class positionT, class valueT>
void
push_transpose_linear_interpolate(Array<3, elemT>& out, 
				  const BasicCoordinate<3, positionT>& point_in_output_coords,
				  valueT value);

/*! \ingroup numerics
  \brief A function object to pull interpolated values from the input array into the grid points of the output array.

  \todo preliminary. We might want to derive this from a class or so.
*/
template <class elemT>
class
PullLinearInterpolator
{
public:
  PullLinearInterpolator()
    : _input_ptr(0)
  {}

  void set_input(const Array<3, elemT>& input) const
  {
    this->_input_ptr = &input;
  }

  template <class positionT>
  elemT operator()(const BasicCoordinate<3, positionT>& point_in_input_coords) const
  {
    return 
      pull_linear_interpolate(*(this->_input_ptr), 
			      point_in_input_coords);
  }
private:
  // todo terribly dangerous
  // we have it such that we can have a default constructor without any arguments
  // this means we cannot use a reference
  // we could use a shared_ptr, but then we can only interpolate data for which we have a shared_ptr
  mutable const Array<3, elemT> * _input_ptr;
};



/*! \ingroup numerics
  \brief A function object to push values at the grid of the input array into the output array 

  \todo preliminary. We might want to derive this from a class or so.
*/
template <class elemT>
class
PushTransposeLinearInterpolator
{
public:
  PushTransposeLinearInterpolator()
    :_output_ptr(0)
  {}

  void set_output(Array<3, elemT>& output) const
  { 
   this->_output_ptr = &output;
  }

  template <class positionT, class valueT>
  void add_to(const BasicCoordinate<3, positionT>& point_in_output_coords, const valueT value) const
  {
    push_transpose_linear_interpolate(*(this->_output_ptr), 
				      point_in_output_coords,
				      value);
  }
private:
  // todo terribly dangerous
  // we have it such that we can have a default constructor without any arguments
  // this means we cannot use a reference
  // we could use a shared_ptr, but then we can only interpolate data for which we have a shared_ptr
  mutable Array<3, elemT> * _output_ptr;
};

/*! \ingroup numerics
  \brief A function object to pull interpolated values from the input array into the grid points of the output array.

  \todo preliminary. We might want to derive this from a class or so.
*/
template <class elemT>
class
PullNearestNeighbourInterpolator
{
public:
  PullNearestNeighbourInterpolator()
    : _input_ptr(0)
  {}

  void set_input(const Array<3, elemT>& input) const
  {
    this->_input_ptr = &input;
  }

  template <class positionT>
  elemT operator()(const BasicCoordinate<3, positionT>& point_in_input_coords) const
  {
    return 
      pull_nearest_neighbour_interpolate(*(this->_input_ptr), 
			      point_in_input_coords);
  }
private:
  // todo terribly dangerous
  // we have it such that we can have a default constructor without any arguments
  // this means we cannot use a reference
  // we could use a shared_ptr, but then we can only interpolate data for which we have a shared_ptr
  mutable const Array<3, elemT> * _input_ptr;
};



/*! \ingroup numerics
  \brief A function object to push values at the grid of the input array into the output array 

  \todo preliminary. We might want to derive this from a class or so.
*/
template <class elemT>
class
PushNearestNeighbourInterpolator
{
public:
  PushNearestNeighbourInterpolator()
    :_output_ptr(0)
  {}

  void set_output(Array<3, elemT>& output) const
  { 
   this->_output_ptr = &output;
  }

  template <class positionT, class valueT>
  void add_to(const BasicCoordinate<3, positionT>& point_in_output_coords, const valueT value) const
  {
    push_nearest_neighbour_interpolate(*(this->_output_ptr), 
				      point_in_output_coords,
				      value);
  }
private:
  // todo terribly dangerous
  // we have it such that we can have a default constructor without any arguments
  // this means we cannot use a reference
  // we could use a shared_ptr, but then we can only interpolate data for which we have a shared_ptr
  mutable Array<3, elemT> * _output_ptr;
};

END_NAMESPACE_STIR

#include "local/stir/numerics/more_interpolators.inl"

#endif
