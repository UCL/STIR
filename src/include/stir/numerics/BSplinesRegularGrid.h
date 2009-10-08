//
// $Id$
//
/*
  Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
#ifndef __stir_numerics_BSplinesRegularGrid__H__
#define __stir_numerics_BSplinesRegularGrid__H__
/*!
  \file 
  \ingroup BSpline
  \brief Implementation of the n-dimensional B-Splines Interpolation 

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/

#include "stir/Array.h"
#include "stir/BasicCoordinate.h"
#include "stir/numerics/BSplines.h"
START_NAMESPACE_STIR

namespace BSpline {

  /*! \brief The type used for relative positions between the grid points.
     \ingroup BSpline
   */
  typedef double pos_type;

  /*! \ingroup BSpline
      \brief A class for n-dimensional BSpline interpolation when the input samples
      are on a regular grid.

      This class is essentially an n-dimensional function, so can be used as a function object.

      \par Example
      \code
      Array<3,float> some_data = ....;
      
      BSplinesRegularGrid<3, float> interpolator(cubic);

      interpolator.set_coef(some_data);

      Coordinate3D<double> position(1.2, 3.4, 5.6);

      float value = interpolator(position);
      \endcode
   */
  template <int num_dimensions, typename out_elemT, typename in_elemT = out_elemT>
    class BSplinesRegularGrid
    {
		
    public:
      // only there for tests
      Array<num_dimensions,out_elemT> get_coefficients() const
	{ return this->_coeffs;  }
	
	
	
      //! constructor given an array of samples and the spline type
      inline
	explicit
	BSplinesRegularGrid(const Array<num_dimensions,in_elemT> & input,
			    const BSplineType & this_type = cubic)
	{
	  this->set_private_values(this_type);	  
	  this->set_coef(input);		
	} 	
	
      //! constructor given an array of samples and a different spline type for every dimension
      inline 
	BSplinesRegularGrid(const Array<num_dimensions,in_elemT> & input,
			    const BasicCoordinate<num_dimensions, BSplineType> & this_type)	
	{	 
	  this->set_private_values(this_type);	  
	  this->set_coef(input);
	}  	

      //! constructor that only sets the spline type
      /*! You need to call set_coef() first before you will get a sensible result.*/
      inline 
	explicit
	BSplinesRegularGrid(
			    const BSplineType & this_type = cubic)
	{
	  this->set_private_values(this_type);	  
	} 	
	
      //! constructor that only sets a different spline type for every dimension
      /*! You need to call set_coef() first before you will get a sensible result.*/
      inline 
	explicit
	BSplinesRegularGrid(const BasicCoordinate<num_dimensions, BSplineType> & this_type)	
	{	 
	  this->set_private_values(this_type);	  
	}  	

      //! destructor
      inline ~BSplinesRegularGrid();
			
      //! Compute the coefficients for the B-splines from an array of samples.
      /*! When the order of the spline is larger than 1, the coefficients multiplying
	  the basic splines are not equal to the samples. This variable stores them
	  for further use.
	  \todo rename
      */
      inline
	void
	set_coef(const Array<num_dimensions,in_elemT> & input);

      //! Compute value of the interpolator
      /*! \param relative_positions
	     A coordinate with respect to the original grid coordinates as used by the 
	     input array. In particular, if the input array was not 0-based, your 
	     \c  relative_positions should not be either.
	  \return the interpolated value.

	  
	  \todo should probably be templated in pos_type.
      */
      inline
	const out_elemT 
	operator() (const BasicCoordinate<num_dimensions,pos_type>& relative_positions) const;

      //! Compute gradient of the interpolator
      /*! \param relative_positions
	     A coordinate with respect to the original grid coordinates as used by the 
	     input array. In particular, if the input array was not 0-based, your 
	     \c  relative_positions should not be either.
	  \return the gradient

	  \todo should probably be templated in pos_type.
      */
      inline
	const BasicCoordinate<num_dimensions, out_elemT> 
	gradient(const BasicCoordinate<num_dimensions,pos_type>& relative_positions) const;

    private:

      // variables that store numbers for the spline type
      // TODO these coefficients and the spline type could/should be integrated into 1 class
      BasicCoordinate<num_dimensions,BSplineType> _spline_types;
      BasicCoordinate<num_dimensions,double> _z1s;
      BasicCoordinate<num_dimensions,double> _z2s;
      BasicCoordinate<num_dimensions,double> _lambdas;
      //! coefficients for B-splines
      Array<num_dimensions,out_elemT> _coeffs;  
		
      inline void 
	set_private_values(const BasicCoordinate<num_dimensions, BSplineType> & this_type);
      inline void 
	set_private_values(const BSplineType & this_type);
	
    };
	
} // end BSpline namespace

END_NAMESPACE_STIR

#include "stir/numerics/BSplinesRegularGrid.inl"

#endif
