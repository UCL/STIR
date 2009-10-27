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
#ifndef __stir_numerics_BSplines__H__
#define __stir_numerics_BSplines__H__

/*!
\file 
\ingroup BSpline
\brief Implementation of the basic components and declarations for B-Splines Interpolation 

\author Charalampos Tsoumpas
\author Kris Thielemans
  
$Date$
$Revision$
*/

#include "stir/common.h"

START_NAMESPACE_STIR

namespace BSpline {

   /*! \brief The type used for relative positions between the grid points.
     \ingroup BSpline
   */
  typedef double pos_type;

  /*! \brief enum providing constants to define the type of B-Spline used for interpolation
    \ingroup BSpline 
  */
  enum BSplineType 
    {near_n, linear, quadratic, cubic, quartic, quintic, oMoms} ;

  /*! \brief compute BSpline coefficients that gives the BSpline that interpolates the given data
  \internal
  \ingroup BSpline
  You should never have to use this function yourself.
  */
  template <class RandIterOut, class IterT>
    inline  
    void
    BSplines_coef(RandIterOut c_begin_iterator, 
                  RandIterOut c_end_iterator,
                  IterT input_begin_iterator, 
                  IterT input_end_iterator, 
                  const double z1, const double z2, const double lambda); // to be taken from the class

  /*! \brief return value of the first derivative of the spline 
      \ingroup BSpline
  */
  template <typename pos_type>
    inline 
    pos_type 
    BSplines_1st_der_weight(const pos_type relative_position, const BSplineType spline_type) ;

  /*! \brief return spline value 
      \ingroup BSpline
  */
  template <typename pos_type>
    inline
    pos_type 
    BSplines_weights(const pos_type relative_position, const BSplineType spline_type);


  //*/
} // end BSpline namespace

END_NAMESPACE_STIR

#include "stir/numerics/BSplines_weights.inl"
#include "stir/numerics/BSplines_coef.inl"

#endif
