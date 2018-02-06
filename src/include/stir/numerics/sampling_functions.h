//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
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
/*!
  \file 
  \ingroup numerics
  \brief Sampling functions (currently only stir::sample_function_on_regular_grid)

  \author Charalampos Tsoumpas
  \author Kris Thielemans

*/

START_NAMESPACE_STIR

/*!
 \brief Generic function to get the values of a 3D function on a regular grid
 \ingroup numerics
 \param[in,out] out array that will be filled with the function values. Its dimensions are used to find
   the coordinates where to sample the function (see below).
 \param[in] func function to sample
 \param[in] offset offset to use for coordinates (see below)
 \param[in] step step size to use for coordinates (see below)
 
 Symbolically, this function does the following computation for every index in the array
 \code
  out(index) = func(index * step - offset)
 \endcode 

 \par requirement for type  FunctionType
 Due to the calling sequence above, the following has to be defined
 \code
   elemT FunctionType::operator(const BasicCoordinate<3, positionT>&)
 \endcode

 \todo  At the moment, only the 3D version is implemented, but this could be templated.
*/
template <class FunctionType, class elemT, class positionT>
inline
void sample_function_on_regular_grid(Array<3,elemT>& out,
                                     FunctionType func,
                                     const BasicCoordinate<3, positionT>&  offset,  
                                     const BasicCoordinate<3, positionT>& step);

END_NAMESPACE_STIR

#include "stir/numerics/sampling_functions.inl"
