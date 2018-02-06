
/*
    Copyright (C) 2003- 2009, Hammersmith Imanet Ltd
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

#ifndef __stir_assign_to_subregion_H__
#define __stir_assign_to_subregion_H__

/*!
  \file 
  \ingroup Array 
  \brief declares the stir::assign_to_subregion function

  \author Kris Thielemans

*/
#include "stir/Array.h"

START_NAMESPACE_STIR

/*!  
   \ingroup Array
   \brief assign a value to a sub-region of an array

   sets all values for indices between \a mask_location - \a half_size and \a mask_location + \a half_size to \a value,
   taking care of staying inside the index-range of the array.
*/
template <class elemT>   
inline void 
assign_to_subregion(Array<3,elemT>& input_array, 
                    const BasicCoordinate<3,int>& mask_location,
                    const BasicCoordinate<3,int>& half_size,
                    const elemT& value);
END_NAMESPACE_STIR

#include "stir/assign_to_subregion.inl"

#endif


