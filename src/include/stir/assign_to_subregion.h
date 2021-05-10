
/*
    Copyright (C) 2003- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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


