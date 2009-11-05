//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet
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
#ifndef __stir_extract_line_H__
#define __stir_extract_line_H__
/*!
  \file
  \ingroup buildblock
  \brief Declaration  of stir::extract_line

  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$

 */
#include "stir/Array.h"
#include "stir/BasicCoordinate.h"
START_NAMESPACE_STIR
                          
/*!
   \ingroup buildblock
   \brief  extracts a line from an array in the direction of the specified dimension.
   \todo make n-dimensional version
*/ 
template <class elemT>
Array<1,elemT>
inline
extract_line(const Array<3,elemT> &,   
             const BasicCoordinate<3,int>& index, 
             const int dimension); 
END_NAMESPACE_STIR

#include "stir/extract_line.inl"
#endif
