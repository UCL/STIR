//
// $Id$
//
/*!
  \file
  \ingroup buildblock

  \brief Input/output of basic vector-like types to/from streams

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2009 Hammersmith Imanet Ltd
    Copyright (C) 2013 Kris Thielemans

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
#ifndef __stir_stream_H__
#define __stir_stream_H__

#include "stir/VectorWithOffset.h"
#include "stir/BasicCoordinate.h"
#include <iostream>
#include <vector>

START_NAMESPACE_STIR

/*!
  \brief Outputs a VectorWithOffset to a stream.

  Output is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  with an endl at the end. 
  
  This can be used for higher dimensional arrays as well, where each 1D subobject 
  will be on its own line.
*/


template <typename elemT>
inline 
std::ostream& 
operator<<(std::ostream& str, const VectorWithOffset<elemT>& v);

/*!
  \brief Outputs a BasicCoordinate to a stream.

  Output is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  with no endl at the end. 
  */
template <int num_dimensions, typename coordT>
inline 
std::ostream& 
operator<<(std::ostream& str, const BasicCoordinate<num_dimensions, coordT>& v);


/*!
  \brief Outputs a vector to a stream.

  Output is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  with an endl at the end. 
  
  For each element of the vector std::ostream::operator<<() will be called.
*/
template <typename elemT>
inline 
std::ostream& 
operator<<(std::ostream& str, const std::vector<elemT>& v);

/*!
  \brief Inputs a vector from a stream.

  Input is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  
  Input is stopped when either the beginning '{', an intermediate ',' or the 
  trailing '}' is not found. The size of the vector will be the number of 
  correctly read elemT elements.
  
  For each element of the vector std::istream::operator>>(element) will be called.

  elemT needs to have a default constructor.
*/
template <typename elemT>
inline 
std::istream& 
operator>>(std::istream& str, std::vector<elemT>& v);

/*!
  \brief Inputs a VectorWithOffset from a stream.

  Input is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  As Array<>'s are constructed from nested VectorWithOffset objects, you can input
  say a 2d array as
  \verbatim
  {{1,2}, {2,4,5}, {5}}
  \endverbatim
  
  
  Input is stopped when either the beginning '{', an intermediate ',' or the 
  trailing '}' is not found.  The size of the vector will be the number of 
  correctly read elemT elements.
  
  v.get_min_index() will be 0 at the end of the call.

  For each element of the vector std::istream::operator>>(element) will be called.

  elemT needs to have a default constructor.
*/
template <typename elemT>
inline 
std::istream& 
operator>>(std::istream& str, VectorWithOffset<elemT>& v);

/*!
  \brief Inputs a coordinate from a stream.

  Input is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  
  Input is stopped when either the beginning '{', an intermediate ',' or the 
  trailing '}' is not found. If the number of correctly read elements is not \a num_dimensions,
  the last few will have undefined values.
  
  For each element of the vector std::istream::operator>>(element) will be called.

  elemT needs to have a default constructor.
*/
template <int num_dimensions, typename coordT>
inline 
std::istream& 
operator>>(std::istream& str, BasicCoordinate<num_dimensions, coordT>& v);

END_NAMESPACE_STIR

#include "stir/stream.inl"

#endif

