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
#ifndef __stir_more_algorithms_H__
#define __stir_more_algorithms_H__
/*!
  \file
  \ingroup buildblock
  
  \brief Declaration of some functions missing from std::algorithm
    
  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/common.h"

START_NAMESPACE_STIR

/*! \ingroup buildblock
  \brief Like std::max_element, but comparing after taking absolute value

  This function using stir::norm_squared(), so works for complex numbers as well.
*/
template <class iterT> 
inline
iterT abs_max_element(iterT start, iterT end);
END_NAMESPACE_STIR

#include "stir/more_algorithms.inl"
#endif
