//
// $Id$
//
#ifndef __interpolate_H__
#define __interpolate_H__
/*!
  \file 
  \ingroup buildblock
 
  \brief declares functions for interpolation

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/VectorWithOffset.h"

START_NAMESPACE_STIR
/*!
  \ingroup buildblock
 
  \brief 'overlap' interpolation (i.e. count preserving) for vectors.
*/

template <typename T>
void
overlap_interpolate(VectorWithOffset<T>& out_data, 
		    const VectorWithOffset<T>& in_data,
		    const float zoom, 
		    const float offset, 
		    const bool assign_rest_with_zeroes = true);

END_NAMESPACE_STIR

#endif
