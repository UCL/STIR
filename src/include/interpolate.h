//
// $Id$: $Date$
//
#ifndef __interpolate_H__
#define __interpolate_H__
/*!
  \file 
  \ingroup buildblock
 
  \brief declares functions for interpolation

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/


#include "VectorWithOffset.h"

START_NAMESPACE_TOMO
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

END_NAMESPACE_TOMO

#endif
