//
//
#ifndef __stir_IO_stir_AVW__H__
#define __stir_IO_stir_AVW__H__
/*
    Copyright (C) 2001- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
\file
\ingroup IO
\brief routines to convert AVW data structures to STIR
\author Kris Thielemans 

*/


#ifdef HAVE_AVW

#include "AVW.h"
#include "stir/common.h"

START_NAMESPACE_STIR

template <class elemT> class VoxelsOnCartesianGrid;

namespace AVW
{

  //! A routine that converts AVW volume to ::stir::VoxelsOnCartesianGrid
  /*! Will return a null pointer if the convertion failed for some reason.
   */
  VoxelsOnCartesianGrid<float> *
    AVW_Volume_to_VoxelsOnCartesianGrid(AVW_Volume const* const avw_volume,
					const bool flip_z = false);

} // end namespace AVW

END_NAMESPACE_STIR

#endif // HAVE_AVW

#endif //__stir_IO_stir_AVW__H__
