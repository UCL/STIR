//
// $Id$
//

/*!
  \file 
  \ingroup buildblock
 
  \brief Implementations of non-inline functions of class DiscretisedDensity

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
#include "stir/DiscretisedDensity.h"
#include "stir/interfile.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include <typeinfo>


START_NAMESPACE_STIR

/*! 
   This function will attempt to determine the type of image in the file,
   construct an object of the appropriate type, and return a pointer to 
   the object.

   The return value is a shared_ptr, to make sure that the caller will
   delete the object.

   If more than 1 image is in the file, only the first image is read.

   Currently only VoxelsOnCartesianGrid<float> objects are supported, specified
   via an Interfile header.

   Developer's note: ideally the return value would be an auto_ptr, but it's
   very difficult to assign auto_ptrs to shared_ptrs.
*/
template<int num_dimensions, typename elemT>
DiscretisedDensity<num_dimensions,elemT> *
DiscretisedDensity<num_dimensions,elemT>::
 read_from_file(const string& filename)
{
  if (num_dimensions != 3 || typeid(elemT) != typeid(float))
    error("DiscretisedDensity::read_from_file currently only supports 3d float images\n");

  return read_interfile_image(filename);
}


/******************************
 instantiations
 *****************************/
#ifdef _MSC_VER
// disable warning on pure virtuals which are not defined
#pragma warning(disable: 4661)
#endif

template class DiscretisedDensity<3,float>;

END_NAMESPACE_STIR
