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
#include "stir/IO/ecat6_utils.h"
#include "stir/IO/stir_ecat6.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/is_null_ptr.h"
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

#ifndef NDEBUG
  warning("DiscretisedDensity::read_from_file trying to read %s as Interfile\n", filename.c_str());
#endif
  DiscretisedDensity<num_dimensions,elemT> * density_ptr =
    read_interfile_image(filename);
  if (!is_null_ptr(density_ptr))
    return density_ptr;


  {
#ifndef NDEBUG
    warning("DiscretisedDensity::read_from_file trying to read %s as ECAT6\n", filename.c_str());
#endif
    USING_NAMESPACE_ECAT6;
      if (is_ecat6_image_file(filename))
      {
        ECAT6_Main_header mhead;
        FILE * cti_fptr=fopen(filename.c_str(), "rb"); 
        if(cti_read_ECAT6_Main_header(cti_fptr, &mhead)!=EXIT_SUCCESS) 
          error ("error reading main header in ECAT 6 file %s\n", filename.c_str());
        
        return
          ECAT6_to_VoxelsOnCartesianGrid(/*frame_num, gate_num, data_num, bed_num*/1,1,0,0,
          cti_fptr, mhead);
      }
  }

#ifdef HAVE_LLN_MATRIX
  {
#ifndef NDEBUG
    warning("DiscretisedDensity::read_from_file trying to read %s as ECAT7\n", filename.c_str());
#endif
    USING_NAMESPACE_ECAT7;

    if (is_ecat7_image_file(filename))
    {
      string interfile_header_name;
      if (write_basic_interfile_header_for_ecat7(interfile_header_name, filename, 1,1,0,0) ==
        Succeeded::no)
        return 0;
#ifndef NDEBUG
      warning("DiscretisedDensity::read_from_file wrote interfile header %s\nNow reading as interfile", 
        interfile_header_name.c_str());
#endif
      return 
        read_interfile_image(interfile_header_name);
    }
  }
#endif // HAVE_LLN_MATRIX


  error("DiscretisedDensity::read_from_file: %s seems to be in an unsupported file format\n",
	filename.c_str());
  return 0;

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
