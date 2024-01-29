//
//

/*!
  \file 
  \ingroup densitydata
 
  \brief Implementations of non-inline functions of class stir::DiscretisedDensity

  \author Kris Thielemans 
  \author Ashley Gillman
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
    Copyright (C) 2018, CSIRO
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
#include "stir/DiscretisedDensity.h"
#if 1
#include "stir/IO/read_from_file.h"
#else
#include "stir/IO/interfile.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/ecat6_utils.h"
#include "stir/IO/stir_ecat6.h"
#include "stir/IO/stir_ecat7.h"
#endif
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/is_null_ptr.h"
#endif

#include <typeinfo>
#include <fstream>
#include "stir/warning.h"
#include "stir/error.h"

using std::fstream;
using std::string;

START_NAMESPACE_STIR

#ifdef HAVE_LLN_MATRIX
USING_NAMESPACE_ECAT
USING_NAMESPACE_ECAT7
USING_NAMESPACE_ECAT6
#endif

/*! 
  \deprecated
   This function just calls stir::read_from_file.
*/
template<int num_dimensions, typename elemT>
DiscretisedDensity<num_dimensions,elemT> *
DiscretisedDensity<num_dimensions,elemT>::
 read_from_file(const string& filename)
{

#if 1
  unique_ptr<DiscretisedDensity<num_dimensions,elemT> > density_aptr
    (stir::read_from_file<DiscretisedDensity<num_dimensions,elemT> >(filename));
  return density_aptr.release();

#else
  if (num_dimensions != 3 || typeid(elemT) != typeid(float))
    error("DiscretisedDensity::read_from_file currently only supports 3d float images\n");
  const int max_length=300;
  char signature[max_length];

  // read signature
  {
    fstream input(filename.c_str(), ios::in | ios::binary);
    if (!input)
      error("DiscretisedDensity::read_from_file: error opening file %s\n", filename.c_str());
    
    input.read(signature, max_length);
    signature[max_length-1]='\0';
  }
  // Interfile
  if (is_interfile_signature(signature))
  {
#ifndef NDEBUG
    warning("DiscretisedDensity::read_from_file trying to read %s as Interfile\n", 
	    filename.c_str());
#endif
    DiscretisedDensity<num_dimensions,elemT> * density_ptr =
      read_interfile_image(filename);
    if (!is_null_ptr(density_ptr))
      return density_ptr;
  }


    
#ifdef HAVE_LLN_MATRIX
  if (strncmp(signature, "MATRIX", 6) == 0)
  {
#ifndef NDEBUG
    warning("DiscretisedDensity::read_from_file trying to read %s as ECAT7\n", filename.c_str());
#endif
    USING_NAMESPACE_ECAT
    USING_NAMESPACE_ECAT7

    if (is_ECAT7_image_file(filename))
    {
      warning("\nReading frame 1, gate 1, data 0, bed 0 from file %s\n",
	      filename.c_str());
      DiscretisedDensity<num_dimensions,elemT> * density_ptr =
	ECAT7_to_VoxelsOnCartesianGrid(filename,
				       /*frame_num, gate_num, data_num, bed_num*/1,1,0,0);
      if (!is_null_ptr(density_ptr))
	return density_ptr;
    }
    else
    {
      if (is_ECAT7_file(filename))
	warning("DiscretisedDensity::read_from_file ECAT7 file %s is of unsupported file type\n", filename.c_str());
    }

  }
#endif // HAVE_LLN_MATRIX


#ifdef HAVE_LLN_MATRIX
  {
    // Try ECAT6
    // ECAT6  does not have a signature
#ifndef NDEBUG
    warning("DiscretisedDensity::read_from_file trying to read %s as ECAT6\n", filename.c_str());
#endif
    USING_NAMESPACE_ECAT;
    USING_NAMESPACE_ECAT6;
      if (is_ECAT6_image_file(filename))
      {
        ECAT6_Main_header mhead;
        FILE * cti_fptr=fopen(filename.c_str(), "rb"); 
        if(cti_read_ECAT6_Main_header(cti_fptr, &mhead)!=EXIT_SUCCESS) 
	  {
	    if (cti_fptr!=NULL)
	      fclose(cti_fptr);
	    error ("error reading main header in ECAT 6 file %s\n", filename.c_str());
	  }
        
	warning("\nReading frame 1, gate 1, data 0, bed 0 from file %s\n",
		filename.c_str());
        VoxelsOnCartesianGrid<float> * tmp =
          ECAT6_to_VoxelsOnCartesianGrid(/*frame_num, gate_num, data_num, bed_num*/1,1,0,0,
          cti_fptr, mhead);
	fclose(cti_fptr);
	return tmp;
      }
  }
#endif // HAVE_LLN_MATRIX


  error("DiscretisedDensity::read_from_file: %s seems to be in an unsupported file format\n",
	filename.c_str());
  return 0;
#endif

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
