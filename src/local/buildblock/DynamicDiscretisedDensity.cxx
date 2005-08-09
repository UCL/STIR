//
// $Id$
//
/*!
  \file
  \ingroup utilities
  \brief Implementation of class stir::DynamicDiscretisedDensity
  \author Kris Thielemans
  \author Charalampos Tsoumpas
  
  $Date$
  $Revision$
*/
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

#include "local/stir/DynamicDiscretisedDensity.h"
#include "stir/IO/stir_ecat7.h"
#include <iostream>
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

DynamicDiscretisedDensity*
DynamicDiscretisedDensity::read_from_file(const string& filename)
{
  const int max_length=300;
  char signature[max_length];

  // read signature
  {
    fstream input(filename.c_str(), ios::in | ios::binary);
    if (!input)
      error("DynamicDiscretisedDensity::read_from_file: error opening file %s\n", filename.c_str());
    
    input.read(signature, max_length);
    signature[max_length-1]='\0';
  }
 
  DynamicDiscretisedDensity * dynamic_image_ptr =
    new DynamicDiscretisedDensity;

#ifdef HAVE_LLN_MATRIX
  if (strncmp(signature, "MATRIX", 6) == 0)
  {
#ifndef NDEBUG
    warning("DynamicDiscretisedDensity::read_from_file trying to read %s as ECAT7\n", filename.c_str());
#endif
    USING_NAMESPACE_ECAT
    USING_NAMESPACE_ECAT7

    if (is_ECAT7_image_file(filename))
    {
      Main_header mhead;
      if (read_ECAT7_main_header(mhead, filename) == Succeeded::no)
	{
	  warning("DynamicDiscretisedDensity::read_from_file cannot read %s as ECAT7\n", filename.c_str());
	  return 0;
	}
      dynamic_image_ptr->_scanner_sptr =
	find_scanner_from_ECAT_system_type(mhead.system_type);

      dynamic_image_ptr->_time_frame_definitions =
        TimeFrameDefinitions(filename);      

      for (unsigned int frame_num=1; frame_num <= _time_frame_definitions.get_num_time_frames(); ++ frame_num)
	{
	   dynamic_image_ptr->_densities[frame_num] =
	    ECAT7_to_VoxelsOnCartesianGrid(filename,
					   frame_num, /*gate_num, data_num, bed_num*/1,0,0);
	  if (!is_null_ptr(densities[frame_num])
	      error();
	}
    }
    else
    {
      if (is_ECAT7_file(filename))
	warning("DynamicDiscretisedDensity::read_from_file ECAT7 file %s should be an image\n", filename.c_str());
    }
  }
#endif // end of HAVE_LLN_MATRIX
  }
  return dynamic_image_ptr;
}


Succeeded 
DynamicDiscretisedDensity::write_to_ecat7(const string&filename) const
{
#ifndef HAVE_LLN_MATRIX
  return Succeeded::no;
#else

  Main_header mhead;
  make_ECAT7_main_header(mhead, _scanner_sptr, filename, this->get_density(1));
  mhead.num_frames = this->get_num_time_frames();
  mhead.acquisition_type =
    mhead.num_frames>1 ? DynamicEmission : StaticEmission;

  MatrixFile* mptr= matrix_create (filename.c_str(), MAT_CREATE, &mhead);
  if (mptr == 0)
    {
      warning("DynamicDiscretisedDensity::write_to_ecat7 cannot write output file %s\n", cti_name);
      return Succeeded::no;
    }
  for (  unsigned int frame_num = 1 ; frame_num<=this->get_num_time_frames() ;  ++frame_num )
    {
      if (DiscretisedDensity_to_ECAT7(mptr,
                                      this->get_density(frame_num),
                                      frame_num)
                                      == Succeeded::no)
      {
        matrix_close(mptr);
        return Succeeded::no;
      }
    }
  matrix_close(mptr);
  return Succeeded::yes;
#endif // HAVE_LLN_MATRIX
}

END_NAMESPACE_STIR
