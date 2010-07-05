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
/*!
  \file
  \ingroup data_buildblock
  \brief Implementation of class stir::DynamicProjData
  \author Kris Thielemans
  \author Charalampos Tsoumpas
  
  $Date$
  $Revision$
*/

#include "stir/DynamicProjData.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/ProjDataFromStream.h"
#include <iostream>
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/round.h"
#include <fstream>

START_NAMESPACE_STIR

const double
DynamicProjData::
get_start_time_in_secs_since_1970() const
{ return this->_start_time_in_secs_since_1970; }

DynamicProjData*
DynamicProjData::
read_from_file(const string& filename) // The written projection data is read in respect to its center as origin!!!
{
  const int max_length=300;
  char signature[max_length];

  // read signature
  {
    std::fstream input(filename.c_str(), std::ios::in | std::ios::binary);
    if (!input)
      error("DynamicProjData::read_from_file: error opening file %s\n", filename.c_str());
    input.read(signature, max_length);
    signature[max_length-1]='\0';
  }

#ifdef HAVE_LLN_MATRIX
  if (strncmp(signature, "MATRIX", 6) == 0)
  {
#ifndef NDEBUG
    warning("DynamicProjData::read_from_file trying to read %s as ECAT7\n", filename.c_str());
#endif
    USING_NAMESPACE_ECAT
    USING_NAMESPACE_ECAT7

    if (is_ECAT7_emission_file(filename))
    {
      Main_header mhead;
      if (read_ECAT7_main_header(mhead, filename) == Succeeded::no)
        {
          warning("DynamicProjData::read_from_file cannot read %s as ECAT7\n", filename.c_str());
          return 0;
        }
      DynamicProjData * dynamic_proj_data_ptr = new DynamicProjData;
      dynamic_proj_data_ptr->_time_frame_definitions =
        TimeFrameDefinitions(filename);      

      dynamic_proj_data_ptr->_scanner_sptr =
        find_scanner_from_ECAT_system_type(mhead.system_type);

      dynamic_proj_data_ptr->_start_time_in_secs_since_1970 =
        static_cast<double>(mhead.scan_start_time);

      const unsigned int num_frames =
        static_cast<unsigned int>(mhead.num_frames);
      dynamic_proj_data_ptr->_proj_datas.resize(num_frames); 

      for (unsigned int frame_num=1; frame_num <= num_frames; ++ frame_num)
        {
          dynamic_proj_data_ptr->_proj_datas[frame_num-1] =
            ECAT7_to_PDFS(filename,
                          frame_num, 
                          /*gate*/1,
                          /*  data_num, bed_num, */ 0,0);
        }
      if (is_null_ptr(dynamic_proj_data_ptr->_proj_datas[0]))
              error("DynamicProjData: No frame available\n");
    }
    else
    {
      if (is_ECAT7_file(filename))
        warning("DynamicProjData::read_from_file ECAT7 file %s should be projection data.\n", filename.c_str());
    }
  }
  else 
    warning("DynamicProjData::read_from_file %s seems to correspond to ECAT6 projection data. I cannot read this\n");
#endif // end of HAVE_LLN_MATRIX
    // }    
  
  // return a zero pointer if we get here
  return 0;
}

Succeeded 
DynamicProjData::
write_to_ecat7(const string& filename) const 
{
#ifndef HAVE_LLN_MATRIX
  return Succeeded::no;
#else

  Main_header mhead;
  ecat::ecat7::make_ECAT7_main_header(mhead, filename, 
                                      *get_proj_data(1).get_proj_data_info_ptr() );
  mhead.num_frames = this->get_num_frames();
  mhead.acquisition_type =
    mhead.num_frames>1 ? DynamicEmission : StaticEmission;

  round_to(mhead.scan_start_time, floor(this->get_start_time_in_secs_since_1970()));
    
  MatrixFile* mptr= matrix_create (filename.c_str(), MAT_CREATE, &mhead);
  if (mptr == 0)
    {
      warning("DynamicProjData::write_to_ecat7 cannot write output file %s\n", filename.c_str());
      return Succeeded::no;
    }
  for (  unsigned int frame_num = 1 ; frame_num<=this->get_num_frames() ;  ++frame_num ) 
    {
      if (ecat::ecat7::ProjData_to_ECAT7(mptr,
                                         get_proj_data(frame_num),
                                         frame_num)
          == Succeeded::no)
      {
        matrix_close(mptr);
        return Succeeded::no;
      }
    }
  matrix_close(mptr);
  return Succeeded::yes;
#endif // end of HAVE_LLN_MATRIX
}

END_NAMESPACE_STIR
