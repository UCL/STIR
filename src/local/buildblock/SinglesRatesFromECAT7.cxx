//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
  \brief Implementation of SinglesRatesFromECAT7

  \author Kris Thielemans
  \author Sanida Mustafovic
  $Date$
  $Revision$
*/

#include "local/stir/SinglesRateFromECAT7.h"
#include "stir/DetectionPosition.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/IndexRange3D.h"
#ifdef HAVE_LLN_MATRIX
#include "ecat_model.h"
#endif

#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::streampos;
#endif

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

const char * const 
SinglesRatesFromECAT7::registered_name = "Singles From ECAT7"; 

SinglesRatesFromECAT7::
SinglesRatesFromECAT7()
{}

Array<3,float> 
SinglesRatesFromECAT7::read_singles_from_file(const string& ECAT7_filename,
		   const ios::openmode open_mode)

{
#ifndef HAVE_LLN_MATRIX
  error("Compiled without ECAT7 support\n");
#else
  MatrixFile* mptr = matrix_open(ECAT7_filename.c_str(), MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
  if (mptr==0)
    error("Error opening '%s' as ECAT7\n", ECAT7_filename.c_str());
  if (!(mptr->mhptr->file_type == Byte3dSinogram ||
    mptr->mhptr->file_type == Short3dSinogram || mptr->mhptr->file_type == Float3dSinogram))
    error("SinglesRatesFromECAT7: filename %s should be an ECAT7 emission file\n",
       ECAT7_filename.c_str());
  scanner_sptr = find_scanner_from_ECAT_system_type(mptr->mhptr->system_type);

  if (scanner_sptr->get_type() != Scanner::E966)
    warning("check SinglesRatesFromECAT7 for non-966\n");
  trans_blocks_per_bucket =scanner_sptr->get_num_transaxial_blocks_per_bucket();
  angular_crystals_per_block =scanner_sptr->get_num_transaxial_crystals_per_block();
  axial_crystals_per_block =scanner_sptr->get_num_axial_crystals_per_block();
  //TODO move to Scanner
  if (scanner_sptr->get_type() == Scanner::E966)
    num_axial_blocks_per_singles_unit = 2;
  else
    num_axial_blocks_per_singles_unit = 1;

  Main_header* main_header = 
    reinterpret_cast<Main_header*>( mptr->mhptr ) ;

  singles =  Array<3,float>(IndexRange3D(1,main_header->num_frames,
					 0,scanner_sptr->get_num_axial_blocks()/num_axial_blocks_per_singles_unit-1,
					 0,scanner_sptr->get_num_transaxial_buckets()-1)); 
  
  MatrixData* matrix ;
  vector<pair<double, double> > time_frames(main_header->num_frames);
  for ( int mat_frame = 1 ; mat_frame <= main_header->num_frames ; mat_frame++ )
  {
    //cerr << "Reading frame " << mat_frame <<endl;
    matrix= matrix_read( mptr, mat_numcod( mat_frame, 1, 1, 0, 0),
			 /*don't read the data*/MAT_SUB_HEADER) ;
    
    Scan3D_subheader* scan_subheader_ptr=  
      reinterpret_cast<Scan3D_subheader *>(matrix->shptr);
    time_frames[mat_frame-1].first=scan_subheader_ptr->frame_start_time/1000.;
    time_frames[mat_frame-1].second=
      time_frames[mat_frame-1].first +
      scan_subheader_ptr->frame_duration/1000.;

    float const* singles_ptr = reinterpret_cast<float const *>(scan_subheader_ptr->uncor_singles);//matrix->data_ptr);
    for(Array<2,float>::full_iterator iter = singles[mat_frame].begin_all(); iter != singles[mat_frame].end_all();)
    {
      *iter++ = *singles_ptr++;
    }
  }
  time_frame_defs =
    TimeFrameDefinitions(time_frames);
#endif
  return singles; 
  
}

float 
SinglesRatesFromECAT7:: get_singles_rate(const DetectionPosition<>& det_pos,
					 const double start_time,
					 const double end_time) const
{ 
  const int denom = trans_blocks_per_bucket*angular_crystals_per_block;
  const int axial_pos = det_pos.axial_coord();
  const int transaxial_pos = det_pos.tangential_coord();
  const int axial_bucket_num = 
    axial_pos/(num_axial_blocks_per_singles_unit*axial_crystals_per_block);
  const int transaxial_bucket_num = (transaxial_pos/denom) ;
  const float blocks_per_singles_unit =
    num_axial_blocks_per_singles_unit*trans_blocks_per_bucket;

  int frame_num = get_frame_number(start_time,end_time);
  //cerr << "Frame_num:   " << frame_num << endl;
  //cerr << " Axial pos: " << axial_bucket_num << endl;
  //cerr << " Transax pos: " << transaxial_bucket_num << endl;
  
  // TODO this is really singles rate per block
  return singles[frame_num][axial_bucket_num][transaxial_bucket_num]/blocks_per_singles_unit;
}


void 
SinglesRatesFromECAT7::
initialise_keymap()
{
  parser.add_start_key("Singles Rates From ECAT7");
  parser.add_key("ECAT7_filename", &ECAT7_filename);
  parser.add_stop_key("End Singles Rates From ECAT7");
}

bool 
SinglesRatesFromECAT7::
post_processing()
{
  read_singles_from_file(ECAT7_filename);
  return false;
}


void 
SinglesRatesFromECAT7::set_defaults()
{
  ECAT7_filename = "";
}

int 
SinglesRatesFromECAT7::get_frame_number (const double start_time, const double end_time) const
{
  assert(end_time >=start_time);
  //cerr << "num frames are" << time_frame_defs.get_num_frames()<<endl;;
  for (unsigned int i = 1; i <=time_frame_defs.get_num_frames(); i++)
    {
      const double start = time_frame_defs.get_start_time(i);
      const double end = time_frame_defs.get_end_time(i);
      //cerr << "Start frame" << start <<endl;
      //cerr << " End frame " << end << endl;
       if (start<=start_time+.001 && end>=end_time-.001)
	{
	  return static_cast<int>(i);
	}
    }
      
  error("SinglesRatesFromECAT7::get_frame_number didn't find a frame for time interval (%g,%g)\n",
	start_time, end_time);
  return 0; // to satisfy compilers
  

}


END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR


