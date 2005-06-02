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
  \ingroup singles_buildblock
  \brief Implementation of stir::ecat::ecat7::SinglesRatesFromECAT7

  \author Kris Thielemans
  \author Sanida Mustafovic
  $Date$
  $Revision$
*/

#include "stir/data/SinglesRatesFromECAT7.h"
#include "stir/DetectionPosition.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/IndexRange2D.h"
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





float 
SinglesRatesFromECAT7::
get_singles_rate(int singles_bin_index, int frame_number) const
{ 
  // Return singles rate for the singles unit.
  return(singles[frame_number][singles_bin_index]);
}





float
SinglesRatesFromECAT7::get_singles_rate(int singles_bin_index,
                                        double start_time,
                                        double end_time) const
{
  int frame_number = get_frame_number(start_time, end_time);
  return(get_singles_rate(singles_bin_index, frame_number));
}





float 
SinglesRatesFromECAT7::
get_singles_rate(const DetectionPosition<>& det_pos,
                 const double start_time,
                 const double end_time) const
{ 
  int singles_bin_index = scanner_sptr->get_singles_bin_index(det_pos);
  
  return(get_singles_rate(singles_bin_index, start_time, end_time));
}






int 
SinglesRatesFromECAT7::
get_frame_number(const double start_time, const double end_time) const
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




int 
SinglesRatesFromECAT7::get_num_frames() const {
  return(time_frame_defs.get_num_frames());
}



double 
SinglesRatesFromECAT7::get_frame_start(unsigned int frame_number) const {
  if ( frame_number < 1 || frame_number > time_frame_defs.get_num_frames() ) {
    return(0.0);
  } else {
    return(time_frame_defs.get_start_time(frame_number));
  }
}
 


double 
SinglesRatesFromECAT7::get_frame_end(unsigned int frame_number) const {
  if ( frame_number < 1 || frame_number > time_frame_defs.get_num_frames() ) {
    return(0.0);
  } else {
    return(time_frame_defs.get_end_time(frame_number));
  }
}









int
SinglesRatesFromECAT7::read_singles_from_file(const string& ECAT7_filename,
                                              const ios::openmode open_mode)

{
  
  int num_frames = 0;

#ifndef HAVE_LLN_MATRIX

  error("Compiled without ECAT7 support\n");

#else

  MatrixFile* mptr = matrix_open(ECAT7_filename.c_str(), MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);

  if (mptr==0) {
    error("Error opening '%s' as ECAT7\n", ECAT7_filename.c_str());
  }

  if (!(mptr->mhptr->file_type == Byte3dSinogram ||
        mptr->mhptr->file_type == Short3dSinogram || 
        mptr->mhptr->file_type == Float3dSinogram)) {
    error("SinglesRatesFromECAT7: filename %s should be an ECAT7 emission file\n",
          ECAT7_filename.c_str());
  }
  
  scanner_sptr = find_scanner_from_ECAT_system_type(mptr->mhptr->system_type);


  if (scanner_sptr->get_type() != Scanner::E966) {
    warning("check SinglesRatesFromECAT7 for non-966\n");
  }


  Main_header* main_header = reinterpret_cast<Main_header*>( mptr->mhptr ) ;

  num_frames = main_header->num_frames;

  // Get total number of bins for this type of scanner.
  const int total_singles_units = scanner_sptr->get_num_singles_units();

  if ( total_singles_units > 0 ) {
    // Create the main array of data.
    singles = Array<2,float>(IndexRange2D(1, main_header->num_frames,
                                          0, total_singles_units - 1));
  }

  
  MatrixData* matrix;
  vector<pair<double, double> > time_frames(main_header->num_frames);


  for ( int mat_frame = 1 ; mat_frame <= num_frames ; mat_frame++ )
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

    // The order of the singles units in the sub header is the same as required
    // by the main singles array. This may not be the case for other file formats.
    for (int singles_bin = 0 ; singles_bin < total_singles_units ; ++singles_bin) {
      singles[mat_frame][singles_bin] = *(singles_ptr + singles_bin);
    }
    
  }

  time_frame_defs = TimeFrameDefinitions(time_frames);
#endif

  return(num_frames); 
  
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






END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR


