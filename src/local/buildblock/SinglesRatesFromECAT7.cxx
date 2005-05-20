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
get_singles_rate(const DetectionPosition<>& det_pos,
                 const double start_time,
                 const double end_time) const
{ 
  int singles_bin_index = scanner_sptr->get_singles_bin_index(det_pos);
  
  return(get_singles_rate(singles_bin_index, start_time, end_time));
}



/*
//  Generate a FramesSinglesRate - containing the average rates
//  for a frame begining at start_time and ending at end_time.
FrameSinglesRates 
SinglesRatesFromECAT7::
get_rates_for_frame(double start_time, double end_time) const {

  int start_frame;
  int end_frame;
  
  
  // Determine which frames to include in the average.
  get_frame_interval(start_time, end_time, start_frame, end_frame); 
  
  
  // Determine the number of singles units.
  int total_singles_units = scanner_sptr->get_num_singles_units();
  
  // Prepare a temporary vector.
  vector<float> average_singles_rates(total_singles_units);
  
  if ( start_frame == 0 ) {

    // Set average to 0 counts.
    for(int singles_bin = 0 ; singles_bin < total_singles_units ; ++singles_bin) {
      average_singles_rates[singles_bin] = 0;
    }
    
    FrameSinglesRates frame_rates(average_singles_rates,
                                  start_time,
                                  end_time,
                                  scanner_sptr);
    
    return(frame_rates);

  } else {
    
    for(int singles_bin = 0 ; singles_bin < total_singles_units ; ++singles_bin) {
      
      double total_rate = 0;
      
      // Loop over these frames and generate an average.
      for(int frame = start_frame; frame <= end_frame ; ++frame) {
        total_rate += singles[frame][singles_bin]; 
      }
      
      average_singles_rates[singles_bin] = 
        static_cast<float>(total_rate / (end_frame - start_frame + 1));
    }
    
    FrameSinglesRates frame_rates(average_singles_rates,
                                  time_frame_defs.get_start_time(start_frame),
                                  time_frame_defs.get_end_time(end_frame),
                                  scanner_sptr);
    
    return(frame_rates);
    
  }
  
}
*/





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



/*
void
SinglesRatesFromECAT7::
get_frame_interval(double start_time, double end_time, 
                   int& start_frame, int& end_frame) const {

  assert(end_time >=start_time);

  start_frame = 0;
  end_frame = 0;

  int num_frames = time_frame_defs.get_num_frames();
  
  if ( num_frames == 0 || end_time < time_frame_defs.get_start_time(1) ) {
    return;
  }
    
  for(int frame = 1 ; frame <= num_frames ; ++frame) {
    if ( time_frame_defs.get_start_time(frame) >= start_time ) {
      start_frame = frame;
      break;
    }
  }
  
  if ( start_frame == 0 ) {
    return;
  }
  
  // end_frame will be num_frames if a frame with a later ending than
  // end_time is not found earlier.
  for(end_frame = start_frame ; end_frame < num_frames ; ++end_frame) {
    if ( time_frame_defs.get_end_time(end_frame) >= end_time ) {
      break;
    }
  }
  
}
*/





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








float 
SinglesRatesFromECAT7::get_singles_rate(int singles_bin_index,
                                        double start_time,
                                        double end_time) const
{ 
  int axial_crystals_per_singles_unit = 
    scanner_sptr->get_num_axial_crystals_per_singles_unit();

  int transaxial_crystals_per_singles_unit =
    scanner_sptr->get_num_transaxial_crystals_per_singles_unit();

  if ( axial_crystals_per_singles_unit == 0 || transaxial_crystals_per_singles_unit == 0 ) {
    return(0.0);
  }
  
  int axial_crystals_per_block = 
    scanner_sptr->get_num_axial_crystals_per_block();

  int transaxial_crystals_per_block = 
    scanner_sptr->get_num_transaxial_crystals_per_block();
 
  
  int axial_blocks_per_singles_unit = 
    axial_crystals_per_block / axial_crystals_per_singles_unit;
 
  int transaxial_blocks_per_singles_unit = 
    transaxial_crystals_per_block / transaxial_crystals_per_singles_unit;
  

  float blocks_per_singles_unit = 
    axial_blocks_per_singles_unit * transaxial_blocks_per_singles_unit;
  
  
  int frame_num = get_frame_number(start_time, end_time);
  
  //cerr << "Frame_num:   " << frame_num << endl;
  //cerr << " Axial pos: " << axial_bucket_num << endl;
  //cerr << " Transax pos: " << transaxial_bucket_num << endl;
  
  // TODO this is really singles rate per block
  return(singles[frame_num][singles_bin_index] / blocks_per_singles_unit);
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


