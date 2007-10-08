//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \ingroup buildblock

  \brief Implementation of class stir::TimeFrameDefinitions
 
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/

#include "stir/TimeFrameDefinitions.h"
#include "stir/IO/stir_ecat6.h"
#include "stir/IO/ecat6_utils.h"     
#include "stir/IO/stir_ecat7.h"

#include <iostream>
#include <fstream>
#include <cmath>

#ifndef STIR_NO_NAMESPACES
using std::make_pair;
using std::cerr;
using std::endl;
using std::ifstream;
#endif

START_NAMESPACE_STIR


double
TimeFrameDefinitions::
get_start_time(unsigned int frame_num) const
{
  assert(frame_num>=1);
  assert(frame_num<=get_num_frames());
  return frame_times[frame_num-1].first;
}

double
TimeFrameDefinitions::
get_end_time(unsigned int frame_num) const
{
  assert(frame_num>=1);
  assert(frame_num<=get_num_frames());
  return frame_times[frame_num-1].second;
}

double
TimeFrameDefinitions::
get_duration(unsigned int frame_num) const
{
  return get_end_time(frame_num) - get_start_time(frame_num);
}

double
TimeFrameDefinitions::
get_start_time() const
{
  return get_start_time(1);
}

double
TimeFrameDefinitions::
get_end_time() const
{
  return get_end_time(get_num_frames());
}

unsigned int
TimeFrameDefinitions::
get_num_frames() const
{
  return frame_times.size();
}

unsigned int
TimeFrameDefinitions::
get_num_time_frames() const
{
  return frame_times.size();
}

TimeFrameDefinitions::
TimeFrameDefinitions()
{}

unsigned int 
TimeFrameDefinitions::
get_time_frame_num(const double start_time, const double end_time) const
{
  assert(end_time >=start_time);
  for (unsigned int i = 1; i <=this->get_num_frames(); i++)
    {
      const double start = this->get_start_time(i);
      const double end = this->get_end_time(i);
      if (std::fabs(start-start_time)<.01 && std::fabs(end-end_time)<.01)
	{
	  return i;
	}
    }
  // not found
  return 0;
}



TimeFrameDefinitions::
TimeFrameDefinitions(const string& filename)
{
  if (ecat::ecat6::is_ECAT6_file(filename))
    read_ECAT6_frame_definitions(filename);
  else
#ifdef HAVE_LLN_MATRIX
  if (ecat::ecat7::is_ECAT7_file(filename))
    read_ECAT7_frame_definitions(filename);
  else
#endif
    read_fdef_file(filename);

#if 0
  cerr << "Frame definitions:\n{";
  for (unsigned frame_num=1; frame_num<=get_num_frames(); ++frame_num)
  {
    cerr << '{' << get_start_time(frame_num) 
         << ',' << get_end_time(frame_num) 
         << '}';
    if (frame_num<get_num_frames())
      cerr << ',';
  }
  cerr << '}' << endl;
#endif
}

void
TimeFrameDefinitions::
read_ECAT7_frame_definitions(const string& filename)
{
#ifdef HAVE_LLN_MATRIX
  USING_NAMESPACE_ECAT;
  USING_NAMESPACE_ECAT7;
  MatrixFile *mptr = matrix_open( filename.c_str(), MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
  if (!mptr) {
    matrix_perror(filename.c_str());
    exit(EXIT_FAILURE);
  }
  
  const int num_frames = std::max(static_cast<int>( mptr->mhptr->num_frames),1);
  // funnily enough, num_bed_pos seems to be offset with 1
  // (That's to say, in a singled bed study, num_bed_pos==0) 
  // TODO maybe not true for multi-bed studies
  const int num_bed_poss = static_cast<int>( mptr->mhptr->num_bed_pos) + 1;
  const int num_gates = std::max(static_cast<int>( mptr->mhptr->num_gates),1);

  // TODO
  if (num_bed_poss!=1)
    error("TimeFrameDefinitions: cannot currently handle multiple bed positions. sorry.\n");
  if (num_gates!=1)
    error("TimeFrameDefinitions: cannot currently handle multiple gates. sorry.\n");

  int min_frame_num = 1;
  int max_frame_num = num_frames;
  const int bed_num = 0;
  const int gate_num = 1;
  const int data_num = 0;
  
  for (int frame_num=min_frame_num; frame_num<=max_frame_num;++frame_num)
    {
      const int matnum = mat_numcod (frame_num, 1, gate_num, data_num, bed_num);
      MatrixData* matrix = matrix_read( mptr, matnum, MAT_SUB_HEADER);
  
      if (matrix==NULL)
	{ 
	  warning("TimeFrameDefinitions: Matrix not found at \"%d,1,%d,%d,%d\" in file \"%s\"\n.",
            frame_num, 1, gate_num, data_num, bed_num,  filename.c_str());
	  continue;
	}

      switch (mptr->mhptr->file_type)
	{
	  case PetImage: 
	case ByteVolume:
	case PetVolume:
	  {
	    Image_subheader *sheader_ptr=
	      reinterpret_cast<Image_subheader*>(matrix->shptr);
	    frame_times.push_back(make_pair(sheader_ptr->frame_start_time/1000.,
					    sheader_ptr->frame_start_time/1000. 
					    + sheader_ptr->frame_duration/1000.));
	  
	    break;
	  }
	case Byte3dSinogram:
	case Short3dSinogram:
	case Float3dSinogram :
	  {
	    Scan3D_subheader *sheader_ptr=
	      reinterpret_cast<Scan3D_subheader*>(matrix->shptr);
	    frame_times.push_back(make_pair(sheader_ptr->frame_start_time/1000.,
					    sheader_ptr->frame_start_time/1000. 
					    + sheader_ptr->frame_duration/1000.));
	  
	    break;
	  }
	case CTISinogram :
	  {
	    Scan_subheader *sheader_ptr=
	      reinterpret_cast<Scan_subheader*>(matrix->shptr);
	    frame_times.push_back(make_pair(sheader_ptr->frame_start_time/1000.,
					    sheader_ptr->frame_start_time/1000. 
					    + sheader_ptr->frame_duration/1000.));
	  
	    break;
	  }
	default:
	  error("\nTimeFrameDefinitions: supporting only image and scan file types for ECAT7 file \"%s\". Sorry.\n",
	    filename.c_str());

	}
      free_matrix_data(matrix);
    }

  matrix_close(mptr);

#endif
}

void
TimeFrameDefinitions::
read_ECAT6_frame_definitions(const string& filename)
{
  USING_NAMESPACE_ECAT;
  USING_NAMESPACE_ECAT6;

  const char * cti_name = filename.c_str();

    // open input file, read main header
  FILE* cti_fptr=fopen(cti_name, "rb"); 
  if(!cti_fptr) {
    error("\nError opening input file: %s\n",cti_name);
  }
  ECAT6_Main_header mhead;
  if(cti_read_ECAT6_Main_header(cti_fptr, &mhead)!=EXIT_SUCCESS) {
    error("\nUnable to read main header in file: %s\n",cti_name);
  }

  if (mhead.file_type != matImageFile && mhead.file_type != matScanFile)
    {
      error("\nTimeFrameDefinitions: supporting only image, scan file type for ECAT6 file \"%s\". Sorry.\n",
	    cti_name);
    }

  // funnily enough, num_bed_pos seems to be offset with 1
  // (That's to say, in a singled bed study, num_bed_pos==0) 
  // TODO maybe not true for multi-bed studies
  const int num_frames = std::max(static_cast<int>( mhead.num_frames),1);
  const int num_bed_poss = std::max(static_cast<int>( mhead.num_bed_pos) + 1,1);
  const int num_gates = std::max(static_cast<int>( mhead.num_gates),1);

  // TODO
  if (num_bed_poss!=1)
    error("TimeFrameDefinitions: cannot curently handle multiple bed positions. sorry.\n");
  if (num_gates!=1)
    error("TimeFrameDefinitions: cannot curently handle multiple gates. sorry.\n");

  int min_frame_num = 1;
  int max_frame_num = num_frames;
  const int bed_num = 0;
  const int gate_num = 1;
  const int data_num = 0;
  
  for (int frame_num=min_frame_num; frame_num<=max_frame_num;++frame_num)
    {
      MatDir entry;
      const long matnum = cti_numcod(frame_num, 1,gate_num, data_num, bed_num);    
      if(!cti_lookup(cti_fptr, matnum, &entry))  // get entry
	{
	  warning("TimeFrameDefinitions: there seems to be no frame %d in this ECAT6 file \"%s\".\n",
		  frame_num, cti_name);
	  continue;  // can't read this frame. check next one
	}    
      switch(mhead.file_type)
	{ 
	case matImageFile:
	  {
	    Image_subheader shead;
	    // read subheader
	    if(cti_read_image_subheader(cti_fptr, entry.strtblk, &shead)!=EXIT_SUCCESS)
	      { 
		error("\nTimeFrameDefinitions: Unable to look up image subheader for frame %d\n", frame_num);
	      }
	    
	    frame_times.push_back(make_pair(shead.frame_start_time/1000.,
					    shead.frame_start_time/1000. + shead.frame_duration/1000.));
	  
	    break;
	  }
	case matScanFile:
	  {     
	    Scan_subheader shead;
	    if(cti_read_scan_subheader (cti_fptr, entry.strtblk, &shead)!=EXIT_SUCCESS)
	      { 
		error("\nTimeFrameDefinitions: Unable to look up scan subheader for frame %d\n", frame_num);
	      }
	    
	    frame_times.push_back(make_pair(shead.frame_start_time/1000.,
					    shead.frame_start_time/1000. + shead.frame_duration/1000.));
	  
	    break;
	  }
	}
    }
  fclose(cti_fptr);
}    

    
void
TimeFrameDefinitions::
read_fdef_file(const string& fdef_filename)
{
  ifstream in(fdef_filename.c_str());
  if (!in)
    error("TimeFrameDefinitions: Error reading \"%s\"\n", fdef_filename.c_str());

  
  double previous_end_time = 0;
  while (true)
  {
    int num;
    double duration;
    in >> num >> duration;
    if (!in)
      break;
    // check if input is ok
    // note: allow negative 'duration' if num==0 to be able to skip in negative direction 
    // (useful for starting the first frame at negative time)
    if (num<0 || (num>0 && duration<=0))
        error("TimeFrameDefinitions: Reading frame_def file \"%s\":\n"
	      "encountered negative numbers (%d, %g)\n",
	      fdef_filename.c_str(), num, duration);

    if (num==0)
      {
	// special case to allow us to skip a time period without storing it
	previous_end_time+=duration;
      }
    while (num--)
    {
      frame_times.push_back(make_pair(previous_end_time, previous_end_time+duration));
      previous_end_time+=duration;
    }
  }
  if (this->get_num_frames()==0)
    error("TimeFrameDefinitions: Reading frame definitions file \"%s\":\n"
	  "I didn't discover any frames. Wrong file format?\n"
	  "Should be an ECAT6, ECAT7 file or a text file with something like\n\n"
	  "3 50.5\n1 10\n0 3\n1 9\n\n"
	  "for 3 frames of 50.5 secs, 1 frame of 10 secs, a gap of 3 secs, 1 frame of 9 secs.",
	  fdef_filename.c_str());
}

TimeFrameDefinitions::
TimeFrameDefinitions(const vector<pair<double, double> >& frame_times)
  : frame_times(frame_times)
{
  if (get_num_frames()==0)
    return;

  // check times are in sequence
  double current_time = get_start_time(1);
  for (unsigned int current_frame = 1; current_frame <= get_num_frames(); ++ current_frame)
    {
      if (current_time > get_start_time(current_frame) + .001) // add .001 to avoid numerical errors
	error("TimeFrameDefinitions: frame number %d start_time (%g) is smaller than "
	      "previous end_time (%g)\n",
	      current_frame, get_start_time(current_frame), current_time);
      if (get_start_time(current_frame) > get_end_time(current_frame) + .01) // add .01 to avoid numerical errors
	error("TimeFrameDefinitions: frame number %d start_time (%g) is larger than "
	      "end_time (%g)\n",
	      current_frame, get_start_time(current_frame), get_end_time(current_frame));
      current_time = get_end_time(current_frame);
    }
}

TimeFrameDefinitions::
TimeFrameDefinitions(const vector<double>& start_times, 
		     const vector<double>& durations)
{
  if (start_times.size() != durations.size())
    error("TimeFrameDefinitions: constructed with start_times "
	  "and durations of different length");

  this->frame_times.resize(start_times.size());

  for (unsigned int current_frame = 1; 
       current_frame <= this->get_num_frames(); 
       ++ current_frame)
    {
      frame_times[current_frame-1].first = 
	start_times[current_frame-1];
      frame_times[current_frame-1].second = 
	start_times[current_frame-1] + durations[current_frame-1];
    }
}

END_NAMESPACE_STIR
