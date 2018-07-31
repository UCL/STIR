/*
    Copyright (C) 2000 - 2008-02-22, Hammersmith Imanet Ltd
    Copyright (C) 2013, Kris Thielemans
    Copyright (C) 2013, University College London
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
*/

#include "stir/TimeFrameDefinitions.h"
#include "stir/ExamInfo.h"
#include "stir/IO/FileSignature.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/stir_ecat6.h"
#include "stir/IO/stir_ecat7.h"
#endif
#include "stir/IO/InterfileHeader.h"
#include "stir/IO/interfile.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/format.hpp>
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/info.h"

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::pair;
using std::vector;
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
  return frame_times.at(frame_num-1).first;
}

double
TimeFrameDefinitions::
get_end_time(unsigned int frame_num) const
{
  assert(frame_num>=1);
  assert(frame_num<=get_num_frames());
  return frame_times.at(frame_num-1).second;
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
  return static_cast<unsigned int>(frame_times.size());
}

unsigned int
TimeFrameDefinitions::
get_num_time_frames() const
{
  return static_cast<unsigned int>(frame_times.size());
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
  const FileSignature file_signature(filename);
  const char * signature = file_signature.get_signature();

#ifdef HAVE_LLN_MATRIX
  if (ecat::ecat6::is_ECAT6_file(filename) ||
      ecat::ecat7::is_ECAT7_file(filename))
    {
      shared_ptr<ExamInfo> exam_info_sptr(ecat::ecat7::read_ECAT7_exam_info(filename));
      *this = exam_info_sptr->time_frame_definitions;
    }
  else
#endif
  // Interfile
  if (is_interfile_signature(signature))
  {
#ifndef NDEBUG
    info(boost::format("TimeFrameDefinitions: trying to read '%s' as Interfile") % filename);
#endif
    InterfileHeader hdr;  

    if (!hdr.parse(filename.c_str(), false)) // silent parsing
      {
	error(boost::format("Parsing of Interfile header failed for file '%s'") % filename);
      }
    *this = hdr.get_exam_info_ptr()->time_frame_definitions;
  }
  else
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

TimeFrameDefinitions::
TimeFrameDefinitions(const TimeFrameDefinitions& org_frame_defs, unsigned int frame_num)
{
  this->frame_times.push_back(make_pair(org_frame_defs.get_start_time(frame_num), org_frame_defs.get_end_time(frame_num)));
}

void
TimeFrameDefinitions::
set_time_frame(const int frame_num, const double start, const double end)
{
    assert(frame_num>=1);
    assert(frame_num<=get_num_frames());
    frame_times[frame_num-1].first = start;
    frame_times[frame_num-1].second = end;
}

END_NAMESPACE_STIR
