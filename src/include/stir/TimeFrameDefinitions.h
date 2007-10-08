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
  \ingroup buildblock  
  \brief Declaration of class stir::TimeFrameDefinitions
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
#ifndef __stir_TimeFrameDefinitions_H__
#define __stir_TimeFrameDefinitions_H__

#include "stir/common.h"
#include <string>
#include <vector>
#include <utility>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::pair;
using std::vector;
#endif

START_NAMESPACE_STIR
/*!
  \ingroup buildblock
  \brief Class used for storing time frame durations

  Times are supposed to be relative to the scan start time.

  Currently this class can read frame info from an ECAT6, ECAT7 and a 'frame definition'
  file. See the documentation for the constructor.

  Will probably be superseded by Study classes.
*/
class TimeFrameDefinitions
{
public:
  //! Default constructor: no time frames at all
  TimeFrameDefinitions();

  //! Read the frame definitions from a file
  /*! 
   The filename can point to an ECAT6 file, and ECAT7 file (if you
   have installed the LLN library), or a simple ASCII text file.

   This latter uses the '.fdef' format used by Peter Bloomfield's software.
   The format is a number of lines, each existing of 2 numbers
  \verbatim
    num_frames_of_this_duration   duration_in_secs
  \endverbatim
  This duration is a double number.

  This class in fact allows an extension of the above. Setting 
  \a num_frames_of_this_duration to 0 allows skipping
  a time period of the corresponding \a duration_in_secs.
  */
  explicit TimeFrameDefinitions(const string& fdef_filename);
  
  //! Construct from a list of time frames
  /*! Times have to be in increasing order*/
  TimeFrameDefinitions(const vector<pair<double, double> >&);

  //! Construct from a list of start times and durations
  /*! start times have to be in increasing order*/
  TimeFrameDefinitions(const vector<double>& start_times, const vector<double>& durations);

  //! \name get info for 1 frame (frame_num is 1 based)
  //@{
  double get_start_time(unsigned int frame_num) const;
  double get_end_time(unsigned int frame_num) const;
  double get_duration(unsigned int frame_num) const;
  //@}

  //! Get start time of first frame
  double get_start_time() const;
  //! Get end time of last frame
  double get_end_time() const;

  //! Get number of frames
  unsigned int get_num_frames() const;
  //! Get number of frames
  unsigned int get_num_time_frames() const;
  
  //! Get the frame number associated with a frame starting and start_time and ending at end_time.
  /*! \return frame number (between 1 and get_num_time_frames()) or 0 if frame not found.
   */
    unsigned int get_time_frame_num(const double start_time, const double end_time) const;

private:
  //! Stores start and end time for each frame
  vector<pair<double, double> > frame_times;

  void  read_ECAT6_frame_definitions(const string& filename);
  void  read_ECAT7_frame_definitions(const string& filename);
  void  read_fdef_file(const string& filename);

};

END_NAMESPACE_STIR
#endif
