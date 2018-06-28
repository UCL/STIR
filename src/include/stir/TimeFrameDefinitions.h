/*
    Copyright (C) 2003 - 2007-10-08, Hammersmith Imanet Ltd
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
  \brief Declaration of class stir::TimeFrameDefinitions
    
  \author Kris Thielemans
*/
#ifndef __stir_TimeFrameDefinitions_H__
#define __stir_TimeFrameDefinitions_H__

#include "stir/common.h"
#include <string>
#include <vector>
#include <utility>

START_NAMESPACE_STIR
/*!
  \ingroup buildblock
  \brief Class used for storing time frame durations

  All times are in seconds as per standard STIR conventions.

  Times are supposed to be relative to the exam start time.

  Currently this class can read frame info from an ECAT6, ECAT7 and a 'frame definition'
  file. See the documentation for the constructor.
*/
class TimeFrameDefinitions
{
public:
  //! Default constructor: no time frames at all
  TimeFrameDefinitions();

  //! Read the frame definitions from a file
  /*! 
   \deprecated
   The filename can point to an ECAT6 file, and ECAT7 file (if you
   have installed the LLN library), an Interfile file, or a simple ASCII text file.

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
  explicit TimeFrameDefinitions(const std::string& fdef_filename);
  
  //! Construct from a list of time frames
  /*! Each frame is specified as a std::pair with start and end time (in seconds).
      Start times have to be in increasing order*/
  TimeFrameDefinitions(const std::vector<std::pair<double, double> >&);

  //! Construct from a list of start times and durations
  /*! start times have to be in increasing order*/
  TimeFrameDefinitions(const std::vector<double>& start_times, const std::vector<double>& durations);

  //! Construct from a single time frame of an existing object 
  TimeFrameDefinitions(const TimeFrameDefinitions&, unsigned int frame_num);

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

    //! Set number of time frames
    void set_num_time_frames(int num_time_frames) { frame_times.resize(num_time_frames); }

    //! Set time frame
    void set_time_frame(int frame, double start, double duration) { frame_times[frame].first = start; frame_times[frame].second = duration; }

private:
  //! Stores start and end time for each frame
  std::vector<std::pair<double, double> > frame_times;
  void  read_fdef_file(const std::string& filename);

};

END_NAMESPACE_STIR
#endif
