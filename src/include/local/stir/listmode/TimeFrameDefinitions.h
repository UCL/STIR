//
// $Id$
//
/*!

  \file
  \ingroup buildblock  
  \brief Declaration of class TimeFrameDefinitions
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_listmode_TimeFrameDefinitions_H__
#define __stir_listmode_TimeFrameDefinitions_H__

#include "stir/common.h"
#include <string>
#include <vector>
#include <utility>

#ifndef STIR_NO_NAMESPACES
using std::ifstream;
using std::string;
using std::pair;
using std::vector;
#endif

START_NAMESPACE_STIR
//! Class used for storing time frame durations
class TimeFrameDefinitions
{
public:
  TimeFrameDefinitions();

  //! Read the frame definitions from a .fdef file
  /*! This uses the '.fdef' format used by Peter Bloomfield's software.
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
  
  //! 1 based
  double get_start_time(unsigned int frame_num) const;
  double get_end_time(unsigned int frame_num) const;
  
  double get_start_time() const;
  double get_end_time() const;
  
  unsigned int get_num_frames() const;
  
private:
  vector<pair<double, double> > frame_times;
};

END_NAMESPACE_STIR
#endif
