/*
    Copyright (C) 2003-2007, Hammersmith Imanet Ltd
    Copyright (C) 2017-2018, University of Leeds
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
  \brief Implementation of SinglesRatesFromGEHDF5

  \author Palak Wadhwa
  \author Kris Thielemans
*/

#include "stir/DetectionPosition.h"
#include "stir/IndexRange.h"
#include "stir/IndexRange2D.h"
#include "stir/data/SinglesRatesFromGEHDF5.h"
#include "stir/round.h"
#include "stir/stream.h"
#include "stir/IO/HDF5Wrapper.h"

#include <vector>
#include <fstream>
#include <algorithm>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::streampos;
using std::ios;
#endif



START_NAMESPACE_STIR
//PW Need to change the registered name
const char * const 
SinglesRatesFromGEHDF5::registered_name = "Singles From listmode File"; 

const double MAX_INTERVAL_DIFFERENCE = 0.05; // 5% max difference.

//PW Will not change this bit of code.
// Constructor
SinglesRatesFromGEHDF5::
SinglesRatesFromGEHDF5()
{}


// Generate a FramesSinglesRate - containing the average rates
// for a frame begining at start_time and ending at end_time.
FrameSinglesRates
SinglesRatesFromGEHDF5::
get_rates_for_frame(double start_time,
                    double end_time) const {

  int num_singles_units = SinglesRates::scanner_sptr->get_num_singles_units();

  // Create a temporary vector
  std::vector<float> average_singles_rates(num_singles_units);


  // Loop over all bins.
  for(int singles_bin = 0 ; singles_bin < num_singles_units ; ++singles_bin) {
    average_singles_rates[singles_bin] = get_singles_rate(singles_bin, start_time, end_time);
  }
  
  // Determine that start and end slice indices.
  int start_slice = get_start_time_slice_index(start_time);
  int end_slice = get_end_time_slice_index(end_time);
  
  double frame_start_time;
  if ( start_slice == 0 ) {
    frame_start_time = _times[0] - _singles_time_interval;
  } else {
    frame_start_time = _times[start_slice - 1];
  }

  double frame_end_time = _times[end_slice];

  // Create temp FrameSinglesRate object
  FrameSinglesRates frame_rates(average_singles_rates,
                                frame_start_time,
                                frame_end_time,
                                SinglesRates::scanner_sptr);
  
  return(frame_rates);
  
}





// Get time slice index.
// Returns the index of the slice that contains the specified time.
int
SinglesRatesFromGEHDF5::
get_end_time_slice_index(double t) const {

  int slice_index = 0;

  // Start with an initial estimate.
  if ( _singles_time_interval != 0 ) {
    slice_index = static_cast<int>(floor(t / _singles_time_interval));
  }

  if ( slice_index >= m_num_time_slices ) {
    slice_index = m_num_time_slices - 1;
  } 


  // Check estimate and determine whether to look further or backwards.
  // Note that we could just move fowards first and then backwards but this
  // method is more intuitive.

  if ( _times[slice_index] < t ) {
    
    // Check forwards.
    while( slice_index < m_num_time_slices - 1 &&
           _times[slice_index] < t ) {
      slice_index++;
    }

  } else {

    // Check backwards.
    while( slice_index > 0 && _times[slice_index - 1] >= t ) {
      slice_index--;
    }

  }
  
  return(slice_index);
}




// Get time slice index.
// Returns first slice ending _after_ t.
int
SinglesRatesFromGEHDF5::
get_start_time_slice_index(double t) const {

  int slice_index = 0;

  // Start with an initial estimate.
  if ( _singles_time_interval != 0 ) {
    slice_index = static_cast<int>(floor(t / _singles_time_interval));
  }

  if ( slice_index >= m_num_time_slices ) {
    slice_index = m_num_time_slices - 1;
  } 


  // Check estimate and determine whether to look further or backwards.
  // Note that we could just move fowards first and then backwards but this
  // method is more intuitive.

  if ( _times[slice_index] < t ) {
    
    // Check forwards.
    while( slice_index < m_num_time_slices - 1 &&
           _times[slice_index] <= t ) {
      slice_index++;
    }

  } else {

    // Check backwards.
    while( slice_index > 0 && _times[slice_index - 1] > t ) {
      slice_index--;
    }

  }
  
  return(slice_index);
}






// Get rates using time slice and singles bin indices.
int 
SinglesRatesFromGEHDF5::
get_singles_rate(int singles_bin_index, int time_slice) const {
  
  // Check ranges.
  int total_singles_units = SinglesRates::scanner_sptr->get_num_singles_units();
  
  if ( singles_bin_index < 0 || singles_bin_index >= total_singles_units ||
       time_slice < 0 || time_slice >= m_num_time_slices ) {
    return(0);
  } else {
    return (*m_singles_sptr)[time_slice][singles_bin_index];
  }

}



// Set a singles rate by bin index and time slice.
void 
SinglesRatesFromGEHDF5::
set_singles_rate(int singles_bin_index, int time_slice, int new_rate) {
  
  int total_singles_units = SinglesRates::scanner_sptr->get_num_singles_units();
  
  if ( singles_bin_index >= 0 && singles_bin_index < total_singles_units &&
       time_slice >= 0 && time_slice < m_num_time_slices ) {
    (*m_singles_sptr)[time_slice][singles_bin_index] = new_rate;
  }
}




unsigned int SinglesRatesFromGEHDF5::rebin(std::vector<double>& new_end_times) {

  const int num_new_slices = new_end_times.size();
  const int total_singles_units = SinglesRates::scanner_sptr->get_num_singles_units();
  
  // Create the new array of singles data.
  Array<2, unsigned int> new_singles = Array<2, unsigned int>(IndexRange2D(0, num_new_slices - 1,
                                                         0, total_singles_units - 1));
  
  
  // Sort the set of new time slices.
  std::sort(new_end_times.begin(), new_end_times.end());

  // Start with initial time of 0.0 seconds.
  double start_time = 0;

  
  // Loop over new time slices.
  for(unsigned int new_slice = 0 ; new_slice < new_end_times.size(); ++new_slice) {

    // End time for the new time slice.
    double end_time = new_end_times[new_slice];

    // If start time is beyond last end time in original data, then use zeros.
    if ( start_time > _times[m_num_time_slices - 1] ) {
      for(int singles_bin = 0 ; singles_bin < total_singles_units ; ++singles_bin) {
        new_singles[new_slice][singles_bin] = 0;
      }
    } else {

      // Get the singles rate average between start and end times for all bins.
      for(int singles_bin = 0 ; singles_bin < total_singles_units ; ++singles_bin ) {
        new_singles[new_slice][singles_bin] = 
          round(get_singles_rate(singles_bin, start_time, end_time));
      }
      
    }
    
    // Next new time slice starts at the end of this slice.
    start_time = end_time;
    
  }
  
  
  // Set the singles and times using the new sets.
  m_singles_sptr.reset(new Array<2, unsigned int>(new_singles));
  _times = new_end_times;
  m_num_time_slices = _times.size();
  
  return(m_num_time_slices);
}


 
std::vector<double> 
SinglesRatesFromGEHDF5::get_times() const
{
  return _times;
}




unsigned int
SinglesRatesFromGEHDF5:: 
get_num_time_slices() const {
  return(m_num_time_slices);
}



double
SinglesRatesFromGEHDF5:: 
get_singles_time_interval() const {
  return(_singles_time_interval);
}





unsigned int
SinglesRatesFromGEHDF5::
read_singles_from_listmode_file(const std::string& _listmode_filename)
{

    unsigned int slice = 1;

    //PW Open the list mode file here.
    m_input_sptr.reset(new HDF5Wrapper(_listmode_filename));


    SinglesRates::scanner_sptr = m_input_sptr->get_scanner_sptr();
    // Get total number of bins for this type of scanner.
    const int total_singles_units = SinglesRates::scanner_sptr->get_num_singles_units();


    m_num_time_slices = m_input_sptr->get_timefreme_definitions()->get_num_frames();

    // Allocate the main array.
    m_singles_sptr.reset(new Array<2, unsigned int>(IndexRange2D(0, m_num_time_slices - 1, 0, total_singles_units - 1)));

    m_input_sptr->initialise_singles_data();
    
    while ( slice < m_num_time_slices)
    {
        m_input_sptr->get_dataspace(slice, m_singles_sptr);
        // Increment the slice index.
        ++slice;
    }

    //PW Modify this bit of code too.
    if (slice != m_num_time_slices)
    {
        error("\nSinglesRatesFromGEHDF5: Couldn't read all records in the file. Read %d of %d. Exiting\n",
              slice, m_num_time_slices);
        //TODO resize singles to return array with new sizes
    }

    _times = std::vector<double>(m_num_time_slices);
    for(unsigned int slice = 0;slice < m_num_time_slices;++slice)
        _times[slice] = slice+1.0;

    assert(_times.size()!=0);
    _singles_time_interval = _times[1] - _times[0];

    // Return number of time slices read.
    return slice;
    
}





// Write SinglesRatesFromGEHDF5 to a singles file.
std::ostream& 
SinglesRatesFromGEHDF5::write(std::ostream& output) {

  output << (*m_singles_sptr) << std::endl;

  return output;
}

//PW Figure this out!Does it need any change?
float
SinglesRatesFromGEHDF5::
get_singles_rate(const int singles_bin_index,
                 const double start_time, const double end_time) const {

  // First Calculate an inclusive range. start_time_slice is the 
  // the first slice with an ending time greater than start_time.
  // end_time_slice is the first time slice that ends at, or after,
  // end_time.
  int start_slice = this->get_start_time_slice_index(start_time);
  int end_slice = this->get_end_time_slice_index(end_time);
  
  
  // Total contribution from all slices.
  double total_singles;


  if ( start_slice == end_slice ) {
    // If the start and end slices are the same then just use that time slice.
    total_singles = static_cast<double>((*m_singles_sptr)[start_slice][singles_bin_index]);
  } else {
    
    // Start and end times for starting and ending slices.
    double slice_start_time;
    double slice_end_time;
    double included_duration; 
    double old_duration;
    double fraction;
    
    
    // Total slices included (including fractional amounts) in the average.
    float total_slices;
        
    

    // Calculate the fraction of the start_slice to include.
    slice_start_time = get_slice_start(start_slice);
    slice_end_time = _times[start_slice];
    
    old_duration = slice_end_time - slice_start_time;
    included_duration = slice_end_time - start_time;
        
    fraction = included_duration / old_duration;
       
    
    // Set the total number of contributing bins to this fraction.
    total_slices = fraction;
    
    // Set the total singles so far to be the fraction of the bin.
    total_singles = fraction * (*m_singles_sptr)[start_slice][singles_bin_index];
    
    
    
    // Calculate the fraction of the end_slice to include.
    slice_start_time = get_slice_start(end_slice);
    slice_end_time = _times[end_slice];
        
    old_duration = slice_end_time - slice_start_time;
    included_duration = end_time - slice_start_time;
        
    fraction = included_duration / old_duration;
    
    // Add this fraction to the total of the number of bins contributing.
    total_slices += fraction;
    
    // Add the fraction of the bin to the running total.
    total_singles += fraction * (*m_singles_sptr)[end_slice][singles_bin_index];
 
    
    // Add all intervening slices.
    for(int slice = start_slice + 1; slice < end_slice ; ++slice, total_slices += 1.0) {
      total_singles += (*m_singles_sptr)[slice][singles_bin_index];
    }
    
    
    // Divide by total amount of contributing slices.
    total_singles = total_singles / total_slices;
       
  }

  return( static_cast<float>(total_singles) );
  
}

float
SinglesRatesFromGEHDF5::
get_singles_rate(const DetectionPosition<>& det_pos,
                const double start_time,
                const double end_time) const
{
    return SinglesRates::get_singles_rate(det_pos, start_time, end_time);
}

/*
 *
 * Private methods.
 *
 */





void 
SinglesRatesFromGEHDF5::
set_time_interval() {

  // Run through the _times vector and calculate an average difference 
  // between the starts of consecutive time slices.
  
  // Min and max differences (slice durations).
  double min_diff = 0;
  double max_diff = 0;
  double total = 0;

  for(std::vector<double>::const_iterator t = _times.begin(); t < _times.end() - 1; ++t) {
    double diff = *(t + 1) - *t;
    total += diff;

    if ( min_diff == 0 || diff < min_diff ) {
      min_diff = diff;
    }

    if ( diff > max_diff ) {
      max_diff = diff;
    }
  }

  _singles_time_interval = total / (_times.size() - 1);

  if ( (max_diff - min_diff) / (_singles_time_interval) > MAX_INTERVAL_DIFFERENCE ) {
    // Slice durations are not consistent enough to be considered the same.
    _singles_time_interval = 0;
  }
  
}


// get slice start time.
double 
SinglesRatesFromGEHDF5::
get_slice_start(int slice_index) const {

  if ( slice_index >= m_num_time_slices ) {
    slice_index = m_num_time_slices - 1;
  }
  
  if ( slice_index == 0 ) {
    return(0);
  } else {
    return(_times[slice_index - 1]);
  }
}
 





void 
SinglesRatesFromGEHDF5::
initialise_keymap()
{
//PW Modify this to change sgl to listmode
  parser.add_start_key("Singles Rates From listmode File");
  parser.add_key("listmode_filename", &_listmode_filename);
  parser.add_stop_key("End Singles Rates From listmode File");
}

bool 
SinglesRatesFromGEHDF5::
post_processing()
{
  read_singles_from_listmode_file(_listmode_filename);
  return false;
}


void 
SinglesRatesFromGEHDF5::set_defaults()
{
  _listmode_filename = "";
}








END_NAMESPACE_STIR



