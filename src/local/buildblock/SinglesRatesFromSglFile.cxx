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

#include "stir/DetectionPosition.h"
#include "stir/IndexRange.h"
#include "stir/IndexRange2D.h"
#include "local/stir/SinglesRatesFromSglFile.h"
#include "stir/round.h"

#include <vector>
#ifdef HAVE_LLN_MATRIX
#include "ecat_model.h"
#include "stir/IO/stir_ecat7.h"
#endif
#include <fstream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::streampos;
#endif



START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7
const unsigned 
SinglesRatesFromSglFile::SIZE_OF_SINGLES_RECORD = 4*128;

const char * const 
SinglesRatesFromSglFile::registered_name = "Singles From Sgl File"; 


const double MAX_INTERVAL_DIFFERENCE = 0.05; // 5% max difference.


static inline 
unsigned long int
convert_4_bytes(unsigned char * buffer)
{
  // The order from the file is always big endian. The native order doesn't matter
  // when converting by multiplying and adding the individual bytes.
  //if (ByteOrder::get_native_order() == ByteOrder::big_endian)
  //  return buffer[0] + 256UL*(buffer[1] + 256UL*(buffer[2] + 256UL*buffer[3]));
  //else
  return buffer[3] + 256UL*(buffer[2] + 256UL*(buffer[1] + 256UL*buffer[0]));

}



static inline
void
convert_int_to_4_bytes(unsigned long int val, unsigned char *buffer) {
  // Big endian
  buffer[0] = (val & 0xff000000) >> 24;
  buffer[1] = (val & 0x00ff0000) >> 16;
  buffer[2] = (val & 0x0000ff00) >> 8;
  buffer[3] = (val & 0x000000ff);
}





// Constructor
SinglesRatesFromSglFile::
SinglesRatesFromSglFile()
{}



// Get the average singles rate for a particular bin.
float
SinglesRatesFromSglFile::
get_singles_rate(const DetectionPosition<>& det_pos,
                 const double start_time, const double end_time) const {
  
  int singles_bin_index = scanner_sptr->get_singles_bin_index(det_pos);
  
  return(get_singles_rate(singles_bin_index, start_time, end_time));
}




// Generate a FramesSinglesRate - containing the average rates
// for a frame begining at start_time and ending at end_time.
FrameSinglesRates
SinglesRatesFromSglFile::
get_rates_for_frame(double start_time,
                    double end_time) const {

  int num_singles_units = scanner_sptr->get_num_singles_units();

  // Create a temporary vector
  vector<float> average_singles_rates(num_singles_units);
  

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
                                scanner_sptr);
  
  return(frame_rates);
  
}





// Get time slice index.
// Returns the index of the slice that contains the specified time.
int
SinglesRatesFromSglFile::
get_end_time_slice_index(double t) const {

  int slice_index = 0;

  // Start with an initial estimate.
  if ( _singles_time_interval != 0 ) {
    slice_index = static_cast<int>(floor(t / _singles_time_interval));
  }

  if ( slice_index >= _num_time_slices ) {
    slice_index = _num_time_slices - 1;
  } 


  // Check estimate and determine whether to look further or backwards.
  // Note that we could just move fowards first and then backwards but this
  // method is more intuitive.

  if ( _times[slice_index] < t ) {
    
    // Check forwards.
    while( slice_index < _num_time_slices - 1 &&
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
SinglesRatesFromSglFile::
get_start_time_slice_index(double t) const {

  int slice_index = 0;

  // Start with an initial estimate.
  if ( _singles_time_interval != 0 ) {
    slice_index = static_cast<int>(floor(t / _singles_time_interval));
  }

  if ( slice_index >= _num_time_slices ) {
    slice_index = _num_time_slices - 1;
  } 


  // Check estimate and determine whether to look further or backwards.
  // Note that we could just move fowards first and then backwards but this
  // method is more intuitive.

  if ( _times[slice_index] < t ) {
    
    // Check forwards.
    while( slice_index < _num_time_slices - 1 &&
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
SinglesRatesFromSglFile::
get_singles_rate(int singles_bin_index, int time_slice) const {
  
  // Check ranges.
  int total_singles_units = scanner_sptr->get_num_singles_units();
  
  if ( singles_bin_index < 0 || singles_bin_index >= total_singles_units ||
       time_slice < 0 || time_slice >= _num_time_slices ) {
    return(0);
  } else {
    return _singles[time_slice][singles_bin_index];
  }

}



// Set a singles rate by bin index and time slice.
void 
SinglesRatesFromSglFile::
set_singles_rate(int singles_bin_index, int time_slice, int new_rate) {
  
  int total_singles_units = scanner_sptr->get_num_singles_units();
  
  if ( singles_bin_index >= 0 && singles_bin_index < total_singles_units &&
       time_slice >= 0 && time_slice < _num_time_slices ) {
    _singles[time_slice][singles_bin_index] = new_rate;
  }
}




int 
SinglesRatesFromSglFile::
rebin(vector<double>& new_end_times) {

  const int num_new_slices = new_end_times.size();
  const int total_singles_units = scanner_sptr->get_num_singles_units();
  
  // Create the new array of singles data.
  Array<2, int> new_singles = Array<2, int>(IndexRange2D(0, num_new_slices - 1, 
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
    if ( start_time > _times[_num_time_slices - 1] ) {
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
  _singles = new_singles;
  _times = new_end_times;
  _num_time_slices = _times.size();
  
  return(_num_time_slices);
}


 
vector<double> 
SinglesRatesFromSglFile::get_times() const
{
  return _times;
}




int
SinglesRatesFromSglFile:: 
get_num_time_slices() const {
  return(_num_time_slices);
}



double
SinglesRatesFromSglFile:: 
get_singles_time_interval() const {
  return(_singles_time_interval);
}





int
SinglesRatesFromSglFile::
read_singles_from_sgl_file(const string& sgl_filename)
{

  int slice = 0;

#ifndef HAVE_LLN_MATRIX

  error("Compiled without ECAT7 support\n");

#else

  ifstream singles_file(sgl_filename.c_str(), ios::binary);
  if (!singles_file) {
    error("\nSinglesRatesFromSglFile: Couldn't open \"%s\".\n", sgl_filename.c_str());
  }

  
  //first find out the size of the file
  singles_file.seekg(0, ios::end);
  const streampos end_stream_position = singles_file.tellg();
  if (!singles_file) {
    error("\nSinglesRatesFromSglFile: Couldn't seek to end of file %s.",sgl_filename.c_str());
  }


  // go to the beginning and read the singles header
  singles_file.seekg(0, ios::beg);
 
  if (!singles_file) {
    error("\nSinglesRatesFromSglFile: Couldn't seek to start of file %s.",sgl_filename.c_str());
  }
  

  {
    char buffer[sizeof(Main_header)];
    singles_file.read(buffer,sizeof(_singles_main_header));
    if (!singles_file)
    {
      error("\nSinglesRatesFromSglFile: Couldn't read main_header from %s.",sgl_filename.c_str());
    }
    else
    {
      unmap_main_header(buffer, &_singles_main_header);
      ecat::ecat7::find_scanner(scanner_sptr, _singles_main_header);
    }
  }

  
  if (scanner_sptr->get_type() != Scanner::E966) {
    warning("check SinglesRatesFromSglFile for non-966\n");
  }


  // Get total number of bins for this type of scanner.
  const int total_singles_units = scanner_sptr->get_num_singles_units();

  // Calculate number of time slices from the length of the data (file size minus header).
  _num_time_slices =  
    static_cast<int>((end_stream_position - static_cast<streampos>(512)) /
                     SIZE_OF_SINGLES_RECORD);

   // Allocate the main array.
  _singles = Array<2, int>(IndexRange2D(0, _num_time_slices - 1, 0, total_singles_units - 1));

  
  singles_file.seekg(512, ios::beg);
  
  while (singles_file && slice < _num_time_slices) {
    
    // Temporary space to store file data.
    sgl_str singles_str;


    {
      unsigned char buffer[SIZE_OF_SINGLES_RECORD];
      
      singles_file.read(reinterpret_cast<char *>(buffer), SIZE_OF_SINGLES_RECORD);
      if (!singles_file) {
        
        if (!singles_file.eof()) {
          warning("Error reading singles file record %d. Stopped reading from this point.", 
                  slice);
        }

        break;
      }

      singles_str.time = convert_4_bytes(buffer);
      singles_str.num_sgl = convert_4_bytes(buffer+4);
      
      for (unsigned int i = 0; i < ( SIZE_OF_SINGLES_RECORD - 8)/4; ++i) {
        singles_str.sgl[i] = convert_4_bytes(buffer+8+4*i);
      }
    }

    
    if (singles_str.num_sgl != total_singles_units) {
      error("Number of singles units should be %d, but is %d in singles file",
	    total_singles_units,  singles_str.num_sgl);
    }
    


    // Copy the singles values to the main array.
    
    // Note. The singles values are arranged num_axial sets of num_transaxial
    // values.
    //
    // For a singles values for a unit at axial_index, transaxial_index
    // the values is found at single_str.sgl[]
    // singles_str.sgl[ transaxial_index + (axial_index * num_transaxial) ]
    //
    // The singles values are stored in the _singles array in the same order.
    // For other file formats the ordering of the units may be different.
    for (int singles_bin = 0; singles_bin < total_singles_units; ++singles_bin) {
      _singles[slice][singles_bin] = static_cast<int>(singles_str.sgl[singles_bin]);
    }
    
    
    // singles in the sgl file given in msec.multiply with 0.001 to convert into sec.
    _times.push_back(singles_str.time*0.001);

    // Add the last two words - total prompts and total randoms.
    _total_prompts.push_back(singles_str.sgl[total_singles_units]);
    _total_randoms.push_back(singles_str.sgl[total_singles_units + 1]);
    
    // Increment the slice index.
    ++slice;
    
  }
  
  assert(_times.size()!=0);
  _singles_time_interval = _times[1] - _times[0];
  
  if (slice != _num_time_slices)
  {
    error("\nSinglesRatesFromSglFile: Couldn't read all records in the .sgl file %s. Read %d of %d. Exiting\n",
	  sgl_filename.c_str(), slice, _num_time_slices);
    //TODO resize singles to return array with new sizes
  }

#endif

  // Return number of time slices read.
  return slice; 
    
}





// Write SinglesRatesFromSglFile to a singles file.
std::ostream& 
SinglesRatesFromSglFile::write(std::ostream& output) {

#ifndef HAVE_LLN_MATRIX

  error("Compiled without ECAT7 support\n");

#else
  
  char header_buffer[SIZE_OF_SINGLES_RECORD];
  unsigned char buffer[SIZE_OF_SINGLES_RECORD];
  
  memset(header_buffer, 0, SIZE_OF_SINGLES_RECORD);

  // Write header to buffer.
  map_main_header(header_buffer, &(this->_singles_main_header));
  
  // Write buffer to output.
  output.write(header_buffer, SIZE_OF_SINGLES_RECORD);

  if (!output) {
    error("\nSinglesRatesFromSglFile: Failed to write to output.");
    return(output);
  }
  
  
  // Empty buffer.
  memset(buffer, 0, SIZE_OF_SINGLES_RECORD);
  
  int total_singles_units = scanner_sptr->get_num_singles_units();
  unsigned long millisecs;
  
  // Write 512 byte blocks. One for each time slice recorded.
  for(int slice = 0 ; slice < _num_time_slices ; ++slice) {
    
    // Write data to buffer.
    millisecs = static_cast<unsigned long>(floor(_times[slice] * 1000.0));
 
    // Time and number of singles units
    convert_int_to_4_bytes(millisecs, buffer);
    convert_int_to_4_bytes(total_singles_units, buffer + 4);

    // Singles units
    // Note that the order of values in _singles is the same as that of the file.
    // This may not be the case for other file formats.
    for(int singles_bin = 0 ; singles_bin < total_singles_units ; ++singles_bin) {
      convert_int_to_4_bytes(_singles[slice][singles_bin], buffer + ((2 + singles_bin) * 4));
    }
    
    // Total prompts and total trues
    convert_int_to_4_bytes(_total_prompts[slice], buffer + ((2 + total_singles_units) * 4));
    convert_int_to_4_bytes(_total_randoms[slice], buffer + ((2 + total_singles_units + 1) * 4));
    
    
    // Write buffer to output.
    output.write(reinterpret_cast<char *>(buffer), SIZE_OF_SINGLES_RECORD);
    
    if (!output) {
      error("\nSinglesRatesFromSglFile: Failed to write to output.");
      break;
    }

  }
  

#endif

  return output;
}




/*
 *
 * Private methods.
 *
 */




float
SinglesRatesFromSglFile::
get_singles_rate(int singles_bin_index,
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
    total_singles = static_cast<double>(_singles[start_slice][singles_bin_index]);
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
    total_singles = fraction * _singles[start_slice][singles_bin_index];
    
    
    
    // Calculate the fraction of the end_slice to include.
    slice_start_time = get_slice_start(end_slice);
    slice_end_time = _times[end_slice];
        
    old_duration = slice_end_time - slice_start_time;
    included_duration = end_time - slice_start_time;
        
    fraction = included_duration / old_duration;
    
    // Add this fraction to the total of the number of bins contributing.
    total_slices += fraction;
    
    // Add the fraction of the bin to the running total.
    total_singles += fraction * _singles[end_slice][singles_bin_index];
 
    
    // Add all intervening slices.
    for(int slice = start_slice + 1; slice < end_slice ; ++slice, total_slices += 1.0) {
      total_singles += _singles[slice][singles_bin_index];
    }
    
    
    // Divide by total amount of contributing slices.
    total_singles = total_singles / total_slices;
       
  }

  return( static_cast<float>(total_singles) );
  
}




void 
SinglesRatesFromSglFile::
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
SinglesRatesFromSglFile::
get_slice_start(int slice_index) const {

  if ( slice_index >= _num_time_slices ) {
    slice_index = _num_time_slices - 1;
  }
  
  if ( slice_index == 0 ) {
    return(0);
  } else {
    return(_times[slice_index - 1]);
  }
}
 





void 
SinglesRatesFromSglFile::
initialise_keymap()
{
  parser.add_start_key("Singles Rates From Sgl File");
  parser.add_key("sgl_filename", &_sgl_filename);
  parser.add_stop_key("End Singles Rates From Sgl File");
}

bool 
SinglesRatesFromSglFile::
post_processing()
{
  read_singles_from_sgl_file(_sgl_filename);
  return false;
}


void 
SinglesRatesFromSglFile::set_defaults()
{
  _sgl_filename = "";
}








END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR



