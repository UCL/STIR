//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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
  \ingroup utilities
  \brief Utility program scans the singles file looking for anomalous values

  \author Kris Thielemans
  \author Tim Borgeaud
  $Date$
  $Revision$
*/


#include "stir/data/SinglesRatesFromSglFile.h"

#include <string>
#include <fstream>
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::string;
using std::vector;
#endif

USING_NAMESPACE_STIR



const int MAX_VALID_VALUE = 1000000;


// Minimum allowed fraction of previous value.
const float MIN_PREVIOUS_FRACTION = 0.225;

// Max allowed multiple of previous value.
const float MAX_PREVIOUS_MULTIPLE = 3.0;

// Minimum allowed fraction of median of next set of values.
const float MIN_MEDIAN_FRACTION = 0.2;

// Max allowed multiple of median of next set of values.
const float MAX_MEDIAN_MULTIPLE = 3.5;



// Tolerance between adjacent bins.
const float ADJACENT_TOLERANCE = 0.75;

const int MEDIAN_SIZE = 7; // Number of elements forward for which the median is calculated.



//
// Function to calculate the median of singles values from
// start_index to end_index (inclusive).
//
int 
calcMedian(const ecat::ecat7::SinglesRatesFromSglFile& singles_rates,
           int singles_bin_index, int start_slice, int end_slice) {
  
  int num_elements = end_slice - start_slice + 1;
  
  // Create a new temporary vector.
  vector<int> elements(num_elements);
  
  // Fill the temporary vector.
  for(int index = 0 ; index < num_elements ; index++) {
    elements[index] = singles_rates.get_singles_rate(singles_bin_index, 
                                                     start_slice + index);
  }
  
  // Calculate which element, of the sorted set, will be the median.
  vector<int>::iterator median = elements.begin() + (num_elements / 2);
  
  // Partially sort the elements.
  nth_element(elements.begin(), median, elements.end());
  
  // Return the element at the median position.
  return(*median);
  
}





//
// Function to calculate a median of up to MEDIAN_SIZE values
// starting from start_slice.
//
int
getMedian(const ecat::ecat7::SinglesRatesFromSglFile& singles_rates,
          int singles_bin_index, int start_slice) {

  int num_slices = singles_rates.get_num_time_slices();

  int end_slice = start_slice + MEDIAN_SIZE - 1;
  if ( end_slice >= num_slices ) {
    end_slice = num_slices - 1;
  }
  
  return(calcMedian(singles_rates, singles_bin_index, start_slice, end_slice));
}



bool
compareAdjacentBin(int value, int slice, int singles_bin_index,
                   const ecat::ecat7::SinglesRatesFromSglFile& singles_rates,
                   int transaxial_offset) {

  const Scanner *scanner = singles_rates.get_scanner_ptr();
  int axial = scanner->get_axial_singles_unit(singles_bin_index);
  int num_transaxial = scanner->get_num_transaxial_singles_units();
  int transaxial = scanner->get_transaxial_singles_unit(singles_bin_index);
  
  transaxial = (transaxial + transaxial_offset) % num_transaxial;

  if ( transaxial < 0 ) {
    transaxial += num_transaxial;
  }
  
    
  int bin_index = scanner->get_singles_bin_index(axial, transaxial);

  int rate = singles_rates.get_singles_rate(bin_index, slice);
  
  if ( (value >= (1.0 - ADJACENT_TOLERANCE) * rate) &&
       (value <= (1.0 + ADJACENT_TOLERANCE) * rate) ) {
    return(true);
  }
  
  return(false);
}




//
// Check a rate against transaxially adjacent bins.
//
// Note that a single bucket controller encompasses a number of 
// axial singles units, so checking bins with the same transaxial 
// position may result checking bins from the same bucket.
//
bool
checkAdjacentBins(int value, int slice, int singles_bin_index, 
                  const ecat::ecat7::SinglesRatesFromSglFile& singles_rates) {
  
  if ( compareAdjacentBin(value, slice, singles_bin_index, singles_rates, -1) ) {
    if ( compareAdjacentBin(value, slice, singles_bin_index, singles_rates, -2) || 
         compareAdjacentBin(value, slice, singles_bin_index, singles_rates, 1)) {
      return(true);
    }
  } else if ( compareAdjacentBin(value, slice, singles_bin_index, singles_rates, 1) && 
              compareAdjacentBin(value, slice, singles_bin_index, singles_rates, 2) ) {
    return(true);
  }
              
  return(false);
}





//
// Check that a value is either within a tolerance of the median
// value of the next MEDIAN_SIZE values or is between the median
// value and a supplied value.
//
// Returns true if the value is ok.
// 
bool
checkValue(int value, int slice, int singles_bin_index, 
           const ecat::ecat7::SinglesRatesFromSglFile& singles_rates,
           int previous_value) {

  
  // Check absolute range ( 0 <= Value <= MAX_VALID_VALUE ). 
  if ( value > 0 && value < MAX_VALID_VALUE ) {
    
    // Check against previous value.
    if ( ((MIN_PREVIOUS_FRACTION * previous_value) - value) <= 0 &&
         ((MAX_PREVIOUS_MULTIPLE * previous_value) - value) >= 0 ) {
      // Value is good.
      return(true);
    }


    // Check against values from adjacent (transaxial) bins.
    if ( checkAdjacentBins(value, slice, singles_bin_index, singles_rates) ) {
      // Value is good.
      return(true);
    }



    // See if there are any further values to compare the current value with.
    int num_slices = singles_rates.get_num_time_slices();
    
    if ( slice < num_slices - 1 ) {
      
      // Calculate the median (should work for only one value).
      int median = getMedian(singles_rates, singles_bin_index, slice + 1);


      // Check against median value. Note that the first slice only needs be less than
      // median * MAX_MULTIPLE.
      if ( (slice == 0 || ((MIN_MEDIAN_FRACTION * median - value)) <= 0) &&
           ((MAX_MEDIAN_MULTIPLE * median) - value) >= 0 ) {
        // Value is good.
        return(true);
      }

      
      // Check to see if the value lies between previous and median.
      if ( ((previous_value > median) && (value >= median && value <= previous_value)) ||
           (value >= previous_value && value <= median) ) {
        // Value is good.
        return(true);
      }
    }
  }
  
  return(false);
}







//
// Function to check and correct a particular value.
//
// Returns true if a value was corrected.
bool
correctSinglesValue(int value, int slice, int singles_bin_index,
                    ecat::ecat7::SinglesRatesFromSglFile& singles_rates) {
  
  
  // Previous value. Set to 0 for the first slice.
  int previous = 0;


  if ( slice > 0 ) {
    previous = singles_rates.get_singles_rate(singles_bin_index, slice - 1);
  }

  
  if ( checkValue(value, slice, singles_bin_index, singles_rates, previous) == true ) {
    return(false);
  }

  
  // The value is not good. Therefore, correct the value using
  // the previous value (which must be good) and the next good value.
  int num_slices = singles_rates.get_num_time_slices();
  int next_slice;
  int next_value;
    
  for(next_slice = slice + 1; next_slice < num_slices ; ++next_slice ) {
    next_value = singles_rates.get_singles_rate(singles_bin_index, next_slice);
    
    if ( checkValue(next_value, next_slice, singles_bin_index, singles_rates, previous) ) {
      break;
    }
  }
  
  if ( next_slice < num_slices ) {
    // Correct using previous and next_slice.
    int next_sep = next_slice - slice;
    
    // Adjust value.
    value = ((previous * next_sep) + next_value) / (1 + next_sep); 
    
    singles_rates.set_singles_rate(singles_bin_index, slice, value);
  }
  
  // Value was bad so return true.
  return(true);
  
}






// 
// check and correct all the singles values. Return true if any values were corrected.
// 
//
bool
correctAllValues(ecat::ecat7::SinglesRatesFromSglFile& singles_rates,
                 std::ostream& output) {

  bool changed = false;

  // Get scanner details and, from these, the number of singles units.
  const Scanner *scanner = singles_rates.get_scanner_ptr();
  int total_singles_units = scanner->get_num_singles_units();
  
  // Get total number of time slices.
  int num_slices = singles_rates.get_num_time_slices();

  // Loop over all bins.
  for(int singles_bin = 0 ; singles_bin < total_singles_units ; ++singles_bin) {
    
    // Loop over each time slice.
    for(int slice = 0 ; slice < num_slices ; ++slice) {
      
      int value = singles_rates.get_singles_rate(singles_bin, slice);
            
      if ( correctSinglesValue(value, slice, singles_bin, singles_rates) ) {
        
        changed = true;
        int new_value = singles_rates.get_singles_rate(singles_bin, slice);

        // Print details of the change to the output.
        output << "Bin " << singles_bin
               << "  Slice index: " << slice
               << "  Rate: " << value
               << "  New estimate: " << new_value
               << "\n";
      }
      
    }
    
  }
  
  return(changed);
  
}







int 
main (int argc, char **argv)
{

  // Check arguments. 
  // Singles filename + optional output filename
  if (argc > 3 || argc < 2) {
      cerr << "Usage: " << argv[0] << " sgl_filename [output_filename]\n";
      exit(EXIT_FAILURE);
  }


  const string sgl_filename = argv[1];

  bool write_file = false;
  string output_filename = "";
  if ( argc == 3 ) {
    write_file = true;
    output_filename = argv[2];
  } 
  
  
  // Singles file object.
  ecat::ecat7::SinglesRatesFromSglFile singles_from_sgl;

  // Read the singles file.
  singles_from_sgl.read_singles_from_sgl_file(sgl_filename);


  const vector<double> times = singles_from_sgl.get_times();



  std::ofstream output;

  // If necessary, open the output file.
  if ( output_filename.length() ) {

    output.open(output_filename.c_str(), std::ios_base::out);

    if (!output.good())
      error("Error opening output file\n");
  }

  
  
  if ( correctAllValues(singles_from_sgl, cout) && write_file ) {
    // Write data to output file.
    singles_from_sgl.write(output);
  }
  
  
  return EXIT_SUCCESS;
}
