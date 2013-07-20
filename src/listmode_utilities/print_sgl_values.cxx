/*
    Copyright (C) 2004-2009, Hammersmith Imanet Ltd
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
  \brief Utility program that prints out values from an sgl file.

  \author Kris Thielemans
  \author Tim Borgeaud
*/


#include "stir/data/SinglesRatesFromSglFile.h"


#include <fstream>
#include <iomanip>
#include <string>
#include <vector>


#ifndef STIR_NO_NAMESPACES
using std::cout;
using std::cerr;
using std::endl;
using std::setw;
using std::string;
using std::vector;
#endif

USING_NAMESPACE_STIR




int 
main (int argc, char **argv)
{

  vector<int> columns;

  // Check arguments. 
  // Singles filename + optional bin indices.
  if (argc < 2) {
    cerr << "Program to print out values from a singles file.\n\n";
    cerr << "Usage: " << argv[0] << " sgl_filename [bin_index ...]\n\n";
    cerr << "If no bin index values are supplied, all bins are output.\n\n";
    exit(EXIT_FAILURE);
  }

  const string sgl_filename = argv[1];

  for (int arg = 2 ; arg < argc ; ++arg) {
    columns.push_back(atoi(argv[arg]));
  } 
  
  
  // Singles file object.
  ecat::ecat7::SinglesRatesFromSglFile singles_from_sgl;

  // Read the singles file.
  singles_from_sgl.read_singles_from_sgl_file(sgl_filename);

  const vector<double> times = singles_from_sgl.get_times();



  // Get total number of time slices.
  int num_time_slices = singles_from_sgl.get_num_time_slices();

  // Get scanner details and, from these, the number of singles units.
  const Scanner *scanner = singles_from_sgl.get_scanner_ptr();
  int total_singles_units = scanner->get_num_singles_units();
  
  
  // If no columns are set. Create a vector with all columns.
  if ( columns.size() == 0 ) {
    for (int singles_bin = 0 ; singles_bin < total_singles_units ; ++singles_bin) {
      columns.push_back(singles_bin);      
    }
  }


  // Print columns
  cout << "# Time    ";
  for (vector<int>::iterator col = columns.begin() ; col < columns.end() ; ++col) {
    cout << setw(9) << *col << " ";
  }
  cout << "\n";

  
  // Loop over all time slices.
  for (int time_slice = 0 ; time_slice < num_time_slices ; ++time_slice) {

    // Output time.
    cout << setw(8) << times[time_slice] << "  ";
    
    for (vector<int>::iterator col = columns.begin() ; col < columns.end() ; ++col) {

      if ( *col >= 0 && *col < total_singles_units ) {
        int val = singles_from_sgl.get_singles_rate(*col, time_slice);
      
        cout << setw(9) << val << " ";
      }
    }
  
    // Output the end of line.
    cout << "\n";

  }


  
  return EXIT_SUCCESS;
}
