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
  \brief Utility program that prints out values from an sgl file.

  \author Kris Thielemans
  \author Tim Borgeaud
  $Date$
  $Revision$
*/


#include "stir/data/SinglesRatesFromECAT7.h"


#include <fstream>
#include <iomanip>
#include <string>
#include <vector>


#ifndef STIR_NO_NAMESPACES
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

  const string ecat7_filename = argv[1];

  for (int arg = 2 ; arg < argc ; ++arg) {
    columns.push_back(atoi(argv[arg]));
  } 
  
  
  // Singles file object.
  ecat::ecat7::SinglesRatesFromECAT7 singles_from_ecat7;


  // Read the singles file.
  singles_from_ecat7.read_singles_from_file(ecat7_filename);


  // Get total number of frames
  int num_frames = singles_from_ecat7.get_num_frames();
  
  // Get scanner details and, from these, the number of singles units.
  const Scanner *scanner = singles_from_ecat7.get_scanner_ptr();
  int total_singles_units = scanner->get_num_singles_units();
  
  
  // If no columns are set. Create a vector with all columns.
  if ( columns.size() == 0 ) {
    for (int singles_bin = 0 ; singles_bin < total_singles_units ; ++singles_bin) {
      columns.push_back(singles_bin);      
    }
  }


  // Print columns
  cout << "# Frame  Frame time         ";
  for (vector<int>::iterator col = columns.begin() ; col < columns.end() ; ++col) {
    cout << setw(9) << *col << " ";
  }
  cout << "\n";

  
  // Loop over all frames.
  for (int frame = 1 ; frame <= num_frames ; ++frame) {
    
    // Ouput frame number, start and end times.
    cout << setw(2) << frame << "  " 
         << setw(8) << singles_from_ecat7.get_time_frame_definitions().get_start_time(frame) << " to  "
         << setw(8) << singles_from_ecat7.get_time_frame_definitions().get_end_time(frame) << "   ";

    for (vector<int>::iterator col = columns.begin() ; col < columns.end() ; ++col) {
      
      if ( *col >= 0 && *col < total_singles_units ) {
        float val = singles_from_ecat7.get_singles_rate(*col, frame);
        
        cout << setw(9) << val << " ";
      }
    }
  
    // Output the end of line.
    cout << endl;
    
  }


  
  return EXIT_SUCCESS;
}
