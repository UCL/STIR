//
// $Id$
//
/*!

  \file
  \brief Utilitiy program scans the singles file looking for anomolous values

  \author Kris Thielemans
  \author Tim Borgeaud
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "local/stir/SinglesRatesFromSglFile.h"

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





int 
main (int argc, char **argv)
{

  // Check arguments. 
  // Singles filename + optional output filename
  if (argc < 4) {
    cerr << "A program to rebin an sgl file.\n\n";
    cerr << "Usage: " << argv[0] << 
      " sgl_input_file sgl_output_file frame_end [frame_ends ...]\n\n";
    cerr << "Frame end times are floating point numbers of seconds\n";
    exit(EXIT_FAILURE);
  }


  const string input_filename = argv[1];
  const string output_filename = argv[2];
  

  vector<double> new_times(argc - 3);

  // Read frame end times
  for(int arg = 3 ; arg < argc ; arg++) {
    new_times[arg - 3] = atof(argv[arg]);
  }
  
  
  
  // Singles file object.
  ecat::ecat7::SinglesRatesFromSglFile singles_from_sgl;

  // Read the singles file.
  singles_from_sgl.read_singles_from_sgl_file(input_filename);



  std::ofstream output;

  // If necessary, open the output file.
  output.open(output_filename.c_str(), std::ios_base::out);

  if (!output.good()) {
    error("Error opening output file\n");
  }


  // rebin
  singles_from_sgl.rebin(new_times);
  
  // Output.
  singles_from_sgl.write(output);
  
  
  return EXIT_SUCCESS;
}
