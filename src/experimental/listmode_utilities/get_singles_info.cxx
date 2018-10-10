//
//
/*!

  \file
  \ingroup utilities
  \brief Utility program that lists the singles per bucket in a frame to a text file

  \author Kris Thielemans
  \author Katie Dinelle
  \author Tim Borgeaud
*/
/*
    Copyright (C) 2004- 2005, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "stir/TimeFrameDefinitions.h"
#include "stir/data/SinglesRatesFromSglFile.h"

#include <string>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::string;
#endif

USING_NAMESPACE_STIR




int 
main (int argc, char* argv[])
{

  if (argc!=5)
    {
      cerr << "Usage: " << argv[0] << " output_filename sgl_filename fdef_filename frame_num\n";
      exit(EXIT_FAILURE);
    }

  const string output_filename = argv[1];
  const string sgl_filename = argv[2];
  const string frame_defs_filename = argv[3];
  const unsigned frame_num = atoi(argv[4]);


  // SinglesRatesFromSglFile object.
  ecat::ecat7::SinglesRatesFromSglFile singles_from_sgl;
  
  // Read in the singles file.
  singles_from_sgl.read_singles_from_sgl_file(sgl_filename);
  

  // read time frame definitions
  const TimeFrameDefinitions frame_defs(frame_defs_filename);

  if (frame_num < 1 || frame_num > frame_defs.get_num_frames()) {
    error("Incorrect frame number\n");
  }

  // open output file
  std::ofstream output(output_filename.c_str());
  if (!output.good()) {
    error("Error opening output file\n");
  }

  // Retrieve start and end times for this frame.
  double start_time = frame_defs.get_start_time(frame_num);
  double end_time = frame_defs.get_end_time(frame_num);
  
  // Create a new FrameSinglesRates object for the frame.
  FrameSinglesRates frame_singles_rates = 
    singles_from_sgl.get_rates_for_frame(start_time, end_time);

  
  // Get scanner details and, from these, the number of singles units.
  const Scanner *scanner = frame_singles_rates.get_scanner_ptr();
  int total_singles_units = scanner->get_num_singles_units();
  
  // Now write to file
  for(int singles_bin = 0 ; singles_bin < total_singles_units ; ++singles_bin) {
    output << singles_bin << "  " 
           << frame_singles_rates.get_singles_rate(singles_bin) << '\n';
  }
  
  return EXIT_SUCCESS;
}
