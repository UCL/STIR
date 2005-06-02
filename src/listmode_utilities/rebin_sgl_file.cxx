//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup utilities
  \brief Utility program scans the singles file looking for anomolous values

  \author Kris Thielemans
  \author Tim Borgeaud
  $Date$
  $Revision$
*/


#include "stir/data/SinglesRatesFromSglFile.h"
#include "stir/TimeFrameDefinitions.h"

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



void
usage(const char *progname) {
  cerr << "A program to rebin an sgl file.\n\n";
  cerr << "There are two ways to use this program.\n"; 
  cerr << "1) " << progname
       << " sgl_input_file sgl_output_file frame_end [frame_ends ...]\n\n";
  cerr << "2) " << progname
       << " -f frame_definition_file sgl_input_file sgl_output_file\n\n";
  cerr << "Frame end times are floating point numbers of seconds\n";
}



int 
main(int argc, char **argv)
{

  // Check arguments. 
  // Singles filename + optional output filename
  if (argc < 4 ) {
    usage(argv[0]);
    exit(EXIT_FAILURE);
  }

  

  string input_filename;
  string output_filename;
  vector<double> new_times; 

 

  // Check to see if -f was supplied as the first argument.
  if ( argv[1][0] == '-' ) {
    // Option supplied
    
    int arg_len = strlen(argv[1]);

    if ( arg_len != 2 || argv[1][1] != 'f' ) {
      
      for (int i = 1 ; i < arg_len ; ++i ) {
        if ( argv[1][i] != 'f' ) {
          cerr << "Unknown option " << argv[1][i] << endl;
        }
      }
      
      usage(argv[0]);
      exit(EXIT_FAILURE);
    }
    
    const string fdef_filename = argv[2];
    input_filename = argv[3];
    output_filename = argv[4];


    TimeFrameDefinitions time_frames(fdef_filename);
  
    double last_end = 0.0;

    // Create the new ending times by looping over the frames.
    for (unsigned int frame = 1 ; frame <= time_frames.get_num_frames() ; ++frame) {

      double frame_start = time_frames.get_start_time(frame);
      double frame_end = time_frames.get_end_time(frame);

      //cerr << "Start: " << frame_start << " End: " << frame_end << endl;
      
      if ( frame_start != last_end ) {
        //cerr << "Added frame at: " << frame_start << endl;
        // Add an additional frame that ends at the start of this frame.
        new_times.push_back(frame_start);
      }

      // Add the frame end.
      new_times.push_back(frame_end);
      
      last_end = frame_end;
    }
    

  } else {
    
    input_filename = argv[1];
    output_filename = argv[2];
    
    // Set up vector of new ending times.
    new_times = vector<double>(argc - 3);
    
    // Read frame end times
    for(int arg = 3 ; arg < argc ; arg++) {
      new_times[arg - 3] = atof(argv[arg]);
    }
    
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
