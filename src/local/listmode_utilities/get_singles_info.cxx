//
// $Id$
//
/*!

  \file
  \brief Utilitiy program that lists the singles per bucket in a frame to a text file

  \author Kris Thielemans
  \author Katie Dinelle
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/TimeFrameDefinitions.h"
#include "local/stir/SinglesRatesFromSglFile.h"

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

  // read singles
  ecat::ecat7::SinglesRatesFromSglFile singles_from_sgl;
  const Array<3,float> array_sgl = 
     singles_from_sgl.read_singles_from_sgl_file (sgl_filename);
  const vector<double> times = 
    singles_from_sgl.get_times();

  // read time frame definitions
  const TimeFrameDefinitions frame_defs(frame_defs_filename);

  if (frame_num < 1 || frame_num>frame_defs.get_num_frames())
    error("Incorrect frame number\n");

  // open output file
  std::ofstream output(output_filename.c_str());
  if (!output.good())
    error("Error opening output file\n");

  // compute total singles in this frame
  Array<2,float> singles_in_this_frame(array_sgl[0].get_index_range());
  unsigned how_many_entries = 0;
  // find entry in start of frame
  int entry_num=array_sgl.get_min_index(); 
  for (;
       static_cast<std::size_t>(entry_num+1)<times.size() && entry_num<= array_sgl.get_max_index(); 
       ++entry_num)
    {
      const double current_time = times[entry_num+1];
      if (current_time >= frame_defs.get_start_time(frame_num) )
	break;
    }
  // now add singles in this frame
  for (;
       static_cast<std::size_t>(entry_num)<times.size() && entry_num<= array_sgl.get_max_index(); 
       ++entry_num)
    {
      const double current_time = times[entry_num];
      if (current_time > frame_defs.get_end_time(frame_num) )
	break;
      singles_in_this_frame += array_sgl[entry_num];
      ++how_many_entries;
    }
  // compute average rate
  singles_in_this_frame /= static_cast<float>(how_many_entries);

  // now write to file
  {
    Array<2,float>::full_iterator array_iter  = 
      singles_in_this_frame.begin_all();
    int singles_num = 0;
    while (array_iter != singles_in_this_frame.end_all())
      {
	output << singles_num << "  " << *array_iter << '\n';
	++singles_num;
	++array_iter;
      }
  }
  
  return EXIT_SUCCESS;
}
