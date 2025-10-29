//
//
/*
    Copyright (C) 2004- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2021, University College London
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \brief Utility program that prints out singles values from an RDF file.

  \author Kris Thielemans
*/

#include "stir/data/SinglesRatesFromGEHDF5.h"
#include "stir/stream.h"
#include "stir/IndexRange3D.h"
#include <iostream>
#include <string>

USING_NAMESPACE_STIR

int
main(int argc, char** argv)
{

  // Check arguments.
  // Singles filename
  if (argc != 2)
    {
      std::cerr << "Program to print out values from a singles file.\n\n";
      std::cerr << "Usage: " << argv[0] << " rdf_filename \n\n";
      exit(EXIT_FAILURE);
    }

  const std::string rdf_filename = argv[1];
  // Singles file object.
  GE::RDF_HDF5::SinglesRatesFromGEHDF5 singles(rdf_filename);

  // Get total number of frames
  // int num_frames = singles.get_num_frames();

  // Get scanner details and, from these, the number of singles units.
  const Scanner& scanner = *singles.get_scanner_ptr();
  const TimeFrameDefinitions time_def = singles.get_time_frame_definitions();

  // print time-frame info
  {
    std::cout << "\nTime frame info (in secs): (duration, start)\n";
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(2);

    const bool units_secs = true;
    const int units = units_secs ? 1 : 1000;
    const std::string units_string = units_secs ? " secs" : " millisecs";

    for (unsigned frame_num = 1; frame_num <= time_def.get_num_time_frames(); ++frame_num)
      {
        const double start_frame = time_def.get_start_time(frame_num);
        const double end_frame = time_def.get_end_time(frame_num);
        const double frame_duration = end_frame - start_frame;

        std::cout << "(" << frame_duration * units << ", " << start_frame * units << ')';
      }
    std::cout << std::endl;
  }

  Array<3, float> singles_arr(IndexRange3D(
      time_def.get_num_time_frames(), scanner.get_num_axial_singles_units(), scanner.get_num_transaxial_singles_units()));

  for (int time_frame_num = 1; static_cast<unsigned>(time_frame_num) <= time_def.get_num_time_frames(); ++time_frame_num)
    {
      for (int ax = 0; ax < scanner.get_num_axial_singles_units(); ++ax)
        {
          for (int transax = 0; transax < scanner.get_num_transaxial_singles_units(); ++transax)
            {
              const int singles_bin_index = scanner.get_singles_bin_index(ax, transax);
              singles_arr[time_frame_num - 1][ax][transax] = singles.get_singles(
                  singles_bin_index, time_def.get_start_time(time_frame_num), time_def.get_end_time(time_frame_num));
            }
        }
    }

  std::cout << singles_arr;

  return EXIT_SUCCESS;
}
