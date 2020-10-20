/*
    Copyright (C) 2002- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2020, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Lesser GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup utilities
  \brief Utility to extract image values along profiles (or another box-shape)
  \author Sanida Mustafovic
  \author Kris Thielemans

  \par Usage
  \code
   list_image_values [--LPS-output] [--csv] [--no-title-row] output_file_name input_image \\
       min_plane max_plane  min_col max_col min_row max_row
  \endcode
  Indices need to be in the STIR convention (plane starts from 0, col,row are centred around 0)

  Writes 4 columns to file, normally "plane row column value", unless \c --LPS-output is on, in which case it writes
  "L P S value".

  Output is separated by spaces, unless \c --csv is on, in which case commas are used.
*/

#include "stir/DiscretisedDensity.h"
#include "stir/IO/read_from_file.h"
#include <fstream>
#include <iomanip>



USING_NAMESPACE_STIR
const char * prog_name;

void print_usage_and_exit()
{
  std::cerr << "Usage:\n" << prog_name << " \\\n"
            << "    [ --LPS-output] [--csv] [--no-title-row] \\\n"
            << "    output_profile_name input_image min_plane max_plane  min_col max_col min_row max_row\n"
            << "Indices need to be in the STIR convention (plane starts from 0, col,row are centred around 0)\n"
            << "Writes 4 columns to file, normally \"plane row column value\", unless --LPS-output is on,\n"
            << "  in which case it writes \"L P S value\"\n"
            << "Output is separated by spaces, unless --csv is on, in which case commas are used.\n";
  
  exit(EXIT_FAILURE);
}

int main(int argc, const char *argv[])
{ 
  bool print_LPS = false;
  bool print_csv = false;
  bool print_first_line = true;
  prog_name = argv[0];

  while (argc>1 && (strncmp(argv[1],"--",2)==0))
    {
      if (strcmp(argv[1],"--LPS-output")==0)
        print_LPS=true;
      else if ((strcmp(argv[1],"--csv")==0) || (strcmp(argv[1],"--CSV")==0))
        print_csv=true;
      else if (strcmp(argv[1],"--no-title-row")==0)
        print_first_line = false;
      else
        print_usage_and_exit();
      ++argv; --argc;
    }
  
  if (argc!= 9)
    print_usage_and_exit();

  const char * const output_filename = argv[1];
  const char * const input_filename = argv[2];

    
  shared_ptr<DiscretisedDensity<3,float> > 
    input_image_sptr(read_from_file<DiscretisedDensity<3,float> >(input_filename));
  const DiscretisedDensity<3,float>& input_image =
    *input_image_sptr;
  
  const int min_plane_num = atoi(argv[3]);
  const int max_plane_num = atoi(argv[4]);    
  const int min_column_num = atoi(argv[5]);
  const int max_column_num = atoi(argv[6]);    
  const int min_row_num   = atoi(argv[7]);
  const int max_row_num   = atoi(argv[8]);  
    
  std::ofstream  profile_file(output_filename);

  using std::setw;

  const char separator = print_csv ? ',' : ' ' ;


  if (print_first_line)
    {
      if (print_LPS)
        profile_file << setw(8) << "L" << separator << setw(8) << "P" 
                     << separator << setw(8) << "S" << separator << setw(10) << "value" <<'\n';
      else
        profile_file << setw(8) << "plane" << separator << setw(8) << "row" 
                     << separator << setw(8) << "column" << separator << setw(10) << "value" <<'\n';
    }

  for (int plane = min_plane_num;plane <=max_plane_num;plane++) 
    for (int row = min_row_num;row <=max_row_num;row++)
      for (int column = min_column_num;column<=max_column_num;column++)
      {
        const BasicCoordinate<3,int> index = make_coordinate(plane, row, column);
        if (print_LPS)
          {
            const CartesianCoordinate3D<float> LPS =
              input_image.get_LPS_coordinates_for_indices(index);
            profile_file << setw(8) << LPS[3] << separator << setw(8) << LPS[2]
                         << separator << setw(8) << LPS[1];
          }
        else
          {
            profile_file << setw(8) << plane << separator << setw(8) << row 
                         << separator << setw(8) << column;
          }
        profile_file << separator << setw(10) << input_image.at(index) << '\n';
      }
  
  return EXIT_SUCCESS;
}

