//
//
/*
    Copyright (C) 2003, Hammersmith Imanet Ltd
    Copyright (C) 2012-07-01 - 2012, Kris Thielemans
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2.0 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup utilities
 
  \brief  This program writes a PGM bitmap for an image (preliminary)

  Run to get a usage message.

  \author Kris Thielemans


  \warning It only supports stir::VoxelsOnCartesianGrid type of images.
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/read_from_file.h"
#include "stir/round.h"
#include "stir/IndexRange3D.h"
#include "stir/Coordinate2D.h"
#include "stir/shared_ptr.h"
#include "stir/error.h"

#include <cstdio>
#include <fstream>
#include <algorithm>

START_NAMESPACE_STIR
/* helper functions.
   Currently copied from manip_image.
*/
static VoxelsOnCartesianGrid<float> 
transpose_13(const VoxelsOnCartesianGrid<float> & image)
{
  CartesianCoordinate3D<float> origin = image.get_origin();
  std::swap(origin.x(), origin.z());
  CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
  std::swap(voxel_size.x(), voxel_size.z());
  VoxelsOnCartesianGrid<float> 
    out(IndexRange3D(image.get_min_x(),image.get_max_x(),
                     image.get_min_y(),image.get_max_y(),
                     image.get_min_z(),image.get_max_z()),
        origin,
        voxel_size);
  for (int x=image.get_min_x(); x<=image.get_max_x(); ++x)
    for (int y=image.get_min_y(); y<=image.get_max_y(); ++y)
      for (int z=image.get_min_z(); z<=image.get_max_z(); ++z)
        out[x][y][z] = image[z][y][x];
  return out;
}

static VoxelsOnCartesianGrid<float> 
transpose_12(const VoxelsOnCartesianGrid<float> & image)
{
  CartesianCoordinate3D<float> origin = image.get_origin();
  std::swap(origin.y(), origin.z());
  CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
  std::swap(voxel_size.y(), voxel_size.z());
  VoxelsOnCartesianGrid<float> 
    out(IndexRange3D(image.get_min_y(),image.get_max_y(),
                     image.get_min_z(),image.get_max_z(),
                     image.get_min_x(),image.get_max_x()),
        origin,
        voxel_size);
  for (int y=image.get_min_y(); y<=image.get_max_y(); ++y)
    for (int z=image.get_min_z(); z<=image.get_max_z(); ++z)
      for (int x=image.get_min_x(); x<=image.get_max_x(); ++x)
        out[y][z][x] = image[z][y][x];
  return out;
}

template <typename elemT>
static 
void 
write_pgm (const std::string& filename,
	   const Array<2,elemT>& plane,
	   const double min_threshold, const double max_threshold)
{
  if (plane.get_length() == 0)
    return;
  
  Coordinate2D<int> min_indices;
  Coordinate2D<int> max_indices;

  if (!plane.get_regular_range(min_indices, max_indices))
  {
    warning("write_pgm: can only display 'regular' arrays. Returning.\n");
    return;
  }

  FILE *pgm = fopen ( filename.c_str() , "wb");
  if (pgm == NULL)
  {
    error("Error opening file %s for output to PGM.",filename.c_str());
  }
  const int pgm_max = 255;
  {
    const int X = max_indices[2] - min_indices[2] + 1;    
    const int Y = (max_indices[1] - min_indices[1] + 1);
    fprintf ( pgm, "P5\n#created by STIR\n%d %d\n%d\n", X , Y, pgm_max);
  }
  
  for ( int y = min_indices[1]; y <= max_indices[1]; y++)
    {
      for ( int x = min_indices[2]; x <= max_indices[2]; x++)
	{
	  double val = static_cast<double>(plane[y][x]);
	  if (val>max_threshold) 
	    val=max_threshold;
	  else if (val<min_threshold) 
	    val=min_threshold;
	  // now to pgm range
	  val = (val - min_threshold) / (max_threshold - min_threshold) * pgm_max;
	  fprintf ( pgm, "%c", static_cast<unsigned char>(stir::round(val)) );
	}			  
    }
  fclose ( pgm);
}

END_NAMESPACE_STIR

void print_usage_and_exit(const std::string& program_name)
{
  std::cerr<< "Usage: " << program_name << "\n\t"
	   << "[--min min_value] [--max max_value] \\\n\t" 
	   << "[--orientation t|c|s] [--slice_index idx] \\\n\t" 
	   << "output_filename.pgm input_filename \n"
	   << "min_value default to 0, max_value to max in image\n"
	   <<"oritentation defaults to transverse\n"
	   << "slice index is zero-based and defaults to the middle of the image (using rounding)\n"; 
  exit(EXIT_FAILURE);
}

int 
main(int argc, char **argv)
{
  const char * const program_name = argv[0];
  // skip program name
  --argc;
  ++argv;

  double min_threshold=0.;
  double max_threshold=-1.;
  char orientation = 't';
  int slice_index = -1;

    // first process command line options
  while (argc>0 && argv[0][0]=='-' && argc>=2)
    {
      if (strcmp(argv[0], "--max")==0)
	{
	  max_threshold = atof(argv[1]);
	} 
      else if (strcmp(argv[0], "--min")==0)
	{
	  min_threshold = atof(argv[1]);
	} 
      else if (strcmp(argv[0], "--orientation")==0)
	{
	  orientation = argv[1][0];
	} 
      else if (strcmp(argv[0], "--slice_index")==0)
	{
	  slice_index = atoi(argv[1]);
	} 
      else
	{
	  std::cerr << "Unknown option: " <<argv[0] << '\n';
	  print_usage_and_exit(argv[0]);
	}
      argc-=2; argv+=2;
    }

  if(argc!=2)
    {
      print_usage_and_exit(program_name);
   }

  const std::string filename = argv[0];
  const std::string input_filename = argv[1];

  stir::VoxelsOnCartesianGrid<float> image(
	dynamic_cast<stir::VoxelsOnCartesianGrid<float> &>
	(* stir::read_from_file<stir::DiscretisedDensity<3,float> >(input_filename)));

  if (max_threshold < min_threshold)
    max_threshold = image.find_max();

  switch (orientation)
    {
    case 't': case 'T':
      // transverse, nothing to do at the moment
      break;
    case 's': case 'S':
      // sagital
      image=stir::transpose_13(image);
      break;
    case 'c': case 'C':
      // coronal
      image=stir::transpose_12(image);
      break;
    default:
      stir::error("Unsupported orientation %d, has to be t,s, or c", orientation);
    }

  if (slice_index<0)
    slice_index = stir::round(image.get_length()/2.);
  else if (slice_index >=  image.get_length())
    stir::error("Requested slice index too large");

  stir::write_pgm (filename,
		   image[slice_index + image.get_min_index()],
		   min_threshold, max_threshold);

  return EXIT_SUCCESS;
}
