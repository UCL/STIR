/*
 Copyright (C) 2011 - 2013, King's College London
 This file is part of STIR.

 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.3 of the License, or
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

 \brief This program inverts x y or z axis. It is also able to remove nan from the image.
 \author Daniel Deidda
 */
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/spatial_transformation/InvertAxis.h"

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if(argc!=4) {
    std::cerr << "Usage: " << argv[0]
              << " <axis name>  <output filename> <input filename>\n"
              << " if <axis name>= nan the utility will substitute every nan value with zero  ";
    exit(EXIT_FAILURE);
  }
  InvertAxis invert;
  char const * const axis_name = argv[1];
  char const * const output_filename_prefix = argv[2];
  char const * const  input_filename= argv[3];
  std::string name(axis_name);
 // const int num_planes = atoi(argv[3]);
std::cout<<" the axis to invert is "<< name<<std::endl;
const std::auto_ptr<DiscretisedDensity<3,float> > image_aptr(DiscretisedDensity<3,float>::read_from_file(input_filename));
const std::auto_ptr<DiscretisedDensity<3,float> > out_image_aptr(image_aptr->clone());

DiscretisedDensity<3,float>& image = *image_aptr;
DiscretisedDensity<3,float>& out_image = *out_image_aptr;

  invert.invert_axis(out_image,
              image,
              name);

  const Succeeded res = OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(output_filename_prefix, out_image);

  return res==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
