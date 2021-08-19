/*
 Copyright (C) 2011 - 2013, King's College London
 This file is part of STIR.

 SPDX-License-Identifier: Apache-2.0

 See STIR/LICENSE.txt for details
 */
/*!
 \file
 \ingroup utilities

 \brief This program inverts x y or z axis.
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
              << " <axis name> has to be x, y, z\n"
              << "\nWARNING: this will reorder the voxel values without adjusting the geometric information\n";
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
