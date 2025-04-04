/*
 Copyright (C) 2009 - 2013, King's College London
 This file is part of STIR.

 SPDX-License-Identifier: Apache-2.0

 See STIR/LICENSE.txt for details
 */
/*!
 \file
 \ingroup utilities
 \brief This program shifts the origin of an image.
 \author Charalampos Tsoumpas
 */
#include "stir/DiscretisedDensity.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/Succeeded.h"

using std::cerr;

USING_NAMESPACE_STIR

int
main(int argc, char** argv)
{
  if (argc < 3 || argc > 6)
    {
      cerr << "Usage: " << argv[0] << " <output image filename> <input image filename> [x-shift] [y-shift] [z-shift] ] ] ]\n"
           << "all shifts are in mm and defaults are set to 0mm\n";
      exit(EXIT_FAILURE);
    }

  // get parameters from command line
  char const* const output_filename = argv[1];
  char const* const input_filename = argv[2];
  const float x_shift_in_mm = (argc > 3) ? static_cast<float>(atof(argv[3])) : 0;
  const float y_shift_in_mm = (argc > 4) ? static_cast<float>(atof(argv[4])) : 0;
  const float z_shift_in_mm = (argc > 5) ? static_cast<float>(atof(argv[5])) : 0;

  // read image
  shared_ptr<DiscretisedDensity<3, float>> density_sptr(DiscretisedDensity<3, float>::read_from_file(input_filename));
  shared_ptr<DiscretisedDensity<3, float>> out_density_sptr(density_sptr->clone());
  const Coordinate3D<float> origin = density_sptr->get_origin();
  const Coordinate3D<float> origin_shift(z_shift_in_mm, y_shift_in_mm, x_shift_in_mm);
  out_density_sptr->set_origin(origin + origin_shift);

  // write image
  Succeeded res
      = OutputFileFormat<DiscretisedDensity<3, float>>::default_sptr()->write_to_file(output_filename, *out_density_sptr);
  return res == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
