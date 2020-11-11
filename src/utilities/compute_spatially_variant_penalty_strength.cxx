/*
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

  \brief Computes the spatially variant penalty strength

  \par Usage
  \verbatim
     compute_spatially_variant_penalty_strength template_proj_data
  \endverbatim

  Computes a spatially variant penalty strength, either using:


  See: Tsai, Y.-J., Schramm, G., Ahn, S., Bousse, A., Arridge, S., Nuyts, J., Hutton, B. F., Stearns, C. W.,
    & Thielemans, K. (2020). Benefits of Using a Spatially-Variant Penalty Strength With Anatomical Priors
    in PET Reconstruction. IEEE Transactions on Medical Imaging, 39(1), 11–22. https://doi.org/10.1109/TMI.2019.2913889
   for more details

  \author Robert Twyman
*/

#include "stir/recon_buildblock/Hessian_row_sum.h"

using std::cerr;
using std::cout;
using std::endl;

START_NAMESPACE_STIR
static void print_usage_and_exit()
{
  //todo:update usage
  std::cerr<<"\nUsage: compute_spatially_variant_penalty_strength template_proj_data\n";
  exit(EXIT_FAILURE);
}

END_NAMESPACE_STIR
USING_NAMESPACE_STIR
int
main(int argc, char *argv[])
{
  if (argc!=2)
    print_usage_and_exit();

  Hessian_row_sum<DiscretisedDensity<3,float>> kappa_computer;
  kappa_computer.parse(argv[1]);
  kappa_computer.process_data();
  return EXIT_SUCCESS;
}
