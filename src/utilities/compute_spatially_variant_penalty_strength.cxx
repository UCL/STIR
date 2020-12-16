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
     compute_spatially_variant_penalty_strength spatially_variant_penalty_strength.par
  \endverbatim

  Computes a spatially variant penalty strength (kappa) image. This utility uses methods proposed by:
  Tsai, Y.-J., Schramm, G., Ahn, S., Bousse, A., Arridge, S., Nuyts, J., Hutton, B. F., Stearns, C. W.,
  & Thielemans, K. (2020). Benefits of Using a Spatially-Variant Penalty Strength With Anatomical Priors
  in PET Reconstruction. IEEE Transactions on Medical Imaging, 39(1), 11–22. https://doi.org/10.1109/TMI.2019.2913889

  Based upon the value of "use approximate hessian" in the parameter file, either of two methods can be used to
  compute the spatially variant penalty strength, a.k.a. kappa.
  Both of the methods compute the Hessian row sum of the objective / likelihood function.
  1. \f[ \hat\kappa = \sqrt{ P^T \bigg( \frac{y}{ (P\lambda+a)^2 }  \bigg) P1 } \f], or
  2. \f[ \tilde\kappa = \sqrt{ P^T \bigg( \frac{1}{y} \bigg)P1 } \f]

  \author Robert Twyman
*/

#include "stir/recon_buildblock/SqrtHessianRowSum.h"

using std::cerr;
using std::cout;
using std::endl;

START_NAMESPACE_STIR
static void print_usage_and_exit()
{
  //todo:update usage
  std::cerr<<"\nThis executable computes a spatially variant penalty strength dependant on a parameter file."
             "\n\nUsage: compute_spatially_variant_penalty_strength spatially_variant_penalty_strength.par"
             "\n\nAn example parameter file can be found in the samples folder." << std::endl;
  exit(EXIT_FAILURE);
}

END_NAMESPACE_STIR
USING_NAMESPACE_STIR
int
main(int argc, char *argv[])
{
  if (argc!=2)
    print_usage_and_exit();

  SqrtHessianRowSum<DiscretisedDensity<3,float>> kappa_computer;
  kappa_computer.parse(argv[1]);
  kappa_computer.process_data();
  return EXIT_SUCCESS;
}
