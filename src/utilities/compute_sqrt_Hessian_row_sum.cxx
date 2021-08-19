/*
    Copyright (C) 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup utilities

  \brief Computes the Square Root of the Hessian Row Sum of the objective function

  \par Usage
  \verbatim
     compute_sqrt_Hessian_row_sum compute_sqrt_Hessian_row_sum.par
  \endverbatim

  This can be used to compute a spatially variant penalty strength (kappa) image. This utility uses methods proposed by:
  Tsai, Y.-J., Schramm, G., Ahn, S., Bousse, A., Arridge, S., Nuyts, J., Hutton, B. F., Stearns, C. W.,
  & Thielemans, K. (2020). Benefits of Using a Spatially-Variant Penalty Strength With Anatomical Priors
  in PET Reconstruction. IEEE Transactions on Medical Imaging, 39(1), 11–22. https://doi.org/10.1109/TMI.2019.2913889

  Based upon the value of "use approximate hessian" in the parameter file, either of two methods can be used to
  compute the spatially variant penalty strength, a.k.a. kappa.
  Both of the methods compute the square root of the Hessian row sum of the objective / likelihood function.
  1. \f[ \hat\kappa = \sqrt{ P^T \bigg( \frac{y}{ (P\lambda+a)^2 }  \bigg) P1 } \f], or
  2. \f[ \tilde\kappa = \sqrt{ P^T \bigg( \frac{1}{y} \bigg)P1 } \f]

  This utility may be used to generate the spatially variant penalty strength (kappa) images.

  \author Robert Twyman
*/

#include "stir/recon_buildblock/SqrtHessianRowSum.h"
#include <iostream>

START_NAMESPACE_STIR
static void print_usage_and_exit()
{
  std::cerr<<"\nThis executable computes the square root of the Hessian row sum of the objective function."
             "\n\nUsage: compute_sqrt_Hessian_row_sum compute_sqrt_Hessian_row_sum.par"
             "\n\n       (The example parameter file can be found in the samples folder.)" << std::endl;
  exit(EXIT_FAILURE);
}

END_NAMESPACE_STIR
USING_NAMESPACE_STIR
int
main(int argc, char *argv[])
{
  if (argc!=2)
    print_usage_and_exit();

  SqrtHessianRowSum<DiscretisedDensity<3,float>> SqrtHessianRowSumObject(argv[1]);
  SqrtHessianRowSumObject.set_up();
  SqrtHessianRowSumObject.process_data();
  return EXIT_SUCCESS;
}
