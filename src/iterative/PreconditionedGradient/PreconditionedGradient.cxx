//
/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup PreconditionedGradient
  \brief main() for stir::PreconditionedGradientReconstruction

  \author Manil Sabeur
*/
#include "stir/PreconditionedGradient/PreconditionedGradientReconstruction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/distributable_main.h"

USING_NAMESPACE_STIR

#ifdef STIR_MPI
int
stir::distributable_main(int argc, char** argv)
#else
int
main(int argc, char** argv)
#endif
{
  PreconditionedGradientReconstruction<DiscretisedDensity<3, float>> reconstruction_object(argc > 1 ? argv[1] : "");

  return reconstruction_object.reconstruct() == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
