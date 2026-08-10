/*
    Copyright (C) 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup main_programs
  \brief main() for stir::OSMAPOSLReconstruction on parametric images

  \author Nicolas A Karakatsanis
*/
#include "stir/Succeeded.h"
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/recon_buildblock/distributable_main.h"

#ifdef STIR_MPI
int
stir::distributable_main(int argc, char** argv)
#else
int
main(int argc, char** argv)
#endif
{
  USING_NAMESPACE_STIR

  HighResWallClockTimer t;
  t.reset();
  t.start();

  OSMAPOSLReconstruction<ParametricVoxelsOnCartesianGrid> reconstruction_object(argc > 1 ? argv[1] : "");

  if (reconstruction_object.reconstruct() == Succeeded::yes)
    {
      t.stop();
      std::cout << "Total Wall clock time: " << t.value() << " seconds" << std::endl;
      return EXIT_SUCCESS;
    }
  else
    {
      t.stop();
      return EXIT_FAILURE;
    }
}
