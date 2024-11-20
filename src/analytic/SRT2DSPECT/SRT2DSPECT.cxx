//
//
/*!
  \file
  \ingroup analytic
  \brief Main program for SRT2DSPECT reconstruction
  \author Dimitra Kyriakopoulou
  \author Kris Thielemans
*/
/*
    Copyright (C) 2024, University College London

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/analytic/SRT2DSPECT/SRT2DSPECTReconstruction.h"
#include "stir/Succeeded.h"
#ifndef PARALLEL
#  define Main main
#else
#  define Main master_main
#endif

USING_NAMESPACE_STIR

int
Main(int argc, char** argv)
{
  SRT2DSPECTReconstruction reconstruction_object(argc > 1 ? argv[1] : "");

  return reconstruction_object.reconstruct() == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
