/*
  Copyright (C) 2022, University College London
  This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup utilities

  \brief A utility to generate images consisting of uniform objects added/subtracted together

  *See GenerateImage documentation for parameter file example and usage documentation.
*/
#include "stir/PatientPosition.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Succeeded.h"
#include <iostream>
#include "stir/Shape/GenerateImage.h"

/************************ main ************************/

USING_NAMESPACE_STIR
int main(int argc, char * argv[])
{
  
  if ( argc!=2) {
    std::cerr << "Usage: " << argv[0] << " par_file\n";
    exit(EXIT_FAILURE);
  }
  GenerateImage application(argc==2 ? argv[1] : 0);
  Succeeded success = application.compute();
  application.save_image();

  return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
