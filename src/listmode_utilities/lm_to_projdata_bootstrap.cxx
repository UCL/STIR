//
//
/*!
  \file
  \ingroup listmode

  \brief Program to bin listmode data to projection data using bootstrapping (uses stir::LmToProjDataBootstrap)

  \author Kris Thielemans

  $Revision $
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/

#include "stir/listmode/LmToProjDataBootstrap.h"

USING_NAMESPACE_STIR

int
main(int argc, char* argv[]) {

  if (argc < 1 && argc > 3) {
    std::cerr << "Usage: " << argv[0] << " [par_file [seed]]\n";
    exit(EXIT_FAILURE);
  }

  // clumsy way of having extra argument
  if (argc == 3) {
    LmToProjDataBootstrap<LmToProjData> application(argc >= 2 ? argv[1] : 0, atoi(argv[2]));
    application.process_data();
  } else {
    LmToProjDataBootstrap<LmToProjData> application(argc == 2 ? argv[1] : 0);
    application.process_data();
  }
  return EXIT_SUCCESS;
}
