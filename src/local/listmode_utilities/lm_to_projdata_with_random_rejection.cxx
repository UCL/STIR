//
//
/*!
  \file 
  \ingroup listmode

  \brief Program to bin listmode data to projection data using bootstrapping (uses stir::LmToProjDataWithRandomRejection)
 
  \author Kris Thielemans
  
  $Revision $
*/
/*
    Copyright (C) 2003- 2012, Hammersmith Imanet Ltd
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

#include "local/stir/listmode/LmToProjDataWithRandomRejection.h"

USING_NAMESPACE_STIR



int main(int argc, char * argv[])
{
  
  if (argc<1 && argc>3) {
    std::cerr << "Usage: " << argv[0] << " [par_file reject_if_above]]\n";
    exit(EXIT_FAILURE);
  }

  LmToProjDataWithRandomRejection<LmToProjData> 
    application(argc>=2 ? argv[1] : 0);
  if (argc==3)
    application.set_reject_if_above(float(atof(argv[2])));
  application.process_data();
  return EXIT_SUCCESS;
}

