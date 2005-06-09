//
// $Id$
//
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
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
/*!
  \file
  \ingroup utilities

  \brief A utility that lists size info on the projection data on stdout.

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/is_null_ptr.h"
#include <iostream> 

USING_NAMESPACE_STIR

int main(int argc, char *argv[])
{ 
  
  if(argc!=2) 
  {
    std::cerr<<"Usage: " << argv[0] << " projdata_file\n";
    return EXIT_FAILURE;

  }

  shared_ptr<ProjData> projdata_ptr = 
  ProjData::read_from_file(argv[1]);

  if (is_null_ptr(projdata_ptr))
    {
      warning("Could not read %s", argv[1]);
      return EXIT_FAILURE;
    }
  std::cout << "Info for file " << argv[1] << '\n';
  std::cout << projdata_ptr->get_proj_data_info_ptr()->parameter_info() << '\n';

  return EXIT_SUCCESS;
}
