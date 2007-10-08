/*
  Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
/*
 * $Id$
 *
 *
 * \file
 * \ingroup test
 *
 * \brief testing GE_IO::niff
 *
 * \author T. Borgeaud
 *
 *
 * $Date$
 *
 * $Revision$
 *
 *
 */




#include "local/stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/is_null_ptr.h"
#include "stir/shared_ptr.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/Succeeded.h"

#include "stir/display.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace stir;


int main(int argc, char **argv) {

  
  // One argument for the input file
  if ( argc != 3 ) {
    std::cout << "Usage: " << argv[0] << " <input filename> <output filename>\n";
        exit(EXIT_FAILURE);
  }
  
  std::cout << "Reading file " << argv[1] << '\n';  

  bool file_exists = true;
  std::fstream fin;
  fin.open(argv[1],ios::in);
  if( !fin.is_open() )
    {
      std::cout<< "The  " << argv[1] << " file does not exist"<<endl;
      file_exists=false;
    }
  fin.close();

  shared_ptr<ParametricVoxelsOnCartesianGrid> parametric_image_sptr;
  if(file_exists)
    parametric_image_sptr =
      ParametricVoxelsOnCartesianGrid::read_from_file(argv[1]);      
  if(!file_exists)
    {
      string input_string;      
      std::cerr << "Please, give an input file\n" ;
      std::cin >> input_string ;
      std::cout << "Reading file " << input_string << '\n';      
      parametric_image_sptr =
      ParametricVoxelsOnCartesianGrid::read_from_file(input_string);
    }

  if (is_null_ptr(parametric_image_sptr))
    {
      warning("failed reading");
      return EXIT_FAILURE;
    }
    
  {
    const ParametricVoxelsOnCartesianGrid::SingleDiscretisedDensityType&
      single_density = parametric_image_sptr->construct_single_density(1);
    stir::display(single_density,single_density.find_max(), "First parameter");
  }
  std::cout << "Writing file " << argv[2] << '\n';  
  OutputFileFormat<ParametricVoxelsOnCartesianGrid>::default_sptr()->
    write_to_file(argv[2], *parametric_image_sptr);
  
  return EXIT_SUCCESS;
}




