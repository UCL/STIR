//
// $Id$
//
/*
    Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
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
  \ingroup test
 
  \brief tests for stir::InputFileFormat

  \author Kris Thielemans 
  $Date$
  $Revision$

*/
#include "stir/DiscretisedDensity.h"
#include "stir/IO/InputFileFormatRegistry.h"


#include <string>
int main(int argc, char **argv)
{
  using namespace stir;

  const std::string filename = argv[1];
  /*
  InputFileFormatRegistry<DiscretisedDensity<3,float> > & registry =
  *InputFileFormatRegistry<DiscretisedDensity<3,float> >::default_sptr();
  const InputFileFormat<DiscretisedDensity<3,float> >& factory =
    registry.find_factory(filename);
  
  std::auto_ptr<DiscretisedDensity<3,float> > density_aptr =
    factory.read_from_file(filename);
  */
  std::auto_ptr<DiscretisedDensity<3,float> > density_aptr =
    read_from_file<DiscretisedDensity<3,float> > (filename);
  std::auto_ptr<DiscretisedDensity<3,float> > density2_aptr =
    std::auto_ptr<DiscretisedDensity<3,float> >
    (DiscretisedDensity<3,float>::read_from_file(filename));

  *density_aptr -= *density2_aptr;
  std::cout << " diff max " << density_aptr->find_max()
	    << " min " << density_aptr->find_min()
	    << '\n';

  return EXIT_SUCCESS;
}
