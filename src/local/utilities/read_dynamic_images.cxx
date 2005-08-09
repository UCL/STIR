//
// $Id$
//
/*!
  \file
  \ingroup utilities
  \brief Reading Dynamic Images
  \author Charalampos Tsoumpas
  $Date$
  $Revision$
*/
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
 /* This program reads dynamic images given from a file. 
 */

#include "stir/IO/read_data_1d.h"
#include "stir/IO/read_data.h"
#include "stir/Array.h"
#include "stir/ByteOrder.h"
#include <cstring>
#include <iostream>

int main(int argc, char *argv[])
{ 
  USING_NAMESPACE_STIR
  
  if (argc<2 || argc>3)
    {
      std::cerr << "Usage:" << argv[0] << "\n"
	   << "\t[dynamic_image_filename]\n" 
	   << "\t[byte_order]\n";
      return EXIT_FAILURE;            
    }       
  BasicCoordinate<2,float> min_index ;
  BasicCoordinate<2,float> max_index ;

  IndexRange<2> this_range(1,4,1,4);
  Array<2,float> dyn_image(this_range) ;

  ByteOrder::Order this_order = ByteOrder::little_endian; //argc!=3 ? argv[2] : ByteOrder::little_endian ; // Convert string into OrderType!!
  ByteOrder byte_order(this_order);
  std::string dyn_image_filename(argv[1]);
  std::ifstream dyn_image_stream(dyn_image_filename.c_str()); 
   dyn_image.read_data(dyn_image_stream,ByteOrder::native);

  std::cout << dyn_image[1][1] << "\n" ;
  if (read_data(dyn_image_stream,dyn_image,byte_order)==Succeeded::no)
      error("error reading image stream");

      /*
  PlasmaData::const_iterator cur_iter;
  for (cur_iter=testing_plasma_data.begin(); 
       cur_iter!=testing_plasma_data.end() ; ++cur_iter)
    std::cout << (*cur_iter).get_time_in_s() << " "<< (*cur_iter).get_counts_in_kBq() << std::endl ;
      */
  return EXIT_SUCCESS;
}

