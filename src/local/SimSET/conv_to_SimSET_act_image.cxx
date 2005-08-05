//
// $Id$
//

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
/*!
\file
\ingroup utilities
\brief Convert a transmission image into SimSET attenuation file input.

  \author Pablo Aguiar
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
	$Date$
	$Revision$
	
	  \par Usage:
	  \code
	  conv_to_SimSET_image [SimSET_image_filename][original_image][multiplicated constant]
	  Output: "SimSET_image_filename".hv AND "SimSET_image_filename".v "multiplicated constant"
	  
		This is a utility program converts a image into SimSET activity input file.
 
		
		See the SimSET documentation for more details.
  
			  \endcode	  
*/
#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/IO/interfile.h"
#include "stir/Succeeded.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include <iostream> 
/***********************************************************/     
int main(int argc, char *argv[])                                  
{         
	USING_NAMESPACE_STIR
		if (argc!=4)
		{
			std::cerr << "Usage:" << argv[0]  
				<< " SimSET_image_filename original_image multiplicated constant\n";    		 	  
			return EXIT_FAILURE;
		}
		shared_ptr< DiscretisedDensity<3,float> >  input_image_sptr = 
			DiscretisedDensity<3,float>::read_from_file(argv[2]);
		string output_image_filename(argv[1]);
                const float constant = argc>=4 ? atof(argv[3]) : 10;
		shared_ptr<DiscretisedDensity<3,float> > output_image_sptr = input_image_sptr->clone();		
		//Array<3,unsigned char>::full_iterator out_iter = output_image.begin_all();

		DiscretisedDensity<3,float>::full_iterator out_iter = output_image_sptr->begin_all();
		DiscretisedDensity<3,float>::const_full_iterator in_iter = input_image_sptr->begin_all_const();
		while( in_iter != input_image_sptr->end_all_const())
		{
       			*out_iter = (*in_iter)*constant;
			++in_iter; ++out_iter;
		}

				
		// write to file as 1 byte without changing the scale
		Succeeded success = 
			write_basic_interfile(output_image_filename, *output_image_sptr,
			NumericType::UCHAR, 1.F);
		return success == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;		
}

