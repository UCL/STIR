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

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
	$Date$
	$Revision$
	
	  \par Usage:
	  \code
	  conv_to_SimSET_image [SimSET_image_filename][original_image]
	  Output: "SimSET_image_filename".hv AND "SimSET_image_filename".v 
	  
		This is a utility program converts a transmission image into SimSET attenuation input file.
		This is done by a simple segmentation of the transmission image, based on index values found 
		in the table below (http://depts.washington.edu/simset/html/user_guide/user_guide_index.html): 
		
		  Material              Attenuation index
		  -----------------      ---------------------
		  Air                          0    //Implemented
		  Water                        1	//Implemented
		  Blood                        2
		  Bone                         3	//Implemented	
		  Brain                        4	
		  Heart                        5
		  Lung                         6	//Implemented
		  Muscle			           7
		  Lead                         8
		  Sodium Iodide			       9
		  BGO                         10
		  Iron                        11
		  Graphite                    12
		  Tin                         13
		  GI Tract                    14
		  Connective tissue			  15
		  Copper					  16
		  Perfect absorber			  17
		  LSO                         18
		  GSO                         19
		  Aluminum		              20
		  Tungsten					  21
		  Liver						  22
		  Fat                         23
		  LaBr3                       24 
		  Low viscosity polycarbonate 25 
		  NEMA polyethylene			  26 
		  Polymethyl methylcrylate	  27 
		  Polystyrene fibers		  28 
		
		See the SimSET documentation for more details.
  
			If at least one attenuation value in the transmission image is not segmented either 
			implement the new attenuation indices or change the input image. 
			HINT: If the image is produced by the STIR utility: generate_image,
			the subsampling parameters should be set to 1 when small voxels sizes are used.
			
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
		if (argc!=3)
		{
			std::cerr << "Usage:" << argv[0]  
				<< " SimSET_image_filename original_image\n";    		 	  
			return EXIT_FAILURE;
		}
		shared_ptr< DiscretisedDensity<3,float> >  input_image_sptr = 
			DiscretisedDensity<3,float>::read_from_file(argv[2]);
		string output_image_filename(argv[1]);
		shared_ptr<DiscretisedDensity<3,float> > output_image_sptr = input_image_sptr->clone();		
		//Array<3,unsigned char>::full_iterator out_iter = output_image.begin_all();
		bool is_implemented=true;
		DiscretisedDensity<3,float>::full_iterator out_iter = output_image_sptr->begin_all();
		DiscretisedDensity<3,float>::const_full_iterator in_iter = input_image_sptr->begin_all_const();
		while( in_iter != input_image_sptr->end_all_const())
		{
		  // values from standard Simset file at 511keV
			if (fabs(*in_iter-0.096)<0.004)     // Water
				*out_iter = 1.F;
			else if (fabs(*in_iter-0.01)<0.010001) // Air
				*out_iter = 0.F;
			else if (fabs(*in_iter-0.19669)<0.004)  // Bone
				*out_iter = 3.F; 
			else if (fabs(*in_iter-0.02468)<0.005) // Lung
				*out_iter = 6.F;
			else if (fabs(*in_iter-0.0011)<0.005) // air
				*out_iter = 30.F;
			// WARNING: next value is really not aluminum (should be 0.22548)
			// However, this is currently a work-around to handle the
			// sphere used for our experiments
			else if (fabs(*in_iter-0.898)<0.005) // Aluminum
			        *out_iter = 20.F;

			else
			{
				is_implemented=false;
				std::cerr  << "\t" << *out_iter ;	
			}
			++in_iter; ++out_iter;
		}

		if(is_implemented==false)
			std::cerr << "\n At least one attenuation value which has the above shown values"
					  << "\n is not corresponded to a SimSET attenuation index in the segmentation."
					  << "\nImplement the new attenuation indices or change the input image. \n" 
					  << "HINT: If produced by generate_image set the subsampling parameters to 1, \n."
					  << "when small voxels sizes are used.\n";			
		/*
		std::ofstream out_stream(output_image_filename.c_str(), std::ios::out | std::ios::binary);
		if (!out_stream)
		error()
		Succeeded succes = write_data(out_stream, output_image);		*/			
		// write to file as 1 byte without changing the scale
		Succeeded success = 
			write_basic_interfile(output_image_filename, *output_image_sptr,
			NumericType::UCHAR, 1.F);
		return success == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;		
}

