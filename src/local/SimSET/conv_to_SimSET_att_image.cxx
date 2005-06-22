/*
Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
	  conv_to_SimSET_image [SimSET_image_filename]
	  [original_image]
	  [SimSET_image_template]	
	  Output: SimSET_image_filename
	  
		This is a utility program converts a transmission image into SimSET attenuation input file.
		This is done by converting the values of the transmission image, which are close to the given 
		values found in the table below: 

				Material              Attenuation index
			-----------------      ---------------------
				  Air                         0
				 Water                        1
				 Blood                        2
				 Bone                         3
				 Brain                        4
				 Heart                        5
				 Lung                         6
				Muscle			              7
				 Lead                         8
			 Sodium Iodide			          9
				 BGO                         10
				 Iron                        11
				Graphite                     12
				 Tin                         13
			   GI Tract                      14
			Connective tissue				 15
			   Copper						 16
			Perfect absorber				 17
				LSO                          18
				GSO                          19
			  Aluminum		                 20
			  Tungsten						 21
			   Liver						 22
				Fat                          23
			   LaBr3                         24 
		Low viscosity polycarbonate		     25 
		 NEMA polyethylene					 26 
	   Polymethyl methylcrylate				 27 
		Polystyrene fibers					 28 
 
		See the SimSET documentation for more details.
		
		  \endcode	  
*/


#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/IO/interfile.h"
#include "stir/Succeeded.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
//#include "stir/Array.h"
//#include "stir/IO/write_data.h"
#include <iostream> 
//#include <fstream>
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
	//Array<3,unsigned char> 
	//	output_image(input_image_sptr->get_index_range());
    shared_ptr<DiscretisedDensity<3,float> > output_image_sptr = input_image_sptr->clone();

   		
#if 0
	
	for(int in_k=in_zmin, out_k=out_zmin; in_k<=in_zmax && out_k<=out_zmax  ; ++in_k, ++out_k)
		for(int in_j=in_ymin, out_j=out_ymin; in_j<=in_ymax && out_j<=out_ymax  ; ++in_j, ++out_j)
			for(int in_i=in_xmin, out_i=out_xmin; in_i<=in_xmax && out_i<=out_xmax  ; ++in_i, ++out_i)
			{
				if (abs(*in_iter-0.096)<0.004) // Water
					*out_iter = 1;
				if (abs(*in_iter-0.00)<0.004) // Air
					*out_iter = 0;
				if (abs(*in_iter-2.2)<0.004) // Bone
					*out_iter = 3; 
				if (abs(*in_iter-0.26)<0.004) // Lung
					*out_iter = 6;
			}
#else
			//Array<3,unsigned char>::full_iterator out_iter = output_image.begin_all();
			DiscretisedDensity<3,float>::full_iterator out_iter = output_image_sptr->begin_all();
			DiscretisedDensity<3,float>::const_full_iterator in_iter = input_image_sptr->begin_all_const();
			while( in_iter != input_image_sptr->end_all_const())
			{
				if (fabs(*in_iter-0.096)<0.004) // Water
					*out_iter = 1.F;
				else if (fabs(*in_iter-0.00)<0.004) // Air
					*out_iter = 0.F;
				else if (fabs(*in_iter-2.2)<0.004) // Bone
					*out_iter = 3.F; 
				else if (fabs(*in_iter-0.26)<0.004) // Lung
					*out_iter = 6.F;
              ++in_iter; ++out_iter;
			}
#endif
 /*
			std::ofstream out_stream(output_image_filename.c_str(), std::ios::out | std::ios::binary);
			if (!out_stream)
				error()
			Succeeded succes = write_data(out_stream, output_image);
*/			
			// write to file as 1 byte without changing the scale
			Succeeded success = 
				write_basic_interfile(output_image_filename, *output_image_sptr,
				                       NumericType::UCHAR,
									   1.F);
			return success == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;

}

/*  
class Material {
public::
Material(unsigned int index, float mue_at_511keV); 
set_Material(unsigned int index, float mue_511keV);

  private::
  {
  unsigned int index; 
  float mue_at_511keV;
  }
  }
  air Density =   1.2927999E-03  Aeff =    14.40540      Zeff =    7.200000
  511.0        0.00011        0.99984        0.99769
  water Density =    1.000000      Aeff =    13.00121      Zeff =    7.216770 
  511.0        0.09580        0.99981        0.99776
  blood Density =    1.060000      Aeff =    12.93043      Zeff =    7.115872
  511.0        0.10067        0.99980        0.99779
	 bone Density =    2.200000      Aeff =    20.99179      Zeff =    10.83470
	 511.0        0.19669        0.99800        0.99480
	 lung Density =   0.2600000      Aeff =    12.94264      Zeff =    7.118764
	 511.0        0.02468        0.99980        0.99779
	 lead Density =    11.35000      Aeff =    207.2000      Zeff =    82.00000   
	 511.0        1.76568        0.49490        0.86404
	 set_Material(unsigned int ind, float mue_at_511keV)
	 {
	 index=ind;
	 mue_at_511keV=mue_511keV;
	 }
	 Material water(1,0.096);
	 Material air(0,0);
	*/	




