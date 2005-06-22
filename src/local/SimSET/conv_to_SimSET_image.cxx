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
--------------      ---------------------
Air                          0
Water                        1
Blood                        2
Bone                         3
Brain                        4
Heart                        5
Lung                         6
Muscle                       7
Lead                         8
Sodium Iodide                9
BGO                         10
Iron                        11
Graphite                    12
Tin                         13
GI Tract                    14
Connective tissue           15
Copper                      16
Perfect absorber            17
LSO                         18
GSO                         19
Aluminum                    20
Tungsten                    21
Liver                       22
Fat                         23
LaBr3                       24 
Low viscosity polycarbonate 25 
NEMA polyethylene           26 
Polymethyl methylcrylate    27 
Polystyrene fibers          28 

 
		See the SimSET documentation for more details.
		
		  \endcode	  
*/

#include "stir/utilities.h"
#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include <iomanip>

#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cout;
using std::cerr;  
using std::setw;
#endif
#include "stir/Array.h"
#include "stir/recon_array_functions.h"
#include "stir/ArrayFunction.h"
#include <iostream> 
#include <fstream>
using std::ofstream;
using std::ifstream;
using std::ios;
using std::cerr;
using std::endl;
using std::cout;
using std::cerr;

/***********************************************************/     
int main(int argc, char *argv[])                                  
{         
  USING_NAMESPACE_STIR
  using namespace std;   

	if (argc!=3)
	{
		cerr << "Usage:" << argv[0]  
			<< "SimSET_image_filename" << endl
			<< "original_image" << endl
			<< "SimSET_image_template" << endl ;    		 	  
		return 0;
	}

	shared_ptr< DiscretisedDensity<3,float> >  input_image_sptr = 
    DiscretisedDensity<3,float>::read_from_file(argv[2]);
	DiscretisedDensity<3,float>& input_image = *input_image_sptr;

	shared_ptr< DiscretisedDensity<3,int> >  output_image_template_sptr= 
		DiscretisedDensity<3,int>::read_from_file(argv[3]);

	string output_image_filename(argv[1]);
	DiscretisedDensity<3,int> 
		output_image(output_image_template_sptr->clone(),output_image_filename);
		
	const int in_zmin=input_image.get_min_z(), 
		in_ymin=input_image.get_min_y(), 
		in_xmin=input_image.get_min_x(),
		in_zmax=input_image.get_max_z(), 
		in_ymax=input_image.get_max_y(), 
		in_xmax=input_image.get_max_x();
	const int out_zmin=output_image.get_min_z(), 
		out_ymin=output_image.get_min_y(), 
		out_xmin=output_image.get_min_x(),
		out_zmax=output_image.get_max_z(), 
		out_ymax=output_image.get_max_y(), 
		out_xmax=output_image.get_max_x();

	assert(in_zmin-in_zmax==out_zmin-out_zmax)
	assert(in_ymin-in_ymax==out_ymin-out_ymax)
	assert(in_xmin-in_xmax==out_ymin-out_ymax)		
	
	for(int in_k=in_zmin, out_k=out_zmin; in_k<=in_zmax && out_k<=out_zmax  ; ++in_k, ++out_k)
		for(int in_j=in_zmin, out_j=out_zmin; in_j<=in_zmax && out_j<=out_zmax  ; ++in_j, ++out_j)
			for(int in_i=in_zmin, out_i=out_zmin; in_i<=in_zmax && out_i<=out_zmax  ; ++in_i, ++out_i)
			{
				if (abs(input_image[in_k][in_j][in_i]-0.096)<0.004) // Water
					output_image[out_k][out_j][out_i] = 1;
				if (abs(input_image[in_k][in_j][in_i]-0.00)<0.004) // Air
					output_image[out_k][out_j][out_i] = 0;
				if (abs(input_image[in_k][in_j][in_i]-2.2)<0.004) // Bone
					output_image[out_k][out_j][out_i] = 3; 
				if (abs(input_image[in_k][in_j][in_i]-0.26)<0.004) // Lung
					output_image[out_k][out_j][out_i] = 6;
			}			
			return 1;
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




