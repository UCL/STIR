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
  \ingroup utilities
  \brief Writes the Bland-Altman values for two images in a text file.
  \author Charalampos Tsoumpas
  $Date$
  $Revision$

  \par Usage:
  \code 
  Bland_Altman_plot output_filename_prefix image_1 image_2 xmin xmax ymin ymax zmin zmax
  \endcode
  
  \param image1/image2 must have the same sizes. 
  \param x/y/zmin/max denote a rectangular region for which the Bland Altman Plot will be estimated.
  

  It writes two lists: Average-Bias of the two images to a text file \a (.txt) and the rest statistical values to another text file \a (.stat)
  \note The Bias is estimated using the \a image1-image2 formula.

  \todo Add to the Doxygen documentation a reference to their paper and how exactly this utility works.
*/

#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/linear_regression.h"
#include "stir/VectorWithOffset.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>

//ChT::ToDo:Include stddev values!!!

USING_NAMESPACE_STIR
using namespace std; 
int main(int argc, char *argv[])
{ 
  if (argc!=10)
    {
      std::cerr << "Returns results in a text file for Bland-Altman evaluation and a stat file.\n Usage:" << argv[0] 
		<< " output_file_name_prefix image_1 image_2 xmin xmax ymin ymax zmin zmax\n ";
      return EXIT_FAILURE;
    }
  string output_txt_string(argv[1]);
  const std::string name=output_txt_string+".txt";
  const std::string statname=output_txt_string+".stat";

  ofstream output_stream(name.c_str(), ios::out); //output file //
    if(!output_stream)    
    {
      std::cerr << "Cannot open " << name << endl ;
      return EXIT_FAILURE;
    }
  shared_ptr< DiscretisedDensity<3,float> >  image_1_sptr = 
    DiscretisedDensity<3,float>::read_from_file(argv[2]);
  DiscretisedDensity<3,float>& image_1 = *image_1_sptr;

  shared_ptr< DiscretisedDensity<3,float> >  image_2_sptr = 
    DiscretisedDensity<3,float>::read_from_file(argv[3]);
  DiscretisedDensity<3,float>& image_2 = *image_2_sptr;

  const VoxelsOnCartesianGrid<float>*  image_1_cartesian_ptr = 
    dynamic_cast< VoxelsOnCartesianGrid<float>*  > (image_1_sptr.get());

  const VoxelsOnCartesianGrid<float>*  image_2_cartesian_ptr = 
    dynamic_cast< VoxelsOnCartesianGrid<float>*  > (image_2_sptr.get());

  CartesianCoordinate3D<float> grid_spacing;
  if (image_1_cartesian_ptr==0 && image_2_cartesian_ptr==0)
    {
      warning("Something strange is going on: at least one input image is not DiscretisedDensityOnCartesianGrid objects.\n");
      return EXIT_FAILURE;
    }
  else if(image_1_cartesian_ptr->get_grid_spacing()!=image_2_cartesian_ptr->get_grid_spacing() && //or is it better to put an assert?
	  (image_1_cartesian_ptr->get_max_x()-image_1_cartesian_ptr->get_min_x())!=
	  (image_2_cartesian_ptr->get_max_x()-image_2_cartesian_ptr->get_min_x()) &&
	  (image_1_cartesian_ptr->get_max_y()-image_1_cartesian_ptr->get_min_y())!=
	  (image_2_cartesian_ptr->get_max_y()-image_2_cartesian_ptr->get_min_y()) &&
	  (image_1_cartesian_ptr->get_max_z()-image_1_cartesian_ptr->get_min_z())!=
	  (image_1_cartesian_ptr->get_max_z()-image_2_cartesian_ptr->get_min_z()) &&
	  image_1_cartesian_ptr->get_x_size()!=image_2_cartesian_ptr->get_x_size() &&
	  image_1_cartesian_ptr->get_y_size()!=image_2_cartesian_ptr->get_y_size() &&
	  image_1_cartesian_ptr->get_z_size()!=image_2_cartesian_ptr->get_z_size())
    {
      warning("Input images have different grid or/and voxel sizes.\n");
      return EXIT_FAILURE;
    }
  else
    {
      const int min_i_index= atoi(argv[4]);
      const int max_i_index= atoi(argv[5]);
      const int min_j_index= atoi(argv[6]); 
      const int max_j_index= atoi(argv[7]);
      const int min_k_index= atoi(argv[8]);
      const int max_k_index= atoi(argv[9]);
      const int num_voxels = (max_k_index-min_k_index+1) * (max_j_index - min_j_index+1) * (max_i_index-min_i_index+1);
  
      VectorWithOffset<float> bias(0,num_voxels-1);
      VectorWithOffset<float> average(0,num_voxels-1);
      VectorWithOffset<float> residual(0,num_voxels-1);
      VectorWithOffset<float> weights(0,num_voxels-1);
      VectorWithOffset<float> fit_values(0,6);
      VectorWithOffset<float> residual_fit_values(0,6);
      std::cerr << "Writing into text file the Bland-Altman values...\n";
      output_stream  << " Average " << "\t" <<  "Bias\n"  ;

      int voxel_num=0;
      for ( int k = min_k_index; k<= max_k_index; ++k)
	for ( int j = min_j_index; j<= max_j_index; ++j)
	  for ( int i = min_i_index; i<= max_i_index; ++i)
	    {	      	      
	      average[voxel_num]=(image_1[k][j][i]+image_2[k][j][i])/2.;
              bias[voxel_num]=image_1[k][j][i]-image_2[k][j][i];
              weights[voxel_num]=1;
	      output_stream << std::setw(8) << (image_1[k][j][i]+image_2[k][j][i])/2. << "\t" 
			    << std::setw(8) <<  image_1[k][j][i]-image_2[k][j][i] << "\n"; 
	      ++voxel_num;
    }
  output_stream.close();
  assert(voxel_num==num_voxels);

  linear_regression(fit_values.begin(),
		    bias.begin(), bias.end(), 
		    average.begin(), weights.begin());

  for ( int voxel_num = 0 ; voxel_num<num_voxels ; ++voxel_num)
    residual[voxel_num]=fabs(fit_values[0]+fit_values[1]*average[voxel_num] - bias[voxel_num]);

  linear_regression(residual_fit_values.begin(),
		    residual.begin(), residual.end(), 
		    average.begin(), weights.begin());

  ofstream stat_stream(statname.c_str(), ios::out); //output file //
  if(!stat_stream)    
    {
      std::cerr << "Cannot open " << statname << endl ;
      return EXIT_FAILURE;
    }
  std::cerr << "Writing into stat file the Bland-Altman values...\n";
  stat_stream  << std::setw(38) << " TotalVoxels "                    << std::setw(18) << num_voxels << "\n"  
	       << std::setw(38) << " BiasAverageConstant "            << std::setw(18) << fit_values[0] << "\n"  
	       << std::setw(38) << " BiasAverageConstantVariance"     << std::setw(18) << fit_values[3] << "\n"  
	       << std::setw(38) << " BiasAverageScale "               << std::setw(18) << fit_values[1] << "\n" 
	       << std::setw(38) << " BiasAverageScaleVariance"        << std::setw(18) << fit_values[4] << "\n" 
	       << std::setw(38) << " BiasAverageStdDev"               << std::setw(18) << sqrt(fit_values[2]/(num_voxels-1)) << "\n" 
	       << std::setw(38) << " BiasAverageChiSquare"            << std::setw(18) << fit_values[2] << "\n" 
	       << std::setw(38) << " BiasAverageConstScaleCovar"      << std::setw(18) << fit_values[5] << "\n" 
	       << std::setw(38) << " ResidualAverageConstant "        << std::setw(18) << residual_fit_values[0] << "\n"  
	       << std::setw(38) << " ResidualAverageConstantVariance" << std::setw(18) << residual_fit_values[3] << "\n" 
	       << std::setw(38) << " ResidualAverageScale "           << std::setw(18) << residual_fit_values[1] << "\n"  
	       << std::setw(38) << " ResidualAverageScaleVariance "   << std::setw(18) << residual_fit_values[4] << "\n" 
	       << std::setw(38) << " ResidualAverageChiSquare"        << std::setw(18) << residual_fit_values[2] << "\n" 
	       << std::setw(38) << " ResidualAverageConstScaleCovar"  << std::setw(18) << residual_fit_values[5] << "\n" 
	       << std::setw(38) << " ResidualAveragePearson "         << std::setw(18) << residual_fit_values[6] << "\n" ;
  stat_stream.close();

  return EXIT_SUCCESS;
    }
}
