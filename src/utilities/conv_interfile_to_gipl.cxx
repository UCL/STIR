/*
 Copyright (C) 2009 - 2013, King's College London
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

 \brief This program converts Images from Interfile Format to GIPL (Guy's Imaging Processing Lab) format. 
 \author Charalampos Tsoumpas
 $Date$
 $Revision$
 */

// general header files
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <iostream>

#include "stir/IO/GIPL_ImageFormat.h"
#include "stir/IO/InterfileOutputFileFormat.h"
#include "stir/shared_ptr.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"


#ifndef STIR_NO_NAMESPACES
using std::string;
using std::ios;
using std::iostream;
using std::cout;
using std::endl;
using std::fstream;
using std::cerr;
using std::endl;
#endif


USING_NAMESPACE_STIR

// -------------------------------------------------------------------------
//   Main function
// -------------------------------------------------------------------------

int main(int argc, char* argv[])
{	
  if(argc>3 || argc<2) {
    std::cerr<<"Usage: " << argv[0] << " <image filename> <orientation flag>\n"
             << "\nConversion of an image from image file format to giplfile.\n"
             << "Orientation flag can be: 1, 2 or 3\n"
             << " For Transverse set to: 1 \n"
             << " For Coronal set to: 2 \n"
             << " For Sagittal set to: 3 \n"
             << "\t Orientation defaults to Coronal \n"
             << "output file will change only extension\n";
    exit(EXIT_FAILURE);
  }
  string filename(argv[1]);
  const shared_ptr<DiscretisedDensity<3,float> > image_sptr(DiscretisedDensity<3,float>::read_from_file(filename));
  string output_filename;
  string::iterator string_iter;
  for(string_iter=filename.begin(); 
      string_iter!=filename.end() && *string_iter!='.' ;
      ++string_iter)  
    output_filename.push_back(*string_iter); 
  output_filename+=".gipl";
  const int orientation = (argc==2) ? 2 : atoi(argv[2]) ;

  const DiscretisedDensity<3,float>& input_image = *image_sptr;	
  const DiscretisedDensityOnCartesianGrid <3,float>*  image_cartesian_ptr = 
    dynamic_cast< DiscretisedDensityOnCartesianGrid<3,float>*  > (image_sptr.get());
	
  const IndexRange<3> data_range=image_sptr->get_index_range();
  BasicCoordinate<3,int> min_range; BasicCoordinate<3,int> max_range;
  data_range.get_regular_range(min_range,max_range);
	
  const int num_voxels=(max_range[3]-min_range[3]+1)*(max_range[2]-min_range[2]+1)*(max_range[1]-min_range[1]+1);	
  Image image(num_voxels,64);	
  const Coordinate3D<float> origin=input_image.get_origin(); 
  const Coordinate3D<float> grid_spacing=image_cartesian_ptr->get_grid_spacing(); 
  if (orientation==1){
    image.m_dim[2]=max_range[1]-min_range[1]+1;
    image.m_dim[1]=max_range[2]-min_range[2]+1;
    image.m_dim[0]=max_range[3]-min_range[3]+1;
    image.m_pixdim[2]=grid_spacing[1];
    image.m_pixdim[1]=grid_spacing[2];
    image.m_pixdim[0]=grid_spacing[3];
    image.m_origin[2]=origin[1]-image.m_pixdim[2]*image.m_dim[2]/2.F;
    image.m_origin[1]=origin[2]-image.m_pixdim[1]*image.m_dim[1]/2.F;
    image.m_origin[0]=origin[3]-image.m_pixdim[0]*image.m_dim[0]/2.F;
    image.ImageOffset[1]=image.m_dim[0]*image.m_dim[1];
    image.ImageOffset[0]=image.m_dim[0];
    for (int k_out=min_range[1]; k_out<=max_range[1]; k_out++)
      for (int j_out=min_range[2]; j_out<=max_range[2]; j_out++)
        for (int i_out=min_range[3]; i_out<=max_range[3]; i_out++)
          {
            int index = i_out-min_range[3] + image.ImageOffset[0]*(j_out-min_range[2]) + image.ImageOffset[1]*(k_out-min_range[1]);
            image.vData_f[index]=input_image[k_out][j_out][i_out];
          }
  }
  else if (orientation==2) {
    image.m_dim[2]=max_range[2]-min_range[2]+1;
    image.m_dim[1]=max_range[1]-min_range[1]+1;
    image.m_dim[0]=max_range[3]-min_range[3]+1;
    image.m_pixdim[2]=grid_spacing[2];
    image.m_pixdim[1]=grid_spacing[1];
    image.m_pixdim[0]=grid_spacing[3];
    image.m_origin[2]=origin[2]-image.m_pixdim[2]*image.m_dim[2]/2.F;
    image.m_origin[1]=origin[1]-image.m_pixdim[1]*image.m_dim[1]/2.F;
    image.m_origin[0]=origin[3]-image.m_pixdim[0]*image.m_dim[0]/2.F;
    image.ImageOffset[1]=image.m_dim[0]*image.m_dim[1];
    image.ImageOffset[0]=image.m_dim[0];
    for (int k_out=min_range[1]; k_out<=max_range[1]; k_out++)
      for (int j_out=min_range[2]; j_out<=max_range[2]; j_out++)
        for (int i_out=min_range[3]; i_out<=max_range[3]; i_out++)
          {
            int index = i_out-min_range[3] + image.ImageOffset[0]*(k_out-min_range[1]) + image.ImageOffset[1]*(j_out-min_range[2]);
            image.vData_f[index]=input_image[k_out][j_out][i_out];
          }
  }
  else if (orientation==3) {
    image.m_dim[0]=max_range[2]-min_range[2]+1;
    image.m_dim[1]=max_range[1]-min_range[1]+1;
    image.m_dim[2]=max_range[3]-min_range[3]+1;
    image.m_pixdim[0]=grid_spacing[2];
    image.m_pixdim[1]=grid_spacing[1];
    image.m_pixdim[2]=grid_spacing[3];
    image.m_origin[2]=origin[3]-image.m_pixdim[2]*image.m_dim[2]/2.F;
    image.m_origin[1]=origin[1]-image.m_pixdim[1]*image.m_dim[1]/2.F;
    image.m_origin[0]=origin[2]-image.m_pixdim[0]*image.m_dim[0]/2.F;
    image.ImageOffset[1]=image.m_dim[0]*image.m_dim[1];
    image.ImageOffset[0]=image.m_dim[0];
    for (int k_out=min_range[1]; k_out<=max_range[1]; k_out++)
      for (int j_out=min_range[2]; j_out<=max_range[2]; j_out++)
        for (int i_out=min_range[3]; i_out<=max_range[3]; i_out++)
          {
            int index = j_out-min_range[2] + image.ImageOffset[0]*(k_out-min_range[1]) + image.ImageOffset[1]*(max_range[3]-i_out);
            image.vData_f[index]=input_image[k_out][j_out][i_out];
          }
  }
  else
    {
      std::cerr << "Orientation flag is not recognised." << std::endl;
      exit(EXIT_FAILURE);
    }

  std::cerr << "Now writing image in gipl format" << std::endl;
  image.GiplWrite(output_filename.c_str());
  return EXIT_SUCCESS;
}
