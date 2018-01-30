/*
 Copyright (C) 2009 - 2013, King's College London
 Copyright (C) 2013, University College London
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
 
 \brief This program converts Images from GIPL (Guy's Imaging Processing Lab) format to Interfile Format.
 \author Charalampos Tsoumpas
 */

#include "stir/IO/InterfileOutputFileFormat.h"
#include "stir/shared_ptr.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/IO/GIPL_ImageFormat.h"

#include <stdio.h>
#include <sstream>
#include <fstream>
#include <iostream>

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
    std::cerr<<"Usage: " << argv[0] << " <gipl filename> <orientation flag>\n"
             << "\nConversion of an image from gipl format to interfile.\n"
             << "Orientation flag can be: 1, 2 or 3\n"
             << " For Transverse set to: 1 \n"
             << " For Coronal set to: 2 \n"
             << " For Sagittal set to: 3 \n"
             << "\t Orientation defaults to Transverse \n"
             << "output file will change only extension\n";	
    exit(EXIT_FAILURE);
  }		

  Image * image= new Image;
  image->GiplRead(argv[1]);
  string filename(argv[1]);
  string output_filename;
  string::iterator string_iter;
  for(string_iter=filename.begin(); 
      string_iter!=filename.end() && *string_iter!='.' ;
      ++string_iter)  
    output_filename.push_back(*string_iter); 
  const int orientation = (argc==2) ? 1 : atoi(argv[2]) ;
	
  if (orientation==1) {
    BasicCoordinate<3,int> min_range; 	BasicCoordinate<3,int> max_range;
    min_range[1]=0;  max_range[1]=image->m_dim[2]-1;
    min_range[2]=-static_cast<int>(floor(image->m_dim[1]/2.F));  max_range[2]=min_range[2] + image->m_dim[1]-1;
    min_range[3]=-static_cast<int>(floor(image->m_dim[0]/2.F));  max_range[3]=min_range[3] + image->m_dim[0]-1;
    IndexRange<3> data_range(min_range,max_range);
    Array<3,float> v_array(data_range);

    for (int k_out=min_range[1]; k_out<=max_range[1]; ++k_out)
      for (int j_out=min_range[2]; j_out<=max_range[2]; ++j_out)
        for (int i_out=min_range[3]; i_out<=max_range[3]; ++i_out)
          {
            int index = i_out-min_range[3] + image->ImageOffset[0]*(j_out-min_range[2]) + image->ImageOffset[1]*(k_out-min_range[1]);
            if (image->m_image_type==15)
              v_array[k_out][j_out][i_out]=(float)image->vData[index];
            else if  (image->m_image_type==64)
              v_array[k_out][j_out][i_out]=image->vData_f[index];
          }
    const CartesianCoordinate3D<float> 
      grid_spacing(image->m_pixdim[2],image->m_pixdim[1],image->m_pixdim[0]);
    CartesianCoordinate3D<float> origin(0.F,0.F,0.F);
	// TODO not sure if this is correct for even/odd-sized data.
    origin[1]=static_cast<float>(image->m_origin[2]+image->m_pixdim[2]*image->m_dim[2]/2.F);
    origin[2]=static_cast<float>(image->m_origin[1]+image->m_pixdim[1]*image->m_dim[1]/2.F);
    origin[3]=static_cast<float>(image->m_origin[0]+image->m_pixdim[0]*image->m_dim[0]/2.F);
    const VoxelsOnCartesianGrid<float> new_image(v_array,origin,grid_spacing);
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
      write_to_file(output_filename, new_image);
  }
  else if (orientation==2) {
    BasicCoordinate<3,int> min_range; 	BasicCoordinate<3,int> max_range;
    min_range[1]=0;  max_range[1]=image->m_dim[1]-1;
    min_range[2]=-static_cast<int>(floor(image->m_dim[2]/2.F));  max_range[2]=min_range[2] + image->m_dim[2]-1;
    min_range[3]=-static_cast<int>(floor(image->m_dim[0]/2.F));  max_range[3]=min_range[3] + image->m_dim[0]-1;
		
    IndexRange<3> data_range(min_range,max_range);
    Array<3,float> v_array(data_range);

    for (int k_out=min_range[1]; k_out<=max_range[1]; ++k_out)
      for (int j_out=min_range[2]; j_out<=max_range[2]; ++j_out)
        for (int i_out=min_range[3]; i_out<=max_range[3]; ++i_out)
          {
            int index = i_out-min_range[3] + image->ImageOffset[0]*(k_out-min_range[1]) + image->ImageOffset[1]*(j_out-min_range[2]);
            if (image->m_image_type==15)
              v_array[k_out][j_out][i_out]=(float)image->vData[index];
            else if  (image->m_image_type==64)
              v_array[k_out][j_out][i_out]=image->vData_f[index];
          }
    const CartesianCoordinate3D<float> 
      grid_spacing(image->m_pixdim[1],image->m_pixdim[2],image->m_pixdim[0]);
    CartesianCoordinate3D<float> origin(0.F,0.F,0.F);
    origin[2]=static_cast<float>(image->m_origin[2]+image->m_pixdim[2]*image->m_dim[2]/2.F);
    origin[1]=static_cast<float>(image->m_origin[1]+image->m_pixdim[1]*image->m_dim[1]/2.F);
    origin[3]=static_cast<float>(image->m_origin[0]+image->m_pixdim[0]*image->m_dim[0]/2.F);
    const VoxelsOnCartesianGrid<float> new_image(v_array,origin,grid_spacing);
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
      write_to_file(output_filename, new_image);
  }
  else if (orientation==3) {
    BasicCoordinate<3,int> min_range; 	BasicCoordinate<3,int> max_range;
    min_range[1]=0;  max_range[1]=image->m_dim[1]-1;
    min_range[2]=-static_cast<int>(floor(image->m_dim[0]/2.F));  max_range[2]=min_range[2] + image->m_dim[0]-1;
    min_range[3]=-static_cast<int>(floor(image->m_dim[2]/2.F));  max_range[3]=max_range[3] + image->m_dim[2]-1;
		
    IndexRange<3> data_range(min_range,max_range);
    Array<3,float> v_array(data_range);

    for (int k_out=min_range[1]; k_out<=max_range[1]; ++k_out)
      for (int j_out=min_range[2]; j_out<=max_range[2]; ++j_out)
        for (int i_out=min_range[3]; i_out<=max_range[3]; ++i_out)
          {
            int index = j_out-min_range[2] + image->ImageOffset[0]*(k_out-min_range[1]) + image->ImageOffset[1]*(max_range[3]-i_out);
            if (image->m_image_type==15)
              v_array[k_out][j_out][i_out]=(float)image->vData[index];
            else if  (image->m_image_type==64)
              v_array[k_out][j_out][i_out]=image->vData_f[index]; 
          }
    const CartesianCoordinate3D<float> 
      grid_spacing(image->m_pixdim[1],image->m_pixdim[0],image->m_pixdim[2]);
    CartesianCoordinate3D<float> origin(0.F,0.F,0.F);
    origin[3]=static_cast<float>(image->m_origin[2]+image->m_pixdim[2]*image->m_dim[2]/2.F);
    origin[1]=static_cast<float>(image->m_origin[1]+image->m_pixdim[1]*image->m_dim[1]/2.F);
    origin[2]=static_cast<float>(image->m_origin[0]+image->m_pixdim[0]*image->m_dim[0]/2.F);

    const VoxelsOnCartesianGrid<float> new_image(v_array,origin,grid_spacing);
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
      write_to_file(output_filename, new_image);
  }
  else
    {
      std::cerr << "Orientation flag is not recognised." << std::endl;
      exit(EXIT_FAILURE);
    }
  return EXIT_SUCCESS;
}
