//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2.0 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup utilities
 
  \brief This program serves as a stand-alone zoom/trim utility for
  image files
  Run without arguments to get a usage message.
  \see zoom_image_in_place for conventions
  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/zoom.h"
#include "stir/IO/OutputFileFormat.h"
//#include "stir/utilities.h"
#include "stir/Succeeded.h"


#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if(argc<4) {
    cerr<<"Usage: \n" 
	<< '\t' << argv[0] << " <output filename> <input filename> sizexy [zoomxy [offset_in_mm_x [offset_in_mm_y [sizez [zoomz [offset_in_mm_z]]]]]]]\n"
	<< "or alternatively\n"
	<< '\t' << argv[0] << " --template template_filename <output filename> <input filename>\n";

    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line
  if (strcmp(argv[1], "--template")==0)
    {      
      char const * const template_filename = argv[2];
      char const * const output_filename = argv[3];
      char const * const input_filename = argv[4];

      shared_ptr<DiscretisedDensity<3,float> >  density_sptr = 
	DiscretisedDensity<3,float>::read_from_file(input_filename);
      shared_ptr<DiscretisedDensity<3,float> >  output_density_sptr = 
	DiscretisedDensity<3,float>::read_from_file(template_filename);
      output_density_sptr->fill(0);

      // zoom
      {
	const VoxelsOnCartesianGrid<float> * image_ptr =
	  dynamic_cast<VoxelsOnCartesianGrid<float> *>(density_sptr.get());
	if (image_ptr==NULL)
	  error("Input image is not of VoxelsOnCartesianGrid type. Sorry\n");
	VoxelsOnCartesianGrid<float> * output_image_ptr =
	  dynamic_cast<VoxelsOnCartesianGrid<float> *>(output_density_sptr.get());
	if (output_image_ptr==NULL)
	  error("Output image is not of VoxelsOnCartesianGrid type. Sorry\n");
	zoom_image(*output_image_ptr, *image_ptr);
      }
      // write it to file
      OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
	write_to_file(output_filename, *output_density_sptr);

      return EXIT_SUCCESS;
    }

  // old cmdline conventions

  char const * const output_filename = argv[1];
  char const * const input_filename = argv[2];
  const int sizexy= atoi(argv[3]);
  float zoomxy=1.F;
  float offset_in_mm_x=0.F;
  float offset_in_mm_y=0.F;
  int sizez=-1; // default
  float zoomz=1.F;
  float offset_in_mm_z=0.F;
  
  if (argc>4)
    zoomxy = static_cast<float>(atof(argv[4]));
  if (argc>5)
    offset_in_mm_x = static_cast<float>(atof(argv[5]));
  if (argc>6)
    offset_in_mm_y = static_cast<float>(atof(argv[6]));
  if (argc>7)
    sizez = atoi(argv[7]);
  if (argc>8)
    zoomz = static_cast<float>(atof(argv[8]));
  if (argc>9)
    offset_in_mm_z = static_cast<float>(atof(argv[9]));
  
  // read image 

  shared_ptr<DiscretisedDensity<3,float> >  density_ptr = 
    DiscretisedDensity<3,float>::read_from_file(input_filename);

  const VoxelsOnCartesianGrid<float> * image_ptr =
    dynamic_cast<VoxelsOnCartesianGrid<float> *>(density_ptr.get());

  if (image_ptr==NULL)
    error("Image is not of VoxelsOnCartesianGrid type. Sorry\n");

  // initialise zooming parameters

  if (sizez<0)
    sizez = image_ptr->get_z_size();

  const CartesianCoordinate3D<float> 
    zooms(zoomz,zoomxy,zoomxy);
  
  const CartesianCoordinate3D<float> 
    offsets_in_mm(offset_in_mm_z,offset_in_mm_y,offset_in_mm_x);
  const CartesianCoordinate3D<int> 
    new_sizes(sizez, sizexy, sizexy);


  const VoxelsOnCartesianGrid<float> new_image = 
    zoom_image(*image_ptr, zooms, offsets_in_mm, new_sizes);

  // write it to file
  OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(output_filename, new_image);

  return EXIT_SUCCESS;

}




