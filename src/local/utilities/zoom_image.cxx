//
// $Id$
//
/*!
  \file 
  \ingroup utilities
 
  \brief This programme serves as a stand-alone zoom/trim utility for
  image files

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/zoom.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"


#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if(argc<4) {
    cerr<<"Usage: " << argv[0] << " <output filename> <input filename> sizexy [zoomxy [offset_in_mm_x [offset_in_mm_y [sizez [zoomz [offset_in_mm_z]]]]]]]\n";
    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line

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
  
  // note z-offset might be  wrong ?

  const CartesianCoordinate3D<float> 
    offsets_in_mm(offset_in_mm_z,offset_in_mm_y,offset_in_mm_x);
  const CartesianCoordinate3D<int> 
    new_sizes(sizez, sizexy, sizexy);


  // copy image to provide space for output

  VoxelsOnCartesianGrid<float> new_image = *image_ptr;
  
  // zoom it
  zoom_image(new_image, zooms, offsets_in_mm, new_sizes);

  // write it to file
  DefaultOutputFileFormat output_file_format;
  output_file_format.write_to_file(output_filename, new_image);

  return EXIT_SUCCESS;

}




