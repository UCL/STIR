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
#include "stir/IO/write_data.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if(argc!=4) {
    cerr<<"Usage: " << argv[0] << " <output filename> <input filename> threshold\n";
    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line

  std::string output_filename = argv[1];
  char const * const input_filename = argv[2];
  const float threshold = atof(argv[3]);
  // read image 

  shared_ptr<DiscretisedDensity<3,float> >  density_ptr = 
    DiscretisedDensity<3,float>::read_from_file(input_filename);

  // binarise
  for (DiscretisedDensity<3,float>::full_iterator iter = density_ptr->begin_all();
       iter != density_ptr->end_all();
	   ++iter)
	{
	  if (*iter < threshold)
		     *iter = 0.F;
	  else
		*iter = 1.F;
	}

  CartesianCoordinate3D<int> min_indices;
  CartesianCoordinate3D<int> max_indices;
  if (!density_ptr->get_regular_range(min_indices, max_indices))
  {
    warning("write_basic_interfile: can handle only regular index ranges\n. No output\n");
    return Succeeded::no;
  }
  CartesianCoordinate3D<int> dimensions = max_indices - min_indices;
  dimensions += 1;

  char * data_name = new char[output_filename.size() + 5];
  char * header_name = new char[output_filename.size() + 5];

  strcpy(data_name, output_filename.c_str());
  // KT 29/06/2001 make sure that a filename ending on .hv is treated correctly
  {
   const char * const extension = strchr(find_filename(data_name),'.');
   if (extension!=NULL && strcmp(extension, ".hv")==0)
     replace_extension(data_name, ".v");
   else
     add_extension(data_name, ".v");
  }
  strcpy(header_name, data_name);
  replace_extension(header_name, ".hv");

  std::ofstream output_data;
  open_write_binary(output_data, data_name);

  float scale = 1;
  write_data(output_data, *density_ptr, NumericType::SCHAR, scale);
  if (scale != 1)
	error("scale after writing is not 1: %g", scale);

  VectorWithOffset<float> scaling_factors(1);
  scaling_factors[0] = scale;
  VectorWithOffset<unsigned long> file_offsets(1);
  file_offsets.fill(0);


    const VoxelsOnCartesianGrid<float> * image_ptr =
    dynamic_cast<VoxelsOnCartesianGrid<float> *>(density_ptr.get());

  if (image_ptr==NULL)
    error("Image is not of VoxelsOnCartesianGrid type. Sorry\n");

  const CartesianCoordinate3D<float> voxel_size =
	  image_ptr->get_voxel_size();

  write_basic_interfile_image_header(header_name,
				   data_name,
				   dimensions, 
				   voxel_size,
				   NumericType::SCHAR,
				   ByteOrder::native,
				   scaling_factors,
				   file_offsets);
  delete header_name;
  delete data_name;

  return EXIT_SUCCESS;

}




