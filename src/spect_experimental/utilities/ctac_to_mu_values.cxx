//
//
/*
  Copyright (C) 2019, Institute of Nuclear Medicine, University College London
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
  \brief Produces an image of mu-values from a CT/CTAC.
  \author Benjamin A. Thomas

  \par Usage:
  \code
  ctac_to_mu_values -o output_filename -i input_directory -j slope_filename
  \endcode
*/

#include <algorithm>
#include <cctype>

#include "stir/IO/ITKImageInputFileFormat.h"
#include "stir/info.h"
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/DiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/stream.h"
#include "stir/getopt.h"


#include <nlohmann/json.hpp>

USING_NAMESPACE_STIR

typedef DiscretisedDensity<3,float> FloatImageType;

Succeeded apply_bilinear_scaling_to_HU(
    const std::unique_ptr<FloatImageType> &input_image_sptr,
    const nlohmann::json &transform,
    std::shared_ptr<FloatImageType> output_image_sptr){

  FloatImageType::full_iterator out_iter = output_image_sptr->begin_all();
  FloatImageType::const_full_iterator in_iter = input_image_sptr->begin_all_const();

  const float a1 = transform["a1"];
  const float b1 = transform["b1"];

  const float a2 = transform["a2"];
  const float b2 = transform["b2"];

  //std::cout << transform.dump(4);

  while( in_iter != input_image_sptr->end_all_const())
  {
    if (*in_iter<0.f) {
      float mu = a1 + b1 *(*in_iter);
      *out_iter = (mu < 0.0f) ? 0.0f : mu;
    } else {
      *out_iter = a2 + b2 * (*in_iter);
    }

    ++in_iter; ++out_iter;
  }

  return Succeeded::yes;
}


int main(int argc, char * argv[])
{
  USING_NAMESPACE_STIR;
  const char * output_filename = 0;
  const char * input_filename = 0;
  const char * slope_filename = 0;
  const char * manufacturer_name = 0;
  const char * keV_str = 0;

  const char * const usage = "ctac_to_mu_values -o output_filename -i input_dicom_slice -j slope_filename -m manufacturer_name -k target_energy\n";
  opterr = 0;
  {
    char c;

    while ((c = getopt (argc, argv, "i:o:j::m:k:?")) != -1)
      switch (c)
	{
	case 'i':
	  input_filename = optarg;
	  break;
	case 'o':
	  output_filename = optarg;
	  break;
	case 'j':
      slope_filename = optarg;
	  break;
    case 'm':
      manufacturer_name = optarg;
      break;
	case 'k':
      keV_str = optarg;
      break;
	case '?':
	  std::cerr << usage;
	  return EXIT_FAILURE;
	default:
	  if (isprint (optopt))
	    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
	  else
	    fprintf (stderr,
		     "Unknown option character `\\x%x'.\n",
		     optopt);
	  std::cerr << usage;
	  return EXIT_FAILURE;
	}
  }

  if (output_filename==0 || input_filename==0 || slope_filename==0 || manufacturer_name==0 || keV_str==0 )
    {
      std::cerr << usage;
      return EXIT_FAILURE;
    }

  //Read slope file
  std::ifstream slope_json_file_stream(slope_filename);
  nlohmann::json slope_json;
  slope_json_file_stream >> slope_json;

  //Put user-specified manufacturer into upper case.
  std::string manufacturer = manufacturer_name;
  std::locale loc;

  for (std::string::size_type i=0; i<manufacturer.length(); ++i)
    manufacturer[i] = std::toupper(manufacturer[i],loc);

  stir::info(boost::format("Manufacturer: '%s'") % manufacturer);

  //Get desired keV as integer value
  int keV;

  std::stringstream ss;
  ss << keV_str;
  ss >> keV;

  stir::info(boost::format("target keV: '%i'") % keV);

  //Extract appropriate chunk of JSON file for given manufacturer.
  nlohmann::json target = slope_json["scale"][manufacturer]["transform"];

  int location = -1;
  int pos = 0;
  for (auto entry : target){
    if (entry["kev"] == keV)
      location = pos;
    pos++;
  }

  if (location == -1){
    std::cerr << "Desired keV: " << keV << " not found! ";
    std::cerr << "Aborting!";
    return EXIT_FAILURE;
  }

  //Extract transform for specific keV.
  nlohmann::json j = target[location];
  //std::cout << j.dump(4);

  //Read DICOM data
  stir::info(boost::format("ctac_to_mu_values: opening file %1%") % input_filename);
  std::unique_ptr< FloatImageType > input_image_sptr(stir::read_from_file<FloatImageType>( input_filename ));

  unique_ptr<VoxelsOnCartesianGrid<float> >image_aptr
      (dynamic_cast<VoxelsOnCartesianGrid<float> *>(
           DiscretisedDensity<3,float>::read_from_file(input_filename))
      );

  BasicCoordinate<3,int> min_indices, max_indices;
  if (!image_aptr->get_regular_range(min_indices, max_indices))
    error("Non-regular range of coordinates. That's strange.\n");

  BasicCoordinate<3,float> edge_min_indices(min_indices), edge_max_indices(max_indices);
  edge_min_indices-= 0.5F;
  edge_max_indices+= 0.5F;

  std::cout << "\nOrigin in mm {z,y,x}    :" << image_aptr->get_origin()
            << "\nVoxel-size in mm {z,y,x}:" << image_aptr->get_voxel_size()
            << "\nMin_indices {z,y,x}     :" << min_indices
            << "\nMax_indices {z,y,x}     :" << max_indices
            << "\nNumber of voxels {z,y,x}:" << max_indices - min_indices + 1
            << "\nPhysical coordinate of first index in mm {z,y,x} :"
            << image_aptr->get_physical_coordinates_for_indices(min_indices)
            << "\nPhysical coordinate of last index in mm {z,y,x}  :"
            << image_aptr->get_physical_coordinates_for_indices(max_indices)
            << "\nPhysical coordinate of first edge in mm {z,y,x} :"
            << image_aptr->get_physical_coordinates_for_indices(edge_min_indices)
            << "\nPhysical coordinate of last edge in mm {z,y,x}  :"
            << image_aptr->get_physical_coordinates_for_indices(edge_max_indices);
  std::cout << std::endl;

  //Create output image from input image.
  shared_ptr< FloatImageType > output_image_sptr(input_image_sptr->clone());
  //Apply scaling.
  apply_bilinear_scaling_to_HU(input_image_sptr, j,  output_image_sptr);
  //Write output file.
  Succeeded success = OutputFileFormat< FloatImageType >::default_sptr()->
      write_to_file(output_filename, *output_image_sptr);

  return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;

}
