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
  ctac_to_mu_values -o output_filename -i input_volume -j slope_filename -m manufacturer_name [-v kilovoltage_peak] -k target_photon_energy
  \endcode
  Default value for tube voltage is 120 kV.
*/

#include <string>

#include "stir/info.h"
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/DiscretisedDensity.h"
#include "stir/getopt.h"

#include <nlohmann/json.hpp>

USING_NAMESPACE_STIR

typedef DiscretisedDensity<3,float> FloatImageType;

Succeeded get_record_from_json(
    nlohmann::json &output_json,
    std::string &manufacturer,
    const nlohmann::json input_json,
    const std::string &keV_str,
    const std::string& kVp_str){

  //Put user-specified manufacturer into upper case.
  std::locale loc;

  for (std::string::size_type i=0; i<manufacturer.length(); ++i)
    manufacturer[i] = std::toupper(manufacturer[i],loc);

  stir::info(boost::format("Manufacturer: '%s'") % manufacturer);

  //Get desired keV as integer value
  float keV_as_float;
  int keV;

  std::stringstream ss;
  ss << keV_str;
  ss >> keV_as_float;

  keV = std::round(keV_as_float);

  stir::info(boost::format("target keV: '%i'") % keV);

  //Get desired kVp as integer value
  int kVp;

  ss.clear();
  ss << kVp_str;
  ss >> kVp;

  stir::info(boost::format("kVp: '%i'") % kVp);

  //Extract appropriate chunk of JSON file for given manufacturer.
  nlohmann::json target = input_json["scale"][manufacturer]["transform"];

  int location = -1;
  int pos = 0;
  for (auto entry : target){
    if ( (entry["kev"] == keV) && (entry["kvp"] == kVp) )
      location = pos;
    pos++;
  }

  if (location == -1){
    stir::error("Desired slope not found!");
    return Succeeded::no;
  }

  //Extract transform for specific keV and kVp.
  output_json = target[location];
  //std::cout << output_json.dump(4);

  return Succeeded::yes;
}

Succeeded apply_bilinear_scaling_to_HU(
    FloatImageType &output_image_sptr,
    const FloatImageType &input_image_sptr,
    const nlohmann::json &transform){

  FloatImageType::full_iterator out_iter = output_image_sptr.begin_all();
  FloatImageType::const_full_iterator in_iter = input_image_sptr.begin_all_const();

  const float a1 = transform["a1"];
  const float b1 = transform["b1"];

  const float a2 = transform["a2"];
  const float b2 = transform["b2"];

  const float breakPoint = transform["break"];

  //std::cout << transform.dump(4);

  while( in_iter != input_image_sptr.end_all_const())
  {
    if (*in_iter < breakPoint) {
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
  const char * kVp_str = 0;
  const char * keV_str = 0;

  const char * const usage = "ctac_to_mu_values -o output_filename -i input_volume -j slope_filename -m manufacturer_name [-v kilovoltage_peak] -k target_photon_energy\n"
	  "Default value for tube voltage is 120 kV.\n";
  opterr = 0;
  {
    char c;

    while ((c = getopt (argc, argv, "i:o:j:m:v:k:?")) != -1)
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
  case 'v':
      kVp_str = optarg;
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

  if (kVp_str == 0){
    //If the user does not specify a value for kVP, assume 120 kVp.
    kVp_str = "120";
    stir::info(boost::format("No value for kVp given, assuming %s") % kVp_str);
  }

  //Read slope file
  std::ifstream slope_json_file_stream(slope_filename);
  nlohmann::json slope_json;
  slope_json_file_stream >> slope_json;

  std::string manufacturer = manufacturer_name;

  nlohmann::json j;

  //Extract the target slope information from the given file of slope definitions.
  if ( get_record_from_json(j, manufacturer, slope_json, keV_str, kVp_str) == Succeeded::no ) {
    stir::error(boost::format("ctac_to_mu_values: unable to find the desired slope reference in %1%") % slope_filename);
    stir::error("Aborting!");
    return EXIT_FAILURE;
  }

  //Read DICOM data
  stir::info(boost::format("ctac_to_mu_values: opening file %1%") % input_filename);
  unique_ptr< FloatImageType > input_image_sptr(stir::read_from_file<FloatImageType>( input_filename ));
  //Create output image from input image.
  shared_ptr< FloatImageType > output_image_sptr(input_image_sptr->clone());
  //Apply scaling.
  apply_bilinear_scaling_to_HU(*output_image_sptr, *input_image_sptr, j);
  //Write output file.
  Succeeded success = OutputFileFormat< FloatImageType >::default_sptr()->
      write_to_file(output_filename, *output_image_sptr);

  return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;

}
