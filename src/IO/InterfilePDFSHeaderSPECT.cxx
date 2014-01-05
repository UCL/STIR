/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2013-2014, University College London
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
\ingroup InterfileIO
\brief  This file implements the classes stir::InterfilePDFSHeaderSPECT

\author Kris Thielemans
\author Berta Marti Fuster
*/

#include "stir/IO/InterfilePDFSHeaderSPECT.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include <numeric>
#include <functional>

#ifndef STIR_NO_NAMESPACES
using std::binary_function;
using std::pair;
using std::sort;
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR

//KT 26/10/98
// KT 13/11/98 moved stream arg from constructor to parse()
InterfilePDFSHeaderSPECT::InterfilePDFSHeaderSPECT()
: InterfileHeader()
{
  num_segments = 1;
  num_views = -1;
  add_key("number of projections", 
          &num_views);
  start_angle=0;
  add_key("start angle", 
          &start_angle);
  direction_of_rotation="cw";
  add_key("direction of rotation", 
          &direction_of_rotation);
  extent_of_rotation=double_value_not_set;
  add_key("extent of rotation", 
          &extent_of_rotation);
  // TODO convert to ASCIIlist
  orbit = "circular";
  add_key("orbit",
          &orbit);
  radius_of_rotation = double_value_not_set;
  add_key("radius",&radius_of_rotation);
  add_key("radii",&radii_of_rotation); // for non-circular orbits
  // overwrite vectored-value, as v3.3 had a scalar
  add_key("data offset in bytes", &data_offset);

}

bool InterfilePDFSHeaderSPECT::post_processing()
{

  if (InterfileHeader::post_processing() == true)
    return true;

  // for compatibility with PET code
  data_offset_each_dataset[0] = data_offset;

  // SPECT v3.3 doesn't really define matrix_labels. We just check that if they're present, they are as in PET
  if (matrix_labels[0].size()>0 && matrix_labels[0] != "bin coordinate")
    { 
      // use error message with index [1] as that is what the user sees.
      warning("Interfile error: expecting 'matrix axis label[1] := bin coordinate'"); 
      return true; 
    }
  if (matrix_labels[1].size()>0 && matrix_labels[1] != "axial coordinate" )
    { 
      // use error message with index [2] as that is what the user sees.
      warning("Interfile error: expecting 'matrix axis label[2] := axial coordinate'"); 
      return true; 
    }

  if (extent_of_rotation == double_value_not_set)
    {
      warning("Interfile error: extent of rotation needs to be set");
      return true;
    }

  num_bins = matrix_size[0][0];
  bin_size_in_cm =  pixel_sizes[0]/10.;

  storage_order =ProjDataFromStream::Segment_View_AxialPos_TangPos;
  num_axial_poss= matrix_size[1][0];

  //Fill the radius depending on the type of orbit (just two orbits are supported)
  // will be in mm (SPECT Interfile uses mm)
  VectorWithOffset<float> radii(0, num_views-1);
  orbit=standardise_keyword(orbit);
  if (orbit == "circular")
    { 
      if (radius_of_rotation == double_value_not_set)
        {
          warning("Interfile error: radius not set");
          return true;
        }
      for ( int i = 0 ; i < num_views ; i++ ) radii[ i ] = static_cast<float>(radius_of_rotation);	
    }
  else if (orbit == "non-circular")
    {
	
      if ( radii_of_rotation.size() != static_cast<std::size_t>(num_views))
        {
          warning("Interfile error: number of projections must be consistent with radius vector length");
          return true; 
        }
      for ( int i = 0 ; i < num_views ; i++ ) radii[ i ] = static_cast<float>(radii_of_rotation[i]);	
    }
  else
    {
      warning("Interfile error: only circular or non-circular orbits are supported"); 
      return true; 
    }
 

  // somewhat strange values to be compatible with PET
  VectorWithOffset<int> sorted_min_ring_diff(0,0);
  VectorWithOffset<int> sorted_max_ring_diff(0,0);
  VectorWithOffset<int> sorted_num_rings_per_segment(0,0);
  sorted_min_ring_diff[0]=0;
  sorted_max_ring_diff[0]=0;
  sorted_num_rings_per_segment[0]=num_axial_poss;

  // we construct a new scanner object with
  // data from the Interfile header (or the guessed scanner).
  // Initialize the scanner values (most are not used in SPECT reconstruction)

  const int num_rings = sorted_num_rings_per_segment[0];
  const int num_detectors_per_ring = num_views*2;  
  const double average_depth_of_interaction_in_cm = 0;
  const double distance_between_rings_in_cm = bin_size_in_cm*2;
  double default_bin_size_in_cm = bin_size_in_cm ;
  const double view_offset_in_degrees = start_angle;
  const int max_num_non_arccorrected_bins = num_bins;
  const int default_num_arccorrected_bins = num_bins;
  const int num_axial_blocks_per_bucket = 1;
  const int num_transaxial_blocks_per_bucket = 1;
  const int num_axial_crystals_per_block = 1;
  const int num_transaxial_crystals_per_block = 1;
  const int num_axial_crystals_per_singles_unit = 1;
  const int num_transaxial_crystals_per_singles_unit = 1;
  const int num_detector_layers = 1;
	
  shared_ptr<Scanner> guessed_scanner_ptr(Scanner::get_scanner_from_name(get_exam_info_ptr()->originating_system));
  shared_ptr<Scanner> scanner_ptr_from_file(
                                            new Scanner(guessed_scanner_ptr->get_type(), 
                                                        get_exam_info_ptr()->originating_system,
                                                        num_detectors_per_ring, 
                                                        num_rings, 
                                                        max_num_non_arccorrected_bins, 
                                                        default_num_arccorrected_bins,
                                                        static_cast<float>(radii[0]),
                                                        static_cast<float>(average_depth_of_interaction_in_cm*10),
                                                        static_cast<float>(distance_between_rings_in_cm*10.),
                                                        static_cast<float>(default_bin_size_in_cm*10),
                                                        static_cast<float>(view_offset_in_degrees*_PI/180),
                                                        num_axial_blocks_per_bucket, 
                                                        num_transaxial_blocks_per_bucket,
                                                        num_axial_crystals_per_block,
                                                        num_transaxial_crystals_per_block,
                                                        num_axial_crystals_per_singles_unit,
                                                        num_transaxial_crystals_per_singles_unit,
                                                        num_detector_layers));

  if (default_bin_size_in_cm <= 0)
    default_bin_size_in_cm =
      scanner_ptr_from_file->get_default_bin_size()/10;
  else if (fabs(bin_size_in_cm - 
		scanner_ptr_from_file->get_default_bin_size()/10)>.001)	
    warning("Interfile warning: unexpected bin size in cm\n",
            bin_size_in_cm,
            scanner_ptr_from_file->get_default_bin_size()/10);

  ProjDataInfoCylindricalArcCorr* my_data_info_ptr = 
    new ProjDataInfoCylindricalArcCorr (
                                        scanner_ptr_from_file,
                                        float(bin_size_in_cm*10.),
                                        sorted_num_rings_per_segment,
                                        sorted_min_ring_diff,
                                        sorted_max_ring_diff,
                                        num_views,num_bins);
	
  my_data_info_ptr->set_ring_radii_for_all_views ( radii);

  direction_of_rotation =  standardise_keyword(direction_of_rotation);
  const float angle_sampling = float (extent_of_rotation)/num_views * float(_PI/180);
  if(direction_of_rotation=="cw")
    {
      my_data_info_ptr->set_azimuthal_angle_sampling(-angle_sampling);
    }
  else if(direction_of_rotation=="ccw")
    {
      my_data_info_ptr->set_azimuthal_angle_sampling(angle_sampling);
    }
  else
    {
      warning("direction of rotation has to be CW or CCW");
      return true;
    }
  this->data_info_sptr.reset(my_data_info_ptr);

  //cerr << data_info_ptr->parameter_info() << endl;

  return false;
}


END_NAMESPACE_STIR
