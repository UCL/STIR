/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
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

$Date: 2013-04-03 
$Revision$

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
  start_angle=-500;
  add_key("start angle", 
          &start_angle);
  direction_of_rotation=" ";
  add_key("direction of rotation", 
          &direction_of_rotation);
  extent_of_rotation=-500;
  add_key("extent of rotation", 
          &extent_of_rotation);
  orbit = "";
  add_key("orbit",
          &orbit);
  add_key("radius (cm)",
          &radius_of_rotation);

}

bool InterfilePDFSHeaderSPECT::post_processing()
{

  if (InterfileHeader::post_processing() == true)
    return true;

  if (matrix_labels[0] != "bin coordinate")
    { 
      // use error message with index [1] as that is what the user sees.
      warning("Interfile error: expecting 'matrix axis label[1] := bin coordinate'\n"); 
      stop_parsing();
      return true; 
    }
  num_bins = matrix_size[0][0];
  bin_size_in_cm =  pixel_sizes[0]/10.;

  if (matrix_labels[1] == "axial coordinate" )
    {
      storage_order =ProjDataFromStream::Segment_View_AxialPos_TangPos;
      num_axial.push_back( matrix_size[1][0]);				
    }
  else
    { 
      warning("Interfile error: matrix labels not in expected (or supported) format\n"); 
      stop_parsing();
      return true; 
    }

  //Fill the radius depending on the type of orbit (just two orbits are supported)
  VectorWithOffset<float> radius(0,num_views-1); // this will be in mm
	
  if (orbit == "circular")
    { 
      for ( int i = 0 ; i < num_views ; i++ ) radius[ i ] = radius_of_rotation[0]*10;	
    }else if (orbit == "no-circular"){
	
    if ( radius_of_rotation.size() != static_cast<std::size_t>(num_views)) {
      warning("Interfile error: number of projections must be consistent with radius vector length \n");
      stop_parsing();
      return true; 
		
    }else
      {
        for ( int i = 0 ; i < num_views ; i++ ) radius[ i ] = radius_of_rotation[i]*10;	
      }
	
  }else{
	
    warning("Interfile error: only circular or no-circular orbits are suported\n"); 
    stop_parsing();
    return true; 
  }

  VectorWithOffset<int> sorted_min_ring_diff(0,0);
  VectorWithOffset<int> sorted_max_ring_diff(0,0);
  VectorWithOffset<int> sorted_num_rings_per_segment(0,0);
  sorted_min_ring_diff[0]=0;
  sorted_max_ring_diff[0]=0;
  sorted_num_rings_per_segment[0]=num_axial[0];

  // we construct a new scanner object with
  // data from the Interfile header (or the guessed scanner).
  // Initialize the scanner values (not used in SPECT reconstruction)

  const int num_rings = sorted_num_rings_per_segment[0];
  const int num_detectors_per_ring = num_views*2;  
  const double average_depth_of_interaction_in_cm = 0;
  const double distance_between_rings_in_cm = bin_size_in_cm*2;
  double default_bin_size_in_cm = bin_size_in_cm ;
  // this intrinsic tilt
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
	
  shared_ptr<Scanner> guessed_scanner_ptr(Scanner::get_scanner_from_name(originating_system));
  shared_ptr<Scanner> scanner_ptr_from_file(
                                            new Scanner(guessed_scanner_ptr->get_type(), 
                                                        originating_system,
                                                        num_detectors_per_ring, 
                                                        num_rings, 
                                                        max_num_non_arccorrected_bins, 
                                                        default_num_arccorrected_bins,
                                                        static_cast<float>(radius[0]),
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

  scanner_ptr_from_file->set_default_intrinsic_tilt( view_offset_in_degrees );

  ProjDataInfoCylindricalArcCorr* my_data_info_ptr = 
    new ProjDataInfoCylindricalArcCorr (
                                        scanner_ptr_from_file,
                                        float(bin_size_in_cm*10.),
                                        sorted_num_rings_per_segment,
                                        sorted_min_ring_diff,
                                        sorted_max_ring_diff,
                                        num_views,num_bins);
	
  my_data_info_ptr->set_ring_radii_for_all_views ( radius );

  const float angle_sampling = float (extent_of_rotation)/num_views * float(_PI/180);
  if(direction_of_rotation!="CC"){
    my_data_info_ptr->set_azimuthal_angle_sampling(-angle_sampling);
  }else{
    my_data_info_ptr->set_azimuthal_angle_sampling(angle_sampling);
  }
  this->data_info_sptr.reset(my_data_info_ptr);

  //cerr << data_info_ptr->parameter_info() << endl;

  return false;
}


END_NAMESPACE_STIR
