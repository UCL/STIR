//
// $Id$
//
/*
  Copyright (C) 2004- $Date$, Hammersmith Imanet
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
  \ingroup scatter
  \brief Implementations of detection modelling in stir::ScatterEstimationByBin

  \author Charalampos Tsoumpas
  \author Pablo Aguiar
  \author Nikolaos Dikaios
  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/scatter/ScatterEstimationByBin.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/numerics/erf.h"
#include <iostream>

START_NAMESPACE_STIR
unsigned 
ScatterEstimationByBin::
find_in_detection_points_vector(const CartesianCoordinate3D<float>& coord) const
{
  std::vector<CartesianCoordinate3D<float> >::const_iterator iter=
    std::find(detection_points_vector.begin(),
              detection_points_vector.end(),
              coord);
  if (iter != detection_points_vector.end())
    {
      return iter-detection_points_vector.begin();
    }
  else
    {
      if (detection_points_vector.size()==static_cast<std::size_t>(this->total_detectors))
        error("More detection points than we think there are!\n");

      detection_points_vector.push_back(coord);
      return detection_points_vector.size()-1;
    }
}

void
ScatterEstimationByBin::
find_detectors(unsigned& det_num_A, unsigned& det_num_B, const Bin& bin) const
{
  CartesianCoordinate3D<float> detector_coord_A, detector_coord_B;
  this->proj_data_info_ptr->
    find_cartesian_coordinates_of_detection(
                                            detector_coord_A,detector_coord_B,bin);
  det_num_A =
    this->find_in_detection_points_vector(detector_coord_A + 
                                          this->shift_detector_coordinates_to_origin);
  det_num_B =
    this->find_in_detection_points_vector(detector_coord_B + 
                                          this->shift_detector_coordinates_to_origin);
}

float
ScatterEstimationByBin::
compute_emis_to_det_points_solid_angle_factor(
                                              const CartesianCoordinate3D<float>& emis_point,
                                              const CartesianCoordinate3D<float>& detector_coord)
{
  
  const CartesianCoordinate3D<float> dist_vector = emis_point - detector_coord ;
 

  const float dist_emis_det_squared = norm_squared(dist_vector);

  const float emis_det_solid_angle_factor = 1.F/ dist_emis_det_squared ;

  return emis_det_solid_angle_factor ;
}

float
ScatterEstimationByBin::
detection_efficiency(const float energy) const
{
  // factor 2.35482 is used to convert FWHM to sigma
  const float sigma_times_sqrt2= 
    sqrt(2.*energy*this->reference_energy)*this->energy_resolution/2.35482;  // 2.35482=2 * sqrt( 2 * ( log(2) )
  
  // sigma_times_sqrt2= sqrt(2) * sigma   // resolution proportional to FWHM    
  
  const float efficiency =
    0.5*( erf((this->upper_energy_threshold-energy)/sigma_times_sqrt2) 
          - erf((this->lower_energy_threshold-energy)/sigma_times_sqrt2 ));     
  /* Maximum efficiency is 1.*/
  return efficiency;
}

float
ScatterEstimationByBin::
max_cos_angle(const float low, const float approx, const float resolution_at_511keV)
{
  return
    2. - (8176.*log(2.))/(square(approx*resolution_at_511keV)*(511. + (16.*low*log(2.))/square(approx*resolution_at_511keV) - 
                                                               sqrt(511.)*sqrt(511. + (32.*low*log(2.))/square(approx*resolution_at_511keV)))) ;
}


float
ScatterEstimationByBin::
energy_lower_limit(const float low, const float approx, const float resolution_at_511keV)
{
  return
  low + (approx*resolution_at_511keV)*(approx*resolution_at_511keV)*(46.0761 - 2.03829*sqrt(22.1807*low/square(approx*resolution_at_511keV)+511.));
}

double
ScatterEstimationByBin::
detection_efficiency_no_scatter(const unsigned det_num_A, 
                                const unsigned det_num_B) const
{
  // TODO: slightly dangerous to use a static here
  // it would give wrong results when the energy_thresholds are changed...
  static const float detector_efficiency_no_scatter =
    detection_efficiency(511.F) > 0 
    ? detection_efficiency(511.F)
    : (std::cerr << "Zero detection efficiency for 511. Will normalise to 1\n", 1.F);

  const CartesianCoordinate3D<float>& detector_coord_A =
    detection_points_vector[det_num_A];
  const CartesianCoordinate3D<float>& detector_coord_B =
    detection_points_vector[det_num_B];
  const CartesianCoordinate3D<float> 
    detA_to_ring_center(0,-detector_coord_A[2],-detector_coord_A[3]);
  const CartesianCoordinate3D<float> 
    detB_to_ring_center(0,-detector_coord_B[2],-detector_coord_B[3]);
  const float rAB_squared=norm_squared(detector_coord_A-detector_coord_B);
  const float cos_incident_angle_A = 
    cos_angle(detector_coord_B - detector_coord_A,
              detA_to_ring_center) ;
  const float cos_incident_angle_B = 
    cos_angle(detector_coord_A - detector_coord_B,
              detB_to_ring_center) ;

  //0.75 is due to the volume of the pyramid approximation!
  return
    1./(  0.75/2./_PI *
    rAB_squared
    /detector_efficiency_no_scatter/
    (cos_incident_angle_A*
     cos_incident_angle_B));
}

END_NAMESPACE_STIR
