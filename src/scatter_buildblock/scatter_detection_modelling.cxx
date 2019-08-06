//
//
/*
  Copyright (C) 2004- 2009, Hammersmith Imanet
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

*/

#include "stir/scatter/ScatterSimulation.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/numerics/erf.h"
#include "stir/info.h"
#include <iostream>

START_NAMESPACE_STIR
unsigned 
ScatterSimulation::
find_in_detection_points_vector(const CartesianCoordinate3D<float>& coord) const
{
  unsigned int ret_value = 0;
#pragma omp critical(SCATTERESTIMATIONFINDDETECTIONPOINTS)
  {
  std::vector<CartesianCoordinate3D<float> >::const_iterator iter=
    std::find(detection_points_vector.begin(),
              detection_points_vector.end(),
              coord);
  if (iter != detection_points_vector.end())
    {
      ret_value = iter-detection_points_vector.begin();
    }
  else
    {
      if (detection_points_vector.size()==static_cast<std::size_t>(this->total_detectors))
        error("More detection points than we think there are!\n");

      detection_points_vector.push_back(coord);
      ret_value = detection_points_vector.size()-1;
    }
  }
  return ret_value;
}

void
ScatterSimulation::
find_detectors(unsigned& det_num_A, unsigned& det_num_B, const Bin& bin) const
{
  CartesianCoordinate3D<float> detector_coord_A, detector_coord_B;
  this->proj_data_info_cyl_noarc_cor_sptr->
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
ScatterSimulation::
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
ScatterSimulation::
detection_efficiency(const float energy,const int en_window) const
{
  // factor 2.35482 is used to convert FWHM to sigma
  const float sigma_times_sqrt2= 
    sqrt(2.*energy*this->proj_data_info_cyl_noarc_cor_sptr->get_scanner_ptr()->get_reference_energy())*
          this->proj_data_info_cyl_noarc_cor_sptr->get_scanner_ptr()->get_energy_resolution()/2.35482f;  // 2.35482=2 * sqrt( 2 * ( log(2) )
  
  // sigma_times_sqrt2= sqrt(2) * sigma   // resolution proportional to FWHM    
  
  const float efficiency =
    0.5f*( erf((this->template_exam_info_sptr->get_high_energy_thres(en_window)-energy)/sigma_times_sqrt2)
          - erf((this->template_exam_info_sptr->get_low_energy_thres(en_window)-energy)/sigma_times_sqrt2 ));
  /* Maximum efficiency is 1.*/
  return efficiency;
}

float
ScatterSimulation::
detection_efficiency_full_model(const float energy,const int en_window) const
{
  const float HLD = this->template_exam_info_sptr->get_high_energy_thres(en_window);
  const float LLD = this->template_exam_info_sptr->get_low_energy_thres(en_window);
  const float ref_energy = HLD - (HLD - LLD)/2; //the reference energy is the center of the given energy window

  const int Z = 66; // atomic number of LSO
  const float H_1 = pow(Z,5)/ref_energy; //high of the photopeak prop. to the photoelectric cross section
  const float H_2 = 8.17*pow(10,25)*total_Compton_cross_section(ref_energy)*Z;
  const float H_3 = 7;
  const float H_4 = 235;
  const float beta = -4.682;
  const float global_scale = 0.0000296;
  const float fwhm = this->proj_data_info_cyl_noarc_cor_sptr->get_scanner_ptr()->get_energy_resolution(); //full width half maximum
  const float std_peak = sqrt(energy*ref_energy)*fwhm/2.35482;
  const float scaling_std_compton = 107;
  const float shift_compton = 0.5916;

  const float f1 = photoelectric(H_1, std_peak, energy, ref_energy);
  const float f2 = compton_plateau(H_2, std_peak, energy, ref_energy,scaling_std_compton,shift_compton);
  const float f3 = flat_continuum(H_3,std_peak, energy, ref_energy);
  const float f4 = exponential_tail(H_4,std_peak, energy, ref_energy,beta);

  return global_scale*(f1+f2+f3+f4);
}

float
ScatterSimulation::
photoelectric(const float K, const float std_peak, const float energy, const float ref_energy) const
{
  const double pi = boost::math::constants::pi<double>();
  const float diff = energy - ref_energy;
  const float pow_diff = pow(diff,2);
  const float pow_std_peak = pow(std_peak,2);
  return  K/(std_peak*sqrt(2*pi))*exp(-pow_diff/(2*pow_std_peak));
}

float
ScatterSimulation::
compton_plateau(const float K, const float std_peak, const float energy, const float ref_energy, const float scaling_std_compton,const float shift_compton) const
{
    const double pi = boost::math::constants::pi<double>();
    const float m_0_c_2 = 511.F;
    const float alpha = ref_energy/m_0_c_2;
    const float theta = 2*pi;
    const float E_1 = ref_energy/(1+alpha*(1-cos(theta)));
    const float mean = ref_energy*shift_compton;
    return ((ref_energy/E_1)+(E_1/ref_energy)-1+cos(theta))*(K*exp(-(pow((energy - mean),2))/(scaling_std_compton*std_peak)));
}
float
ScatterSimulation::
flat_continuum(const float K, const float std_peak, const float energy, const float ref_energy) const
{
    const float den = sqrt(2)*std_peak;
    float f = 0;
        if (energy<=ref_energy)
            f = K* erfc((energy-ref_energy)/den);
        else
            f = 0;
     return f;
}

float
ScatterSimulation::
exponential_tail(const float K, const float std_peak, const float energy, const float ref_energy, const float beta) const
{
    const double pi = boost::math::constants::pi<double>();
    const float den1 = sqrt(2)*pi*std_peak*beta;
    const float den2 = sqrt(2)*std_peak;
    const float den3 = 2*beta;
    float f = 0;

    if (energy > 100)
        f = K * exp((energy-ref_energy)/den1)*erfc((energy-ref_energy)/den2+1/den3);
    else
        f =0;
    return f;
}

float
ScatterSimulation::
max_cos_angle(const float low, const float approx, const float resolution_at_511keV)
{
  return
    2.f - (8176.*log(2.))/(square(approx*resolution_at_511keV)*(511. + (16.*low*log(2.))/square(approx*resolution_at_511keV) -
                                                               sqrt(511.)*sqrt(511. + (32.*low*log(2.))/square(approx*resolution_at_511keV)))) ;
}


float
ScatterSimulation::
energy_lower_limit(const float low, const float approx, const float resolution_at_511keV)
{
  return
  low + (approx*resolution_at_511keV)*(approx*resolution_at_511keV)*(46.0761 - 2.03829*sqrt(22.1807*low/square(approx*resolution_at_511keV)+511.));
}

double
ScatterSimulation::
detection_efficiency_no_scatter(const unsigned det_num_A, 
                                const unsigned det_num_B) const
{
  // TODO: slightly dangerous to use a static here
  // it would give wrong results when the energy_thresholds are changed...
    static const float detector_efficiency_no_scatter = 1;
   /*detection_efficiency(511.F, en_window) > 0
    ? detection_efficiency(511.F, en_window)
    : (info("Zero detection efficiency for 511. Will normalise to 1"), 1.F);*/

  const CartesianCoordinate3D<float>& detector_coord_A =
    detection_points_vector[det_num_A];
  const CartesianCoordinate3D<float>& detector_coord_B =
    detection_points_vector[det_num_B];
  const CartesianCoordinate3D<float> 
    detA_to_ring_center(0,-detector_coord_A[2],-detector_coord_A[3]);
  const CartesianCoordinate3D<float> 
    detB_to_ring_center(0,-detector_coord_B[2],-detector_coord_B[3]);
  const float rAB_squared=static_cast<float>(norm_squared(detector_coord_A-detector_coord_B));
  const float cos_incident_angle_A = static_cast<float>(
    cos_angle(detector_coord_B - detector_coord_A,
              detA_to_ring_center)) ;
  const float cos_incident_angle_B = static_cast<float>(
    cos_angle(detector_coord_A - detector_coord_B,
              detB_to_ring_center)) ;

  //0.75 is due to the volume of the pyramid approximation!
  return
    1./(  0.75/2./_PI *
    rAB_squared
    /pow(detector_efficiency_no_scatter,2.0)/
    (cos_incident_angle_A*
     cos_incident_angle_B));
}

END_NAMESPACE_STIR
