//
// $Id$
//
/*
    Copyright (C) 2004 - $Date$, Hammersmith Imanet Ltd
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
  \brief Inline implementations of class stir::ScatterEstimationByBin.
  
  \author Nikolaos Dikaios
  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/
START_NAMESPACE_STIR

float
ScatterEstimationByBin::
compute_emis_to_scatter_points_solid_angle_factor_doubles11(const CartesianCoordinate3D<float>& scatter_point_1,
								      const CartesianCoordinate3D<float>& scatter_point_2,
								      const CartesianCoordinate3D<float>& emis_point)

{
#if 1
  const CartesianCoordinate3D<float> dist_vector = scatter_point_2 - scatter_point_1 ;

  const float dist_sp1_sp2_squared = norm_squared(dist_vector);

  // attempt to avoid overflow by saying that the maximum
  // solid angle is about 4Pi/9
  // However, this doesn't work very well yet as the 1/d^2
  // would need to be multiplied with the area of the voxel
  // to have a solid angle
  return
    std::min(static_cast<float>(_PI/2), 1.F  / dist_sp1_sp2_squared);
 
#else
const CartesianCoordinate3D<float> dist_vector_1 = scatter_point_1 - emis_point ; 
   
  const CartesianCoordinate3D<float> dist_vector_2 = scatter_point_2 - emis_point ; 

  const float dist_emis_sp1_squared = norm_squared(dist_vector_1);
      
  const float dist_emis_sp2_squared = norm_squared(dist_vector_2);
      
  const float emis_to_scatter_point_1_solid_angle_factor = 1.F / dist_emis_sp1_squared ;

   const float emis_to_scatter_point_2_solid_angle_factor = 1.F / dist_emis_sp2_squared ; 
   
   return
     std::min(std::min(emis_to_scatter_point_1_solid_angle_factor,
		       emis_to_scatter_point_2_solid_angle_factor),
	      static_cast<float>(_PI/2.F));
#endif
}

float
ScatterEstimationByBin::
compute_sc1_to_sc2_solid_angle_factor_doubles20(const CartesianCoordinate3D<float>& scatter_point_1,
						const CartesianCoordinate3D<float>& scatter_point_2)
  
{

  const CartesianCoordinate3D<float> dist_vector = scatter_point_2 - scatter_point_1 ;

  const float dist_sp1_sp2_squared = norm_squared(dist_vector);

  // attempt to avoid overflow by saying that the maximum
  // solid angle is about 4Pi/9
  // However, this doesn't work very well yet as the 1/d^2
  // would need to be multiplied with the area of the voxel
  // to have a solid angle
  const float scatter_points_solid_angle_factor = 
    std::min(static_cast<float>(_PI/2), 1.F  / dist_sp1_sp2_squared);
 
  return scatter_points_solid_angle_factor ;
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
dif_cross_section(const float cos_theta, float energy)
{ 
  const double Re = 2.818E-13;   // aktina peristrofis electroniou gia to atomo tou H
  const float sin_theta_2= 1-cos_theta*cos_theta ;
  const float P= 1/(1+(energy/511.0)*(1-cos_theta)); 
  return( (Re*Re/2) * P * (1 - P * sin_theta_2 + P * P));
}

float
ScatterEstimationByBin::
dif_polarized_cross_section(const float cos_theta1, const float cos_theta2, const float cos_phi, float energy1, float energy2)
{ 
  const float sin_theta1_2= 1-cos_theta1*cos_theta1 ;
  const float sin_theta2_2= 1-cos_theta2*cos_theta2 ;
  const float sin_phi_2= 1-cos_phi * cos_phi;
  const float P1= 1/(1+(energy1/511.0)*(1-cos_theta1));
  const float P2= 1/(1+(energy2/511.0)*(1-cos_theta2));
  const float gamma1=P1+1/P1;
  const float gamma2=P2+1/P2;
  static const float prefactor_2 =square(9/(2*_PI*(40 - 27*log(3.))))*1E-30;
   

  return  prefactor_2*P1*P1*P2*P2* (gamma1*gamma2-gamma1*sin_theta2_2-gamma2*sin_theta1_2+2*sin_theta2_2*sin_theta1_2*sin_phi_2)*_PI ;
}

float
ScatterEstimationByBin::
energy_after_scatter(const float cos_theta, const float energy)
{
  return energy/(1+(energy/511.0)*(1-cos_theta));   // For an arbitrary energy 
}

float
ScatterEstimationByBin::
energy_after_scatter_511keV(const float cos_theta)
{
  return 511/(2-cos_theta); // for a given energy, energy := 511 keV
}

float
ScatterEstimationByBin::
total_cross_section(const float energy)
{
  const float a= energy/511.0;
  const float l= log(1.0+2.0*a); 
    const double sigma0= 6.65E-25;   // sigma0=8*pi*a*a/(3*m*m)
    return( 0.75*sigma0  * ( (1.0+a)/(a*a)*( 2.0*(1.0+a)/(1.0+2.0*a)- l/a ) + l/(2.0*a) - (1.0+3.0*a)/(1.0+2.0*a)/(1.0+2.0*a) ) ); // Klein - Nishina formula = sigma / sigma0
}


float
ScatterEstimationByBin::
total_cross_section_relative_to_511keV(const float energy)
{
  const float a= energy/511.0;
  static const float prefactor = 9/(-40 + 27*log(3.)); //Klein-Nishina formula for a=1 & devided with 0.75 == (40 - 27*log(3)) / 9

  return //check this in Mathematica
    prefactor*
    (((-4 - a*(16 + a*(18 + 2*a)))/square(1 + 2*a) +       
      ((2 + (2 - a)*a)*log(1 + 2*a))/a)/square(a)
     );
}

float
ScatterEstimationByBin::
detection_efficiency(const float energy)
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
max_cos_angle(const float low, const float approx, const float resolution)
{
	return
	2. - (8176.*log(2.))/(square(approx*resolution)*(511. + (16.*low*log(2.))/square(approx*resolution) - 
	sqrt(511.)*sqrt(511. + (32.*low*log(2.))/square(approx*resolution)))) ;
}


float
ScatterEstimationByBin::
energy_lower_limit(const float low, const float approx, const float resolution)
{
  return
  low + (approx*resolution)*(approx*resolution)*(46.0761 - 2.03829*sqrt(22.1807*low/square(approx*resolution)+511.));
}

END_NAMESPACE_STIR
