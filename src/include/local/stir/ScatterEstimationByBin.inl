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
  low + (approx*resolution)*(approx*resolution)*(46.0761 - 2.03829*sqrt(22.1807*low/square(approx*resolution_at_511keV)+511.));
}

END_NAMESPACE_STIR
