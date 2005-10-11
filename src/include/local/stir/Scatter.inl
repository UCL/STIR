//
// $Id$
//

/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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
  \brief Inline implementations for Scatter functions

  \author Kris Thielemans
  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

inline
float dif_cross_section(const float cos_theta, float energy)
{ 
  const float sin_theta2= 1-cos_theta*cos_theta ;
  const float P= 1/(1+(energy/511.0)*(1-cos_theta));
  return(  Re*Re/2*P* (1 - P*sin_theta2 + P*P) );
}
inline
float dif_cross_section(const CartesianCoordinate3D<float>& scatter_point,
	  const CartesianCoordinate3D<float>& detector_coord_A,
	  const CartesianCoordinate3D<float>& detector_coord_B, float energy)
{
  /* the scatter angle theta  should be 0 when the 2 detectors 
     are at opposite sides of the scatter point. So, we need
	 to set cos_theta equal to the following (with a minus sign!)
	 */
  return
	  dif_cross_section(-cos_angle(detector_coord_A - scatter_point,
		                                  detector_coord_B - scatter_point), energy);
}
inline
float
energy_after_scatter(const float cos_theta, const float energy)
{
   return energy/(1+(energy/511.0)*(1-cos_theta));
}
inline
float
energy_after_scatter_511keV(const float cos_theta)
{
   return 511/(2-cos_theta);
}
inline
float dif_cross_section_511keV(const float cos_theta)
{
  const float sin_theta2= 1-cos_theta*cos_theta ;
  const float P= 1/(2-cos_theta);
  return(  Re*Re/2*P* (1 - P*sin_theta2 + P*P) );
}
inline
float dif_cross_section_511keV(const CartesianCoordinate3D<float>& scatter_point,
	  const CartesianCoordinate3D<float>& detector_coord_A,
	  const CartesianCoordinate3D<float>& detector_coord_B)
{
  /* the scatter angle theta  should be 0 when the 2 detectors 
     are at opposite sides of the scatter point. So, we need
	 to set cos_theta equal to the following (with a minus sign!)
	 */
  return
	  dif_cross_section_511keV(-cos_angle(detector_coord_A - scatter_point,
		                                  detector_coord_B - scatter_point)); 
}
inline
float total_cross_section(const float energy)
{
  const float a= energy/511.0;
  const float l= log(1.0+2.0*a);
  const double sigma0= 6.65E-25;
  return( 0.75*sigma0  * ( (1.0+a)/(a*a)*( 2.0*(1.0+a)/(1.0+2.0*a)- l/a ) + l/(2.0*a) - (1.0+3.0*a)/(1.0+2.0*a)/(1.0+2.0*a) ) );
}
inline
float total_cross_section_relative_to_511keV(const float energy)
{
  const float a= energy/511.0;
  static const float prefactor = 9/(-40 + 27*log(3.));
  return //check this in Mathematica
	  prefactor*
	  (((-4 - a*(16 + a*(18 + 2*a)))/square(1 + 2*a) + 
       ((2 + (2 - a)*a)*log(1 + 2*a))/a)/square(a)
	   );
}
inline 
float detection_efficiency( const float low, const float high, 
						    const float energy, 
							const float reference_energy, const float resolution)
{
	// factor 2.35482 is used to convert FWHM to sigma
	const float sigma_times_sqrt2= 
		sqrt(2.*energy*reference_energy)*resolution/2.35482;
	
	const float efficiency =
		0.5*( erf((high-energy)/sigma_times_sqrt2) 
		      - erf((low-energy)/sigma_times_sqrt2 ));	
                /* Maximum efficiency is 1.*/
	return efficiency;
}

inline
float max_cos_angle(const float low, const float approx, const float resolution)
{
	return
	2. - (8176.*log(2.))/(square(approx*resolution)*(511. + (16.*low*log(2.))/square(approx*resolution) - 
	sqrt(511.)*sqrt(511. + (32.*low*log(2.))/square(approx*resolution)))) ;
}

inline 
float energy_lower_limit(const float low, const float approx, const float resolution)
{
  return
  low + (approx*resolution)*(approx*resolution)*(46.0761 - 2.03829*sqrt(22.1807*low/square(approx*resolution)+511.));
}

END_NAMESPACE_STIR

