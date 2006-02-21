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


#include <math.h>
#include "stir/CartesianCoordinate3D.h"
#include "stir/BasicCoordinate.h"
#include "stir/cross_product.h"

START_NAMESPACE_STIR




inline
float dif_cross_section(const float cos_theta, float energy)
{ 
  const float sin_theta_2= 1-cos_theta*cos_theta ;
  const float P= 1/(1+(energy/511.0)*(1-cos_theta));  //  P = E / energy 
  return( (Re*Re/2) * P * (1 - P * sin_theta_2 + P * P));//*sqrt(sin_theta_2) );
}


inline
float dif_cross_section_sin(const float cos_theta, float energy)
{ 
  const float sin_theta_2= 1-cos_theta*cos_theta ;
  return dif_cross_section(cos_theta,energy)* sqrt(sin_theta_2) ;
  
}




#if 0

inline
float dif_cross_section(const CartesianCoordinate3D<float>& scatter_point,
	  const CartesianCoordinate3D<float>& detector_coord_A,
	  const CartesianCoordinate3D<float>& detector_coord_B, float energy)
{
  /* the scatter angle theta  should be 0 when the 2 detectors 
     are at opposite sides of the scatter point. So, we need
     to set cos_theta equal to the following (with a minus sign theta=pi-angle,     thus cos(pi-angle)=-cos(angle) !!!!)
  */
  return
    dif_cross_section(-cos_angle(detector_coord_A - scatter_point,
				 detector_coord_B - scatter_point), energy);
}


inline
float dif_cross_section_sin(const CartesianCoordinate3D<float>& scatter_point,
			    const CartesianCoordinate3D<float>& detector_coord_A,
			    const CartesianCoordinate3D<float>& detector_coord_B, float energy)
{
 
  return
    dif_cross_section_sin(-cos_angle(detector_coord_A - scatter_point,
				     detector_coord_B - scatter_point), energy)*sqrt(1-square(cos_angle(detector_coord_A - scatter_point,
													detector_coord_B - scatter_point)));

}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
inline
float dif_polarized_cross_section(const float cos_theta1, const float cos_theta2, const float cos_phi, float energy1, float energy2)
{ 
  const float sin_theta1_2= 1-cos_theta1*cos_theta1 ;
  const float sin_theta2_2= 1-cos_theta2*cos_theta2 ;
  const float sin_phi_2= 1-cos_phi * cos_phi;
  const float P1= 1/(1+(energy1/511.0)*(1-cos_theta1));
  const float P2= 1/(1+(energy2/511.0)*(1-cos_theta2));
  const float gamma1=P1+1/P1;
  const float gamma2=P2+1/P2;
  static const float prefactor_2 =square(9/(2*_PI*(40 - 27*log(3.))))*1E-45;
  // 0.060317979 

  return  prefactor_2*P1*P1*P2*P2* (gamma1*gamma2-gamma1*sin_theta2_2-gamma2*sin_theta1_2+2*sin_theta2_2*sin_theta1_2*sin_phi_2)*_PI ;//*(sqrt(sin_theta1_2*sin_theta2_2));

}

#if 0

inline
float dif_polarized_cross_section(const CartesianCoordinate3D<float>& scatter_point_1,
			const CartesianCoordinate3D<float>& scatter_point_2,
			const CartesianCoordinate3D<float>& detector_coord_A,
			const CartesianCoordinate3D<float>& detector_coord_B, float energy1,float energy2)
{



 const CartesianCoordinate3D<float> temp_diff_scatter_points = scatter_point_2-scatter_point_1;
  const CartesianCoordinate3D<float> temp_diff_A1 = detector_coord_A-scatter_point_1;
  const CartesianCoordinate3D<float> temp_diff_B2 = detector_coord_B-scatter_point_2;

  // phi is the angle of the two planes of scattering
  // L1 is a vector perpendicular to the plane of scatter 1
  // L2 is a vector perpendicular to the plane of scatter 2


    const CartesianCoordinate3D<float> L1=cross_product(temp_diff_scatter_points,temp_diff_A1) ;

  const CartesianCoordinate3D<float> L2=cross_product(temp_diff_scatter_points,temp_diff_B2) ; 
	
  /* compute cosphi, taking care of case where L1 or L2 or 0 vectors.
    This will happen when one of the detectors is on the line 
    between S1 and S2. However, in the dif_polarized_cross_section
    the cosphi is then arbitrary, as it's multiplied with sintheta, 
    which will be 0 in this case.
    So, we set cos_phi to 0 in this case.
  */
  

  const float norm_L1= norm (L1);
  const float norm_L2= norm (L2);

 
  const float cos_phi=
    norm_L1<1 || norm_L2 <1
    ? 0.F
    : inner_product(L1,L2)/(norm_L1*norm_L2);
  


 return
   dif_polarized_cross_section( -cos_angle(detector_coord_A - scatter_point_1, scatter_point_2  - scatter_point_1),-cos_angle(scatter_point_1 - scatter_point_2, detector_coord_B - scatter_point_2), cos_phi, energy1, energy2);
}

#endif


inline
float dif_polarized_cross_section_sin(const float cos_theta1, const float cos_theta2, const float cos_phi, float energy1, float energy2)
{ 
 
const float sin_theta1_2= 1-cos_theta1*cos_theta1 ;
  const float sin_theta2_2= 1-cos_theta2*cos_theta2 ;
  return  dif_polarized_cross_section(cos_theta1,cos_theta2, cos_phi,energy1,energy2)*(sqrt(sin_theta1_2*sin_theta2_2));

}

#if 0

inline
float dif_polarized_cross_section_sin(const CartesianCoordinate3D<float>& scatter_point_1,
				      const CartesianCoordinate3D<float>& scatter_point_2,
				      const CartesianCoordinate3D<float>& detector_coord_A,
				      const CartesianCoordinate3D<float>& detector_coord_B ,float energy1, float energy2)
{

 const CartesianCoordinate3D<float> temp_diff_scatter_points = scatter_point_2-scatter_point_1;
  const CartesianCoordinate3D<float> temp_diff_A1 = detector_coord_A-scatter_point_1;
  const CartesianCoordinate3D<float> temp_diff_B2 = detector_coord_B-scatter_point_2;

 

    const CartesianCoordinate3D<float> L1=cross_product(temp_diff_scatter_points,temp_diff_A1) ;

  const CartesianCoordinate3D<float> L2=cross_product(temp_diff_scatter_points,temp_diff_B2) ; 
       

  const float norm_L1= norm (L1);
  const float norm_L2= norm (L2);

  const float cos_phi=
    norm_L1<1 || norm_L2 <1
    ? 0.F
    : inner_product(L1,L2)/(norm_L1*norm_L2);
  



  return dif_polarized_cross_section_sin(-cos_angle(detector_coord_A - scatter_point_1, scatter_point_2  - scatter_point_1),-cos_angle(scatter_point_1 - scatter_point_2, detector_coord_B - scatter_point_2), cos_phi, energy1, energy2)  ;

}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////





inline
float
energy_after_scatter(const float cos_theta, const float energy)
{
  return energy/(1+(energy/511.0)*(1-cos_theta));   // For an arbitrary energy 
}
inline
float
energy_after_scatter_511keV(const float cos_theta)
{
  return 511/(2-cos_theta); // for a given energy, energy := 511 keV
}



inline
float dif_cross_section_511keV(const float cos_theta)
{
  const float sin_theta2= 1-cos_theta*cos_theta ;
  const float P= 1/(2-cos_theta);
  return(  Re*Re/2*P* (1 - P*sin_theta2 + P*P));
}


inline
float dif_cross_section_sin_511keV(const float cos_theta)
{
  return dif_cross_section_sin(cos_theta, 511);
}


#if 0


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
float dif_cross_section_sin_511keV(const CartesianCoordinate3D<float>& scatter_point,
	  const CartesianCoordinate3D<float>& detector_coord_A,
	  const CartesianCoordinate3D<float>& detector_coord_B)
{
  /* the scatter angle theta  should be 0 when the 2 detectors 
     are at opposite sides of the scatter point. So, we need
     to set cos_theta equal to the following (with a minus sign!)
	 */
  return
    dif_cross_section_511keV(-cos_angle(detector_coord_A - scatter_point,
					detector_coord_B - scatter_point))*sqrt(1-square(cos_angle(detector_coord_A - scatter_point,
					detector_coord_B - scatter_point))); 
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline
float dif_polarized_cross_section_511keV(const float cos_theta1, const float cos_theta2, const float cos_phi)
{
  return dif_polarized_cross_section(cos_theta1, cos_theta2, cos_phi, 511, 511) ;
}

inline
float dif_polarized_cross_section_sin_511keV(const float cos_theta1, const float cos_theta2, const float cos_phi)
{ 
 
  const float sin_theta1_2= 1-cos_theta1*cos_theta1 ;
  const float sin_theta2_2= 1-cos_theta2*cos_theta2 ;
  return  dif_polarized_cross_section_511keV(cos_theta1, cos_theta2, cos_phi)*(sqrt(sin_theta1_2*sin_theta2_2));

}

inline
float dif_polarized_cross_section_511keV(const CartesianCoordinate3D<float>& scatter_point_1,
                                         const CartesianCoordinate3D<float>& scatter_point_2,
                                         const CartesianCoordinate3D<float>& detector_coord_A,
                                         const CartesianCoordinate3D<float>& detector_coord_B)
{

const CartesianCoordinate3D<float> temp_diff_scatter_points = scatter_point_2-scatter_point_1;
  const CartesianCoordinate3D<float> temp_diff_A1 = detector_coord_A-scatter_point_1;
  const CartesianCoordinate3D<float> temp_diff_B2 = detector_coord_B-scatter_point_2;

  // phi is the angle of the two planes of scattering
  // L1 is a vector perpendicular to the plane of scatter 1
  // L2 is a vector perpendicular to the plane of scatter 2


    const CartesianCoordinate3D<float> L1=cross_product(temp_diff_scatter_points,temp_diff_A1) ;

  const CartesianCoordinate3D<float> L2=cross_product(temp_diff_scatter_points,temp_diff_B2) ; 
	
  /* compute cosphi, taking care of case where L1 or L2 or 0 vectors.
    This will happen when one of the detectors is on the line 
    between S1 and S2. However, in the dif_polarized_cross_section
    the cosphi is then arbitrary, as it's multiplied with sintheta, 
    which will be 0 in this case.
    So, we set cos_phi to 0 in this case.
  */
  

  const float norm_L1= norm (L1);
  const float norm_L2= norm (L2);

 
  const float cos_phi=
    norm_L1<1 || norm_L2 <1
    ? 0.F
    : inner_product(L1,L2)/(norm_L1*norm_L2);
  
  return dif_polarized_cross_section_511keV(-cos_angle(detector_coord_A - scatter_point_1, scatter_point_2  - scatter_point_1),-cos_angle(scatter_point_1 - scatter_point_2, detector_coord_B - scatter_point_2), cos_phi) ; 
}  


inline
float dif_polarized_cross_section_sin_511keV(const CartesianCoordinate3D<float>& scatter_point_1,
                                         const CartesianCoordinate3D<float>& scatter_point_2,
                                         const CartesianCoordinate3D<float>& detector_coord_A,
                                         const CartesianCoordinate3D<float>& detector_coord_B)
{

const CartesianCoordinate3D<float> temp_diff_scatter_points = scatter_point_2-scatter_point_1;
  const CartesianCoordinate3D<float> temp_diff_A1 = detector_coord_A-scatter_point_1;
  const CartesianCoordinate3D<float> temp_diff_B2 = detector_coord_B-scatter_point_2;

 

    const CartesianCoordinate3D<float> L1=cross_product(temp_diff_scatter_points,temp_diff_A1) ;

  const CartesianCoordinate3D<float> L2=cross_product(temp_diff_scatter_points,temp_diff_B2) ; 
       

  const float norm_L1= norm (L1);
  const float norm_L2= norm (L2);

  const float cos_phi=
    norm_L1<1 || norm_L2 <1
    ? 0.F
    : inner_product(L1,L2)/(norm_L1*norm_L2);
  


  return dif_polarized_cross_section_511keV(-cos_angle(detector_coord_A - scatter_point_1, scatter_point_2  - scatter_point_1),-cos_angle(scatter_point_1 - scatter_point_2, detector_coord_B - scatter_point_2), cos_phi)*sqrt(1-square(cos_angle(detector_coord_A - scatter_point_1, scatter_point_2  - scatter_point_1))*sqrt(1-square(cos_angle(scatter_point_1 - scatter_point_2, detector_coord_B - scatter_point_2)))  );

}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

inline
float total_cross_section(const float energy)
{
  const float a= energy/511.0;
  const float l= log(1.0+2.0*a); 
    const double sigma0= 6.65E-25;   // sigma0=8*pi*a*a/(3*m*m)
    return( 0.75*sigma0  * ( (1.0+a)/(a*a)*( 2.0*(1.0+a)/(1.0+2.0*a)- l/a ) + l/(2.0*a) - (1.0+3.0*a)/(1.0+2.0*a)/(1.0+2.0*a) ) ); // Klein - Nishina formula = sigma / sigma0
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/*
NOTE THAT YOU MIGHT NEED TO CHANGE THE TOTAL CROSS SECTION ????????????
inline
float total_polarized_cross_section(const float energy)
{
//const float a= energy/511.0;
//const float l= log(1.0+2.0*a); 
//const double sigma0= 6.65E-25;   
//return( 0.75*sigma0  * ( ((2*a*(2+a*(1+a)*(8+a)))/((1+2*a)*(1+2*a))) + (-2 + (-2+a)*a)*l )/(a*a*a) );            */
 
////////////////////////////////////////////////////////////////////////////////////////////////////


inline
float total_cross_section_relative_to_511keV(const float energy)
{
  const float a= energy/511.0;
  static const float prefactor = 9/(-40 + 27*log(3.)); //Klein-Nishina formula for a=1 & devided with 0.75 == (40 - 27*log(3)) / 9

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
    sqrt(2.*energy*reference_energy)*resolution/2.35482;  // 2.35482=2 * sqrt( 2 * ( log(2) )
  
  // sigma_times_sqrt2= sqrt(2) * sigma   // resolution proportional to FWHM 	
  
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

