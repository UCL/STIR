//
// $Id$
//
/*!
  \file
  \ingroup scatter
  \brief Implementations of functions defined in scatter.h

  \author Charalampos Tsoumpas
  \author Pablo Aguiar
  \author Kris Thielemans

  $Date$
  $Revision$

    Copyright (C) 2004- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details

//COPYWRIGHT BY WERLING ETC..............................................

*/
#include "local/stir/scatter.h"
#include <cmath>

--------------------------------------------------------------------------------------
/*!						
  \ingroup scatter
  \brief computes the differential cross section
   Better inline?????

  This function computes the differential cross section
	for Compton scatter, based on the Klein-Nishina-Formula
	(cf. http://www.physik.uni-giessen.de/lehre/fpra/compton/ )

	theta:	azimuthal angle between incident and scattered photon
	energy:	energy of incident photon ( in keV )
*/  	
float dif_cross_section(const CartesianCoordinate3D<float>& scatter_point,
	                      const CartesianCoordinate3D<float>& detector_coord,
	                      const float energy);
/*!						
  \ingroup scatter
  \brief computes the total cross section
   Better inline?????

  n  This function computes the total cross section
	for Compton scatter, based on the Klein-Nishina-Formula
	(cf.Am. Institute of Physics Handbook, page 87, chapter 8, formula 8f-22 )

	energy:	energy of incident photon ( in keV )

*/
float total_cross_section(float energy);

----------------------------------------------------------------------------------------

using namespace std;

START_NAMESPACE_STIR


float dif_cross_section(const CartesianCoordinate3D<float>& scatter_point,
	  const CartesianCoordinate3D<float>& detector_coord, float energy)
{
  const CartesianCoordinate3D<float> origin(0,0,0);
  scatter_point_det = detector_coord - scatter_point ;

  const float s = two_points_distance(detector_coord,origin); // the distance of the detector from the origin
  const float d = two_points_distance(scatter_point,origin);  // the distance of the scatter_point from the origin
  const float sd = two_points_distance(scatter_point_det,origin); // the distance of the scatter_point from the detector

  const float cos_theta = (s*s+sd*sd-d*d)/(2*sd*s)  ;
  const float sin_theta2= 1-cos_theta*cos_theta;
  const float P= 1/(1+energy/511.0*(1-cos_theta));

  return(  r0*r0/2* (P - P*P*sin_theta2 + P*P*P) );
}


float total_cross_section(float energy)
{
  const float a= energy/511.0;
  const float l= log(1.0+2.0*a);
  const float sigma0= 6.65E-25;

  return( 0.75*sigma0  * ( (1.0+a)/(a*a)*( 2.0*(1.0+a)/(1.0+2.0*a)- l/a ) + l/(2.0*a) - (1.0+3.0*a)/(1.0+2.0*a)/(1.0+2.0*a) ) );
}

END_NAMESPACE_STIR
