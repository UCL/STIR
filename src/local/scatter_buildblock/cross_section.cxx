//
// $Id$
//
/*!
  \file
  \ingroup scatter
  \brief Implementations of functions defined in Scatter.h

  \author Charalampos Tsoumpas
  \author Pablo Aguiar
  \author Kris Thielemans

  $Date$
  $Revision$

    Copyright (C) 2004- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details

//COPYWRIGHT BY WERLING ETC..............................................

*/
#include "local/stir/Scatter.h"
#include <cmath>

using namespace std;

START_NAMESPACE_STIR


float dif_cross_section(const CartesianCoordinate3D<float>& scatter_point,
	  const CartesianCoordinate3D<float>& detector_coord_A,
	  const CartesianCoordinate3D<float>& detector_coord_B, float energy)
{
  const CartesianCoordinate3D<float> scatter_point_det_A = detector_coord_A - scatter_point ;  //vector from detector_A to scatter_point
  const CartesianCoordinate3D<float> scatter_point_det_B = detector_coord_B - scatter_point ;  //vector from detector_B to scatter_point

  const float scalar_product = 
	inner_product(scatter_point_det_A, scatter_point_det_B);

  const float dAS = norm(detector_coord_A-scatter_point); // the distance of the detector_A to scatter_point S
  const float dBS = norm(detector_coord_B-scatter_point); // the distance of the detector_B to scatter_point S

  const float cos_theta = scalar_product/(dAS*dBS) ;
  const float sin_theta2= 1-cos_theta*cos_theta ;
  const float P= 1/(1+(energy/511.0)*(1-cos_theta));


  return(  Re*Re/2* (P - P*P*sin_theta2 + P*P*P) );
}

float dif_cross_section_511keV(const CartesianCoordinate3D<float>& scatter_point,
	  const CartesianCoordinate3D<float>& detector_coord_A,
	  const CartesianCoordinate3D<float>& detector_coord_B)
{
  const CartesianCoordinate3D<float> scatter_point_det_A = detector_coord_A - scatter_point ;  //vector from detector_A to scatter_point
  const CartesianCoordinate3D<float> scatter_point_det_B = detector_coord_B - scatter_point ;  //vector from detector_B to scatter_point

  const float scalar_product = 
	inner_product(scatter_point_det_A, scatter_point_det_B);

  const float dAS = norm(detector_coord_A-scatter_point); // the distance of the detector_A to scatter_point S
  const float dBS = norm(detector_coord_B-scatter_point); // the distance of the detector_B to scatter_point S

  const float cos_theta = scalar_product/(dAS*dBS) ;
  const float sin_theta2= 1-cos_theta*cos_theta ;
  const float P= 1/(2-cos_theta);


  return(  Re*Re/2* (P - P*P*sin_theta2 + P*P*P) );
}


float total_cross_section(float energy)
{
  const float a= energy/511.0;
  const float l= log(1.0+2.0*a);
  const double sigma0= 6.65E-25;

  return( 0.75*sigma0  * ( (1.0+a)/(a*a)*( 2.0*(1.0+a)/(1.0+2.0*a)- l/a ) + l/(2.0*a) - (1.0+3.0*a)/(1.0+2.0*a)/(1.0+2.0*a) ) );
}

END_NAMESPACE_STIR

