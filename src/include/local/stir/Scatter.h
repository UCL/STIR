//
// $Id$
//
/*!
  \file
  \ingroup scatter
  \brief A collection of functions to measure the single scatter component
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  \author Pablo Aguiar

  $Date$
  $Revision$

 */
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/


#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/common.h"
#include "stir/ProjData.h"
#include <vector>
#include <cmath>


START_NAMESPACE_STIR

const double Qe=1.602E-19;   
const double Me=9.109E-31;
const double C=2.997E8;
const double Re=2.818E-13;


template <class coordT> class CartesianCoordinate3D;

/*!
   \ingroup scatter
   \brief samples the scatters points randomly in the attenuation_map
   is used to sample points of the attenuation_map which is the transmission image.
   The sampling is uniformly randomly with a threshold that is defined by att_threshold,
   that only above of this the points are sampled. It returns a vector containing the 
   scatter points location in voxel units.
   max_scatt_points are the wanted sample scatter points and is used as a reference, so 
   that when the sample scatter points are less than max_scatt_points, the sample scatter 
   points are assigned to the max_scatt_points, giving a warning.
*/  

std::vector<CartesianCoordinate3D<float> > 
sample_scatter_points(
		const DiscretisedDensityOnCartesianGrid<3,float>& attenuation_map,
		int & max_scatt_points,
        const float att_threshold);

/*!						  
  \ingroup scatter
  \brief Implementations of functions defined in scatter.h
     Function calculates the integral along LOR in a image (attenuation or 
	 emission). From scatter point to detector coordinate.
	 For the start voxel, the intersection length of the LOR with the whole 
	 voxel is computed, not just from the start_point to the next edge. 
	 The same is true for the end voxel.
*/
float integral_scattpoint_det (
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const CartesianCoordinate3D<float>& scatter_point, 
	  const CartesianCoordinate3D<float>& detector_coord);  

/*!
  \ingroup scatter
  \brief 

*/

float scatter_estimate_for_one_scatter_point(
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const CartesianCoordinate3D<float>& scatter_point, 
	  const CartesianCoordinate3D<float>& detector_coord_A, 
	  const CartesianCoordinate3D<float>& detector_coord_B);
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
	                     const CartesianCoordinate3D<float>& detector_coordA,
				          		 const CartesianCoordinate3D<float>& detector_coordB,
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



/*!
  \ingroup scatter
  \brief 

*/
float scatter_estimate_for_all_scatter_points(
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const std::vector<CartesianCoordinate3D<float> >& scatter_points, 
	  const CartesianCoordinate3D<float>& detector_coord_A, 
	  const CartesianCoordinate3D<float>& detector_coord_B);
/*!
  \ingroup scatter
  \brief 

*/
void scatter_viewgram( 
	ProjData& proj_data,
	const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
    int max_scatt_points, const float att_threshold);


// 3 Functions that will be in the BasicCoordinate Class

inline 
BasicCoordinate<3,float> convert_int_to_float(const BasicCoordinate<3,int>& cint)
	{	  
	  BasicCoordinate<3,float> cfloat;

	  for(int i=1;i<=3;++i)
		  cfloat[i]=(float)cint[i];
	  return cfloat;
	}

inline
BasicCoordinate<3,int> convert_float_to_int(const BasicCoordinate<3,float>& cfloat)
	{	  
	  BasicCoordinate<3,int> cint;
      for(int i=1;i<=3;++i)
		  {
		  	  if(ceil(cfloat[i]) - cfloat[i]<=0.5) 
				  	  cint[i] = (int)ceil(cfloat[i]) ;
			  else 
				  	  cint[i] = (int)cfloat[i] ;
		  }
	  return cint;
	}

inline
float two_points_distance(const BasicCoordinate<3,float>& c1,
						  const BasicCoordinate<3,float>& c2)
	{  
	   BasicCoordinate<3,float> c=c1-c2;
	   return sqrt(c[1]*c[1]+c[2]*c[2]+c[3]*c[3]);
	}



END_NAMESPACE_STIR


