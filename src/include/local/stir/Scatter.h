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
#include "stir/round.h"
#include "stir/ProjData.h"
#include <vector>
#include <cmath>
//#define point_sample_step 1

START_NAMESPACE_STIR

const double Qe = 1.602E-19;   
const double Me = 9.109E-31;
const double C = 2.997E8;
const double Re = 2.818E-13;
enum image_type{act_image_type, att_image_type};

template <class coordT> class CartesianCoordinate3D;
class ProjDataInfoCylindricalNoArcCorr;



struct ScatterPoint
{ 
  CartesianCoordinate3D<float> coord;
  float mu_value;
};

extern std::vector< ScatterPoint> scatt_points_vector;
extern std::vector<CartesianCoordinate3D<float> > detection_points_vector;
extern int total_detectors;

/*!
   \ingroup scatter
   \brief samples the scatters points randomly in the attenuation_map
   is used to sample points of the attenuation_map which is the transmission image.
   The sampling is uniformly randomly with a threshold that is defined by att_threshold,
   that only above of this the points are sampled. It returns a vector containing the 
   scatter points location in mm units.
   scatt_points are the wanted sample scatter points and is used as a reference, so 
   that when the sample scatter points are less than scatt_points, the sample scatter 
   points are assigned to the scatt_points, giving a warning.
*/  

void
sample_scatter_points(
		const DiscretisedDensityOnCartesianGrid<3,float>& attenuation_map,
		int & scatt_points,
        const float att_threshold, 
		const bool random);

inline 
float calc_efficiency( float low, float high, float energy, float resolution );


/*!						  
  \ingroup scatter
  \brief Implementations of functions defined in Scatter.h
     Function calculates the integral along LOR in a image (attenuation or 
	 emission). From scatter point to detector coordinate.
	 For the start voxel, the intersection length of the LOR with the whole 
	 voxel is computed, not just from the start_point to the next edge. 
	 The same is true for the end voxel.
*/
float integral_scattpoint_det (
	  const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
	  const CartesianCoordinate3D<float>& scatter_point, 
	  const CartesianCoordinate3D<float>& detector_coord);  

float  cached_factors(const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
	  				  const unsigned scatter_point_num, 
					  const unsigned det_num,
					  const image_type input_image_type);
/*!
  \ingroup scatter
 \brief 
*/

float scatter_estimate_for_one_scatter_point(
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const std::size_t scatter_point_num, 
	  const unsigned det_num_A, 
	  const unsigned det_num_B,
	  const float lower_energy_threshold, 
	  const float upper_energy_threshold,		
	  const bool use_cosphi,
	  const bool use_cache);
/*!	\name Klein-Nishina functions					
  \ingroup scatter
  \brief computes the differential cross section
 
  These functions computes the differential cross section
	for Compton scatter, based on the Klein-Nishina-Formula
	(cf. http://www.physik.uni-giessen.de/lehre/fpra/compton/ )

	theta:	azimuthal angle between incident and scattered photon
	energy:	energy of incident photon ( in keV )
*/ 
//@{
inline
float dif_cross_section_511keV(const float cos_theta);
 	
inline
float dif_cross_section(const CartesianCoordinate3D<float>& scatter_point,
	                    const CartesianCoordinate3D<float>& detector_coordA,
				        const CartesianCoordinate3D<float>& detector_coordB,
	                    const float energy);	
inline
float dif_cross_section_511keV(const CartesianCoordinate3D<float>& scatter_point,
	                           const CartesianCoordinate3D<float>& detector_coordA,
				          	   const CartesianCoordinate3D<float>& detector_coordB);

/*!						
  \ingroup scatter
  \brief computes the total cross section

  This function computes the total cross section
	for Compton scatter, based on the Klein-Nishina-Formula
	(cf.Am. Institute of Physics Handbook, page 87, chapter 8, formula 8f-22 )

	energy:	energy of incident photon ( in keV )

*/ 
inline
float total_cross_section(float energy);
//@}

inline
float energy_after_scatter_511keV(const float cos_theta);


/*!
  \ingroup scatter
  \brief 

*/
float scatter_estimate_for_all_scatter_points(
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	  const unsigned det_num_A, 
	  const unsigned det_num_B,
	  const float lower_energy_threshold, 
	  const float upper_energy_threshold,		
	  const bool use_cosphi,
	  const bool use_cache);
/*!
  \ingroup scatter
  \brief 

*/
void scatter_viewgram( 
	ProjData& proj_data,
	const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
    int& scatt_points, const float att_threshold, 
	const float lower_energy_threshold, const float upper_energy_threshold,		
	const bool use_cosphi, const bool use_cache, const bool random);


// Functions that could be in the BasicCoordinate Class
template<int num_dimensions>
inline 
BasicCoordinate<num_dimensions,float> convert_int_to_float(const BasicCoordinate<num_dimensions,int>& cint);

/* !\ingroup scatter
     \brief Temporary implementation of the error function for the Visual C++
*/
#ifdef _MSC_VER
extern "C"  double erf(const double x);
#endif

/* !\ingroup scatter
     \brief Temporary implementation of writing log information
*/
void writing_log(const DiscretisedDensityOnCartesianGrid<3,float>& activity_image,
				 const DiscretisedDensityOnCartesianGrid<3,float>& density_image,
				 const ProjDataInfoCylindricalNoArcCorr * proj_data_info_ptr,
				 const float given_attenuation_threshold,
				 const int total_scatt_points,
				 const float lower_energy_threshold, 
				 const float upper_energy_threshold,		
				 const bool use_cosphi,
				 const bool use_cache,
				 const bool random, 
				 const char *argv[]);

/* !\ingroup scatter
     \brief Temporary implementation of writing time information
*/
void writing_time(const int simulation_time, const int scatt_points_vector_size);

void writing_time(const double simulation_time, const int scatt_points_vector_size);


END_NAMESPACE_STIR

#include "local/stir/Scatter.inl"
