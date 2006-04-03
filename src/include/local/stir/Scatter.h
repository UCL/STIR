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
  \brief A collection of functions to measure the scatter component  
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  \author Pablo Aguiar

  $Date$
  $Revision$
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/round.h"
#include "local/stir/numerics/erf.h"
#include "stir/ProjData.h"
#include <vector>
#include <cmath>
// include this for now just to get at NEWSCATTER
#include "local/stir/ScatterEstimationByBin.h"

START_NAMESPACE_STIR

#ifndef NEWSCATTER

const double Qe = 1.602E-19;  // charge of the electron    
const double Me = 9.109E-31;  // mass of the electron
const double C = 2.997E8;      // light speed
const double Re = 2.818E-13;   // aktina peristrofis electroniou gia to atomo tou H
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

/*!	
   \ingroup scatter
   \brief detection efficiency for a given energy window   
   This function provides a simple model of the interaction of radiation
   with the detector .	
   \param low, high Discriminator bounds of the detector in keV
   \param energy	 Energy of incident photon in keV
   \param resolution Energy resolution of the detector B (between 0 and 1) at the reference energy
   \param reference_energy  Energy where the FWHM is given by \c resolution			  
   The energy spectrum is assumed to be Gaussian. The FWHM is assumed to 
   be proportional to sqrt(energy). This is reasonable given the Poisson 
   statistics of scintillation detectors. The proportionality factor is 
   determined by requiring that FWHM(reference_energy)=resolution*reference_energy.
   This formula is the same as the one used by SIMSET for Simple_PET detector.
*/
//@{
inline 
float detection_efficiency( const float low, const float high, 
			    const float energy, 
			    const float reference_energy, const float resolution);
//@}

/*!						  
  \ingroup scatter
  \brief Implementations of functions defined in Scatter.h
     Function calculates the integral along LOR in a image (attenuation or 
	 emission). From scatter point to detector coordinate.
	 For the start voxel, the intersection length of the LOR with the whole 
	 voxel is computed, not just from the start_point to the next edge. 
	 The same is true for the end voxel.
	 If cached factor is enabled the cached_factors() stores the values 
	 of the two integrals (scatter_point-detection_point) in a static array. 
	 The cached_factors_2() stores the values in of the two integrals 
	 (scatter_point_1-scatter_point_2) in an static array.
*/

float integral_scattpoint_det (const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
			       const CartesianCoordinate3D<float>& scatter_point, 
			       const CartesianCoordinate3D<float>& detector_coord);

float integral_over_activity_image_between_scattpoint_det (const DiscretisedDensityOnCartesianGrid<3,float>& activity_image,
			      const CartesianCoordinate3D<float>& scatter_point,
			      const CartesianCoordinate3D<float>& detector_coord); 

 

float integral_over_activity_image_between_scattpoints (const DiscretisedDensityOnCartesianGrid<3,float>& activity_image,
							const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
							const CartesianCoordinate3D<float>& scatter_point_1, 
							const CartesianCoordinate3D<float>& scatter_point_2);

float  cached_factors(const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
		      const unsigned scatter_point_num, 
		      const unsigned det_num,
		      const image_type input_image_type);
float cached_factors_2(const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
		       const unsigned scatter_point_1_num, 
		       const unsigned scatter_point_2_num,
		       const image_type input_image_type);
/*!
  \ingroup scatter
  \brief Estimate of the scatter probability for a number of scatter points.
*/
//@{
float scatter_estimate_for_one_scatter_point(const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
					     const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
					     const std::size_t scatter_point_num,    // scatter volume misses
					     const unsigned det_num_A, 
					     const unsigned det_num_B,
					     const float lower_energy_threshold, 
					     const float upper_energy_threshold,
					     const float resolution,		
					     const bool use_cache);
void
scatter_estimate_for_two_scatter_points(double& scatter_ratio_11,
					double& scatter_ratio_02,
					const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
					const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
					const std::size_t scatter_point_1_num, 
					const std::size_t scatter_point_2_num, 
					const unsigned det_num_A, 
					const unsigned det_num_B,
					const float lower_energy_threshold, 
					const float upper_energy_threshold,
					const float resolution,		
					const bool use_cache,	
					const bool use_polarization);
void
scatter_estimate_for_all_scatter_points(double& scatter_ratio_01,
					double& scatter_ratio_11,
					double& scatter_ratio_02,
					const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
					const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
					const float scatter_volume,
					const unsigned det_num_A, 
					const unsigned det_num_B,
					const float lower_energy_threshold, 
					const float upper_energy_threshold,
					const float resolution,		
					const bool use_cache, 
					const bool use_polarization,	
					const int scatter_level);

float
scatter_estimate_for_all_scatter_points(const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
					const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
					const unsigned det_num_A, 
					const unsigned det_num_B,
					const float lower_energy_threshold, 
					const float upper_energy_threshold,
					const float resolution,		
					const bool use_cache, 
					const bool use_polarization,
					const int scatter_level);

float scatter_estimate_for_none_scatter_point(const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
					      const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
					      const unsigned det_num_A, 
					      const unsigned det_num_B,
					      const float lower_energy_threshold, 
					      const float upper_energy_threshold,
					      const float resolution);
//@}
/*!
  \ingroup scatter
  \brief Splits the double scatter in two terms: DS_1_1 and DS_0_2+DS_2_0
*/
double scatter_estimate_for_two_scatter_points_splitted(  // splitted 11 case differs from 20 case
							const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
							const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
							const std::size_t scatter_point_1_num, 
							const std::size_t scatter_point_2_num, 
							const unsigned det_num_A, 
							const unsigned det_num_B,
							const float lower_energy_threshold, 
							const float upper_energy_threshold, 
							const float resolution,		
							const bool use_cache, 	
							const bool use_polarization,
							const int split);



float scatter_estimate_for_all_scatter_points_splitted(             // ? all scatter points splitted
						       const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
						       const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
						       const DiscretisedDensityOnCartesianGrid<3,float>& smooth_image_as_density,
						       const unsigned det_num_A, 
						       const unsigned det_num_B,
						       const float lower_energy_threshold, 
						       const float upper_energy_threshold, 
						       const float resolution,		
						       const bool use_cache,
						       const bool use_polarization,
						 
						       const int scatter_level,
						       const int split);
void scatter_viewgram_splitted( 
			       ProjData& proj_data,
			       ProjData& proj_data2,
			       const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
			       const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
			       const DiscretisedDensityOnCartesianGrid<3,float>& smooth_image_as_density,
			       int& scatt_points, const float att_threshold, 
			       const float lower_energy_threshold, const float upper_energy_threshold, 
			       const float resolution, 
			       const bool use_cache,
			       const bool use_polarization, 
			       
			       const int scatter_level, const bool random);

/*!	\name Klein-Nishina functions					
  \ingroup scatter
  \brief computes the differential cross section
  These functions computes the differential cross section
  for Compton scatter, based on the Klein-Nishina-Formula
  (cf. http://www.physik.uni-giessen.de/lehre/fpra/compton/ )
  theta:	azimuthal angle between incident and scattered photon
  energy:	energy of incident photon ( in keV )
  photon energy set to 511keV instead 510.99906keV
*/ 
//@{
inline
float dif_cross_section(const float cos_theta, float energy);

inline 
float dif_polarized_cross_section(const float cos_theta1, const float cos_theta2, const float cos_phi, float energy1, float energy2 );


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
inline
float energy_after_scatter_511keV(const float cos_theta);
inline
float total_cross_section_relative_to_511keV(const float energy);
//@}

/*!	\ingroup scatter
   \brief lower accepting limit of the energy and corresponding scattering angle to speed up the process. 
   
	 These functions return the limit of the Detected Energy or its scattering angle 
	 in order to speed up the simulation. It is set at approx*sigma lower than the lower energy threshold 
	 (low). This means that we keep the 95.45% of the distribution integral of a=2.
   */
//@{
inline
float max_cos_angle(const float low, const float approx, const float resolution);
inline 
float energy_lower_limit(const float low, const float approx, const float resolution);
//@}
/*!  \ingroup scatter
  \brief uses the given proj_data writes the scatter viewgram 
*/
void scatter_viewgram( 
	ProjData& proj_data,
	const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
	const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
	int& scatt_points, const float att_threshold, 
	const float lower_energy_threshold, const float upper_energy_threshold,	const float resolution,	
	const bool use_cache, 
	const bool use_polarization,
	const int scatter_level, const bool random);

/////////////////////////
float att_estimate_for_no_scatter(const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
				  const unsigned det_num_A, 
				  const unsigned det_num_B);

#endif // NEWSCATTER

// give mask_radius_in_mm negative to ignore it
Array<2,float>
  scale_factors_per_sinogram(const ProjData& no_scatter_proj_data, 
			     const ProjData& scatter_proj_data, 
			     const ProjData& att_proj_data, 
			     const float attenuation_threshold,
			     const float mask_radius_in_mm);
Array<2,float>
  scale_factors_per_viewgram(const ProjData& no_scatter_proj_data, 
			     const ProjData& scatter_proj_data, 
			     const ProjData& att_proj_data, 
			     const float attenuation_threshold,
			     const float mask_radius_in_mm);

	void scale_scatter_per_sinogram(
		ProjData& scaled_scatter_proj_data, 		
		const ProjData& scatter_proj_data, 
		const Array<2,float> scale_factor_per_sinogram);


	void scale_scatter_per_viewgram(
		ProjData& scaled_scatter_proj_data, 		
		const ProjData& scatter_proj_data, 
		const Array<2,float> scale_factor_per_viewgram);

/*	float estimate_scale_factor(
		const shared_ptr<ProjData> & no_scatter_proj_data_sptr, 
		const shared_ptr<ProjData> & scatter_proj_data_sptr, 
		const ProjData& att_proj_data, 
		const float attenuation_threshold);*/

	void estimate_att_viewgram(ProjData& proj_data,					  
		const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density);

	/*void substract_scatter(
		ProjData& corrected_scatter_proj_data, 
		const shared_ptr<ProjData> & no_scatter_proj_data_sptr, 
		const shared_ptr<ProjData> & scatter_proj_data_sptr, 
		const ProjData& att_proj_data, 
		const float global_scatter_factor) ;*/

/////////////////////////

#ifndef NEWSCATTER


//#ifdef _MSC_VER
/* !\ingroup scatter
     \brief Temporary implementation of the error function for Visual C++
*/
/*extern "C" double erf(const double x);
#endif
*/
/* !\ingroup scatter
     \brief Temporary implementation of writing log information
	 The log information is written in the statistics.txt file
*/
//@{
	void writing_log(const DiscretisedDensityOnCartesianGrid<3,float>& activity_image,
		 const DiscretisedDensityOnCartesianGrid<3,float>& density_image,
		 const ProjDataInfoCylindricalNoArcCorr * proj_data_info_ptr,
		 const float given_attenuation_threshold,
		 const int total_scatt_points,
		 const float lower_energy_threshold, 
		 const float upper_energy_threshold,
		 const float resolution,
		 const bool use_cache,  
		 const bool use_polarization,
		 const bool random, 
		 const char *argv[]);

void writing_time(const int simulation_time, 
		  const int scatt_points_vector_size,
		  const int scatter_level);
void writing_time(const double simulation_time, 
		  const int scatt_points_vector_size,
		  const int scatter_level, 
		  const float total_scatter);
//@}
#endif // NEWSCATTER

END_NAMESPACE_STIR

#ifndef NEWSCATTER
#include "local/stir/Scatter.inl"
#endif // NEWSCATTER
