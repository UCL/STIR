//
// $Id$
//
#ifndef __stir_ScatterEstimationByBin_H__
#define __stir_ScatterEstimationByBin_H__

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
  \brief Definition of class stir::ScatterEstimationByBin.
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/ProjData.h"
#include "stir/ParsingObject.h"
#include <vector>
#include <cmath>
#include "stir/CartesianCoordinate3D.h"
START_NAMESPACE_STIR

class Succeeded;
class ProjDataInfoCylindricalNoArcCorr;

#if 0
struct ScatterPoint
{ 
  CartesianCoordinate3D<float> coord;
  float mu_value;
};

// TODO move to the class
extern std::vector< ScatterPoint> scatt_points_vector;
extern std::vector<CartesianCoordinate3D<float> > detection_points_vector;
extern int total_detectors;
#endif

/*!
  \ingroup scatter
 \brief Estimate of the scatter probability for a number of scatter points.
*/
class ScatterEstimationByBin : public ParsingObject
{
 public:
  //! Default constructor (calls set_defaults())
  ScatterEstimationByBin();
  Succeeded process_data();
  /*  virtual float scatter_estimate(
			 const unsigned det_num_A, 
			 const unsigned det_num_B);
  */

  // TODO write_log can't be const because parameter_info isn't const
  void
    write_log(const double simulation_time, 
	      const float total_scatter);
  void set_defaults();

 protected:
  void initialise_keymap();
  bool post_processing();
  
  float attenuation_threshold;

  bool random;
  bool use_cache;
  bool use_polarization;
  int scatter_level;
  bool write_scatter_orders_in_separate_files;
  
  float energy_resolution;
  float lower_energy_threshold;
  float upper_energy_threshold;

  std::string activity_image_filename;
  std::string density_image_filename;
  std::string density_image_for_scatter_points_filename;
  std::string template_proj_data_filename;
  std::string output_proj_data_filename;

  shared_ptr<DiscretisedDensity<3,float> > density_image_for_scatter_points_sptr;
  shared_ptr<DiscretisedDensity<3,float> > density_image_sptr;
  shared_ptr<DiscretisedDensity<3,float> > activity_image_sptr;
  shared_ptr<ProjData> output_proj_data_sptr; // currently no scatter
  shared_ptr<ProjData> output_proj_data_00_sptr;
  shared_ptr<ProjData> output_proj_data_01_sptr;
  shared_ptr<ProjData> output_proj_data_11_sptr;
  shared_ptr<ProjData> output_proj_data_02_sptr;

virtual 
  void
  find_detectors(unsigned& det_num_A, unsigned& det_num_B, const Bin& bin) const; 

 unsigned 
  find_in_detection_points_vector(const CartesianCoordinate3D<float>& coord) const;
// private:
  const ProjDataInfoCylindricalNoArcCorr * proj_data_info_ptr;
  CartesianCoordinate3D<float>  shift_detector_coordinates_to_origin;

  /*************** functions that do the work **********/

#define NEWSCATTER
#ifdef NEWSCATTER
enum image_type{act_image_type, att_image_type};
struct ScatterPoint
{ 
  CartesianCoordinate3D<float> coord;
  float mu_value;
};

std::vector< ScatterPoint> scatt_points_vector;
// next needs to be mutable because find_in_detection_points_vector is const
mutable std::vector<CartesianCoordinate3D<float> > detection_points_vector;
int total_detectors;
       
  void 
    sample_scatter_points(const DiscretisedDensityOnCartesianGrid<3,
			  float>& attenuation_map,
			  int & scatt_points,
			  const float att_threshold, 
			  const bool random);
	

	inline float 
	  detection_efficiency( const float low, 
				const float high, 
				const float energy, 
				const float reference_energy, 
				const float resolution);
	

	float 
	  integral_scattpoint_det (const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
				   const CartesianCoordinate3D<float>& scatter_point, 
				   const CartesianCoordinate3D<float>& detector_coord);
	


	float 
	  integral_over_activity_image_between_scattpoint_det (const DiscretisedDensityOnCartesianGrid<3,float>& activity_image,
							       const CartesianCoordinate3D<float>& scatter_point,
							       const CartesianCoordinate3D<float>& detector_coord);
	
 

	float 
	  integral_over_activity_image_between_scattpoints (const DiscretisedDensityOnCartesianGrid<3,float>& activity_image,
							    const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
							    const CartesianCoordinate3D<float>& scatter_point_1, 
							    const CartesianCoordinate3D<float>& scatter_point_2);
	

	float 
	  cached_factors(const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
			 const unsigned scatter_point_num, 
			 const unsigned det_num,
			 const image_type input_image_type);
	
	
	float 
	  cached_factors_2(const DiscretisedDensityOnCartesianGrid<3,float>& discretised_image,
			   const unsigned scatter_point_1_num, 
			   const unsigned scatter_point_2_num,
			   const image_type input_image_type);
	

	
	float 
	scatter_estimate_for_one_scatter_point(const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
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


      
	float 
	  scatter_estimate_for_none_scatter_point(const DiscretisedDensityOnCartesianGrid<3,float>& image_as_activity,
						  const DiscretisedDensityOnCartesianGrid<3,float>& image_as_density,
						  const unsigned det_num_A, 
						  const unsigned det_num_B,
						  const float lower_energy_threshold, 
						  const float upper_energy_threshold,
						  const float resolution);
 public:
	static
	inline float 
	  dif_cross_section(const float cos_theta, float energy);
	
	static
	inline float 
	  dif_polarized_cross_section(const float cos_theta1, const float cos_theta2, const float cos_phi, float energy1, float energy2 );
	
	static
	inline float 
	  total_cross_section(float energy);
	
	static
	inline float
	  energy_after_scatter(const float cos_theta, const float energy);

	static
	inline float
	  energy_after_scatter_511keV(const float cos_theta);

	static
	inline float 
	  total_cross_section_relative_to_511keV(const float energy);
	
	static
	inline float 
	  max_cos_angle(const float low, const float approx, const float resolution);

	static
	inline float 
	energy_lower_limit(const float low, const float approx, const float resolution);
 protected:
	inline float
	  compute_emis_to_scatter_points_solid_angle_factor_doubles11(const CartesianCoordinate3D<float>& scatter_point_1,
								      const CartesianCoordinate3D<float>& scatter_point_2,
								      const CartesianCoordinate3D<float>& emis_point) ;
	
	
	inline float 
	  compute_sc1_to_sc2_solid_angle_factor_doubles20(const CartesianCoordinate3D<float>& scatter_point_1,
							  const CartesianCoordinate3D<float>& scatter_point_2) ;
	  
	  inline float  
	    compute_emis_to_det_points_solid_angle_factor(const CartesianCoordinate3D<float>& emis_point,
							  const CartesianCoordinate3D<float>& detector_coord) ;
	  
#endif

};


END_NAMESPACE_STIR

#include "local/stir/ScatterEstimationByBin.inl"

#endif
