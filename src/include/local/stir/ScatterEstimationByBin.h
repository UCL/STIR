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
  //! reference used when specifying the energy
  float reference_energy; 
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

  enum image_type{act_image_type, att_image_type};
  struct ScatterPoint
  { 
    CartesianCoordinate3D<float> coord;
    float mu_value;
  };
  
  std::vector< ScatterPoint> scatt_points_vector;
  float scatter_volume;
// next needs to be mutable because find_in_detection_points_vector is const
  mutable std::vector<CartesianCoordinate3D<float> > detection_points_vector;
  int total_detectors;

  // fill in scatt_points_vector and scatter_volume       
  void 
    sample_scatter_points();
	

  inline float 
    detection_efficiency(const float energy);
	
  static
    float 
    integral_between_2_points(const DiscretisedDensity<3,float>& density,
			      const CartesianCoordinate3D<float>& scatter_point, 
			      const CartesianCoordinate3D<float>& detector_coord);

  float 
    exp_integral_over_attenuation_image_between_scattpoint_det (const CartesianCoordinate3D<float>& scatter_point, 
								const CartesianCoordinate3D<float>& detector_coord);
	


  float 
    integral_over_activity_image_between_scattpoint_det (const CartesianCoordinate3D<float>& scatter_point,
							 const CartesianCoordinate3D<float>& detector_coord);
  
 

  float 
    integral_over_activity_image_between_scattpoints (const CartesianCoordinate3D<float>& scatter_point_1, 
							  const CartesianCoordinate3D<float>& scatter_point_2);

  float 
    exp_integral_over_attenuation_image_between_scattpoints (const CartesianCoordinate3D<float>& scatter_point_1, 
							     const CartesianCoordinate3D<float>& scatter_point_2);
    
  float 
    cached_integral_over_activity_image_between_scattpoint_det(const unsigned scatter_point_num, 
							       const unsigned det_num);
  
  float 
    cached_exp_integral_over_attenuation_image_between_scattpoint_det(const unsigned scatter_point_num, 
								      const unsigned det_num);

  float
    cached_integral_over_activity_image_between_scattpoints(unsigned scatter_point_num_1, 
									  unsigned scatter_point_num_2);	
  
  float
    cached_exp_integral_over_attenuation_image_between_scattpoints(unsigned scatter_point_num_1, 
									     unsigned scatter_point_num_2);	
  
  
	
  float 
    scatter_estimate_for_one_scatter_point(const std::size_t scatter_point_num,
					   const unsigned det_num_A, 
					   const unsigned det_num_B);
  void
    scatter_estimate_for_two_scatter_points(double& scatter_ratio_11,
					    double& scatter_ratio_02,
					    const std::size_t scatter_point_1_num, 
					    const std::size_t scatter_point_2_num, 
					    const unsigned det_num_A, 
					    const unsigned det_num_B);
	
	
  void
    scatter_estimate_for_all_scatter_points(double& scatter_ratio_01,
					    double& scatter_ratio_11,
					    double& scatter_ratio_02,
					    const unsigned det_num_A, 
					    const unsigned det_num_B);

      
  float
    scatter_estimate_for_all_scatter_points(const unsigned det_num_A, 
					    const unsigned det_num_B);
  

      
  float 
    scatter_estimate_for_none_scatter_point(const unsigned det_num_A, 
					    const unsigned det_num_B);
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

private:
  Array<2,float> cached_activity_integral_scattpoint_det;
  Array<2,float> cached_attenuation_integral_scattpoint_det;
  void initialise_cache_for_scattpoint_det();

  Array<2,float> cached_activity_integral_scattpoints;
  Array<2,float> cached_attenuation_integral_scattpoints;
  void initialise_cache_for_scattpoints();

  bool use_solid_angle_for_points;
};


END_NAMESPACE_STIR

#include "local/stir/ScatterEstimationByBin.inl"

#endif
