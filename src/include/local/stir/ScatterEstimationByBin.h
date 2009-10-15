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
class ViewSegmentNumbers;

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

  // TODO write_log can't be const because parameter_info isn't const
  void
    write_log(const double simulation_time, 
	      const float total_scatter);

 protected:
  void set_defaults();
  void initialise_keymap();
  bool post_processing();
  
  float attenuation_threshold;

  bool random;
  bool use_cache;
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
  shared_ptr<ProjData> output_proj_data_sptr; 

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

  //! fill in scatt_points_vector and scatter_volume       
  void 
    sample_scatter_points();
	

  inline float 
    detection_efficiency(const float energy) const;

  double
    detection_efficiency_no_scatter(const unsigned det_num_A, 
				    const unsigned det_num_B) const;

  //! computes scatter for one viewgram
  /*! \return total scatter estimated for this viewgram */
  virtual double
    process_data_for_view_segment_num(const ViewSegmentNumbers& vs_num);

  static
    float 
    integral_between_2_points(const DiscretisedDensity<3,float>& density,
			      const CartesianCoordinate3D<float>& point1, 
			      const CartesianCoordinate3D<float>& point2);

  float 
    exp_integral_over_attenuation_image_between_scattpoint_det (const CartesianCoordinate3D<float>& scatter_point, 
								const CartesianCoordinate3D<float>& detector_coord);
	


  float 
    integral_over_activity_image_between_scattpoint_det (const CartesianCoordinate3D<float>& scatter_point,
							 const CartesianCoordinate3D<float>& detector_coord);
  
 
    
  float 
    cached_integral_over_activity_image_between_scattpoint_det(const unsigned scatter_point_num, 
							       const unsigned det_num);
  
  float 
    cached_exp_integral_over_attenuation_image_between_scattpoint_det(const unsigned scatter_point_num, 
								      const unsigned det_num);

  
	
  float 
    single_scatter_estimate_for_one_scatter_point(const std::size_t scatter_point_num,
					   const unsigned det_num_A, 
					   const unsigned det_num_B);


  void
    single_scatter_estimate(double& scatter_ratio_singles,
			    const unsigned det_num_A, 
			    const unsigned det_num_B);

      
  virtual double
    scatter_estimate(const unsigned det_num_A, 
		     const unsigned det_num_B);
  

      
 public:
  static
    inline float 
    dif_cross_section(const float cos_theta, float energy);
	
	
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
    max_cos_angle(const float low, const float approx, const float resolution_at_511keV);

  static
    inline float 
    energy_lower_limit(const float low, const float approx, const float resolution_at_511keV);
 protected:
	  
  inline float  
    compute_emis_to_det_points_solid_angle_factor(const CartesianCoordinate3D<float>& emis_point,
						  const CartesianCoordinate3D<float>& detector_coord) ;

 private:
  Array<2,float> cached_activity_integral_scattpoint_det;
  Array<2,float> cached_attenuation_integral_scattpoint_det;
  void initialise_cache_for_scattpoint_det();

};


END_NAMESPACE_STIR

#include "local/stir/ScatterEstimationByBin.inl"

#endif
