//
// $Id$
//
#ifndef __stir_scatter_ScatterEstimationByBin_H__
#define __stir_scatter_ScatterEstimationByBin_H__

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
  \author Nikolaos Dikaios
  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ProjData.h"
#include "stir/ParsingObject.h"
#include "stir/numerics/BSplines.h"
#include <vector>
#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR

class Succeeded;
class ProjDataInfoCylindricalNoArcCorr;
class ViewSegmentNumbers;

/*!
  \ingroup scatter
  \brief Estimate the scatter probability using a model-based approach

  This class computes the single Compton scatter estimate for PET data using an analytical 
  approximation of an integral. It takes as input an emission image and an attenuation image.
  This is effectively an implementation of the simulation part of the algorithms of 
  Watson and Ollinger.

  One non-standard feature is that you can specify a different attenuation image to find the
  scatter points and one to compute the integrals over the attenuation image. The idea is that
  maybe you want to compute the integrals on a finer grid than you sample the attenuation image.
  This is probably not very useful though.

  \todo Currently this can only be run by initialising it via parsing of a file. We need
  to add a lot of set/get members.
  
  \todo This class currently uses a simple Gaussian model for the energy resolution. This model
  and its parameters (\a reference_energy and \a energy_resolution) really should be moved the
  the Scanner class. Also the \a lower_energy_threshold and \a upper_energy_threshold should
  be read from the emission data, as opposed to setting them here.

  \todo detector coordinates are derived from ProjDataInfo, but areas and orientations are 
  deteremined by using a cylindrical scanner.

  \todo This class should be split into a generic class and one specific to PET single scatter.

  \par References
  This implementation is described in the following
  <ol>
  <li> C. Tsoumpas, P. Aguiar, K. S. Nikita, D. Ros, K. Thielemans, 
  <i>Evaluation of the Single Scatter Simulation Algorithm Implemented in the STIR Library</i>, 
  Proc. of IEEE Medical Imaging Conf. 2004, vol. 6 pp. 3361 - 3365. 
  </li>
  </ol>
  Please refer to the above if you use this implementation.

  See also these papers for extra evaluation

  <ol>
  <li>N. Dikaios , T. J. Spinks , K. Nikita , K. Thielemans, 
     <i>Evaluation of the Scatter Simulation Algorithm Implemented in STIR,</i> 
     proc. 5th ESBME, Patras, Greece.
  </li>
  <li>P. Aguiar, Ch. Tsoumpas, C. Crespo, J. Pavía, C. Falcón, A. Cot, K. Thielemans and D. Ros, 
     <i>Assessment of scattered photons in the quantification of the small animal PET studies,</i> 
     Eur J Nucl Med Mol I 33:S315-S315 Sep 2006, Proc. EANM 2006, Athens, Greece.
  </li>
  </ol>
*/
class ScatterEstimationByBin : public ParsingObject
{
 public:
  //! upsample coarse scatter estimate and fit it to tails of the emission data
 static void
   upsample_and_fit_scatter_estimate(ProjData& scaled_scatter_proj_data,
				     const  ProjData& emission_proj_data,
				     const ProjData& scatter_proj_data,
				     const ProjData& weights_proj_data,
				     const float min_scale_factor,
				     const float max_scale_factor,
				     const unsigned half_filter_width,
				     BSpline::BSplineType spline_type,
				     const bool remove_interleaving = true);


  //! Default constructor (calls set_defaults())
  ScatterEstimationByBin();
  //! Function that does all the actual work
  Succeeded process_data();

  // TODO write_log can't be const because parameter_info isn't const
  void
    write_log(const double simulation_time, 
	      const float total_scatter);

 protected:
  void set_defaults();
  void initialise_keymap();
  bool post_processing();

  //! threshold below which a voxel in the attenuation image will not be considered as a candidate scatter point
  float attenuation_threshold;

  //! boolean to see if we need to move the scatter point randomly within in its voxel
  /*! This was first recommended by Watson. It is recommended to leave this on, as otherwise
     discretisation errors are more obvious.

     Note that the random generator is seeded via date/time, so re-running the scatter 
     simulation will give a slightly different result if this boolean is on.
  */
  bool random;
  //! boolean to see if we need to cache the integrals
  /*! By default, we cache the integrals over the emission and attenuation image. If you run out
      of memory, you can switch this off, but performance will suffer dramatically.
  */
  bool use_cache;

  //! \name Parameters determining the energy detection efficiency of the scanner
  //@{
  //! reference energy used when specifying the energy resolution of the detectors (in units of keV)
  float reference_energy; 
  //! resolution of the detectors at \a reference_energy
  /*! specify as a fraction of the \a reference_energy. For example, a Discovery STE has an energy
  resolution of about .16 at 511 keV.
  */
  float energy_resolution;

  //! Lower energy threshold set during acquisition
  float lower_energy_threshold;
  //! Upper energy threshold set during acquisition
  float upper_energy_threshold;
  //@}

  std::string activity_image_filename;
  std::string density_image_filename;
  std::string density_image_for_scatter_points_filename;
  std::string template_proj_data_filename;
  std::string output_proj_data_filename;

  shared_ptr<DiscretisedDensity<3,float> > density_image_for_scatter_points_sptr;
  shared_ptr<DiscretisedDensity<3,float> > density_image_sptr;
  shared_ptr<DiscretisedDensity<3,float> > activity_image_sptr;
  shared_ptr<ProjData> output_proj_data_sptr; 


  /*************** functions that do the work **********/

  enum image_type{act_image_type, att_image_type};
  struct ScatterPoint
  { 
    CartesianCoordinate3D<float> coord;
    float mu_value;
  };
  
  std::vector< ScatterPoint> scatt_points_vector;
  float scatter_volume;

  //! fill in scatt_points_vector and scatter_volume       
  void 
    sample_scatter_points();

  /************************************************************************/

  //! \name detection related functions
  //@{
  //! energy-dependent detection efficiency (Gaussian model)
  float 
    detection_efficiency(const float energy) const;

	
  //! maximum angle to consider above which detection after Compton scatter is considered too small
  static
    float 
    max_cos_angle(const float low, const float approx, const float resolution_at_511keV);

  //! mimumum energy to consider above which detection after Compton scatter is considered too small
  static
    float 
    energy_lower_limit(const float low, const float approx, const float resolution_at_511keV);

  virtual 
    void
    find_detectors(unsigned& det_num_A, unsigned& det_num_B, const Bin& bin) const; 

  unsigned 
    find_in_detection_points_vector(const CartesianCoordinate3D<float>& coord) const;
  // private:
  const ProjDataInfoCylindricalNoArcCorr * proj_data_info_ptr;
  CartesianCoordinate3D<float>  shift_detector_coordinates_to_origin;

  //! average detection efficiency of unscattered counts
  double
    detection_efficiency_no_scatter(const unsigned det_num_A, 
				    const unsigned det_num_B) const;

  // next needs to be mutable because find_in_detection_points_vector is const
  mutable std::vector<CartesianCoordinate3D<float> > detection_points_vector;
 private:
  int total_detectors;

  //@}
 protected:
  //! computes scatter for one viewgram
  /*! \return total scatter estimated for this viewgram */
  virtual double
    process_data_for_view_segment_num(const ViewSegmentNumbers& vs_num);

  /************************************************************************/

  //! \name integrating functions
  //@{
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
  //@}
  
  /************************************************************************/
	
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
  /************************************************************************/

  //! \name Compton scatter cross sections
  //@{
  static
    inline float 
    dif_Compton_cross_section(const float cos_theta, float energy);
	
	
  static
    inline float 
    total_Compton_cross_section(float energy);
	
  static
    inline float
    photon_energy_after_Compton_scatter(const float cos_theta, const float energy);

  static
    inline float
    photon_energy_after_Compton_scatter_511keV(const float cos_theta);

  static
    inline float 
    total_Compton_cross_section_relative_to_511keV(const float energy);
  //@}
  /************************************************************************/
 protected:
	  
  float  
    compute_emis_to_det_points_solid_angle_factor(const CartesianCoordinate3D<float>& emis_point,
						  const CartesianCoordinate3D<float>& detector_coord) ;

 private:
  Array<2,float> cached_activity_integral_scattpoint_det;
  Array<2,float> cached_attenuation_integral_scattpoint_det;
  void initialise_cache_for_scattpoint_det();

};


END_NAMESPACE_STIR

#include "stir/scatter/ScatterEstimationByBin.inl"

#endif
