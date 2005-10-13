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
};

END_NAMESPACE_STIR

#endif
