//
/*
 Copyright (C) 2009 - 2013, King's College London
 This file is part of STIR.
 
 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.3 of the License, or
 (at your option) any later version.
 
 This file is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
 See STIR/LICENSE.txt for details
 */  
/*!
 \file
 \ingroup spatial_transformation
 \brief Declaration of class stir::GatedSpatialTransformation
 \author Charalampos Tsoumpas
 
*/

#ifndef __stir_spatial_transformation_GatedSpatialTransformation_H__
#define __stir_spatial_transformation_GatedSpatialTransformation_H__

#include "stir/GatedDiscretisedDensity.h"
#include "stir/DiscretisedDensity.h"
#include "stir/spatial_transformation/SpatialTransformation.h"
#include "stir/numerics/BSplinesRegularGrid.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/Succeeded.h"
#include <fstream>
#include <iostream>

START_NAMESPACE_STIR

//! Class for spatial transformations for gated images
/*!
 \ingroup spatial_transformation
*/
class GatedSpatialTransformation: public RegisteredParsingObject<GatedSpatialTransformation,SpatialTransformation>
{ 
 public:
  static const char * const registered_name; 

  GatedSpatialTransformation(); //!< default constructor
  ~GatedSpatialTransformation(); //!< default destructor
  //!  Construct an empty GatedSpatialTransformation based on a shared_ptr<DiscretisedDensity<3,float> >
  GatedSpatialTransformation(const TimeGateDefinitions& time_gate_definitions,
                const shared_ptr<DiscretisedDensity<3,float> >& density_sptr);

  void read_from_files(const std::string input_string);
  void write_to_files(const std::string output_string); 

  //! \name Functions to get parameters @{
  GatedDiscretisedDensity get_spatial_transformation_z() const;
  GatedDiscretisedDensity get_spatial_transformation_y() const;
  GatedDiscretisedDensity get_spatial_transformation_x() const;
  const TimeGateDefinitions & get_time_gate_definitions() const;
  //!@}
  //! \name Functions to set parameters @{
  void set_spatial_transformations(const GatedDiscretisedDensity & motion_z, 
                          const GatedDiscretisedDensity & motion_y, 
                          const GatedDiscretisedDensity & motion_x);
  void set_gate_defs(const TimeGateDefinitions & gate_defs); 
  //!@}

  //! Warping functions from to gated images. @{
  void 
    warp_image(GatedDiscretisedDensity & new_gated_image,
               const GatedDiscretisedDensity & gated_image) const ;	
  void 
    warp_image(DiscretisedDensity<3, float> & new_reference_image,
               const GatedDiscretisedDensity & gated_image) const ;
  void
    warp_image(GatedDiscretisedDensity & gated_image,
               const DiscretisedDensity<3, float> & reference_image) const;
  void 
    accumulate_warp_image(DiscretisedDensity<3, float> & new_reference_image,
                          const GatedDiscretisedDensity & gated_image) const ;
  void set_defaults();
  Succeeded set_up(); 
  //@}
 private:
  typedef RegisteredParsingObject<GatedSpatialTransformation,SpatialTransformation> base_type;
  void initialise_keymap();
  bool post_processing();	
  std::string _transformation_filename_prefix;
  GatedDiscretisedDensity _spatial_transformation_z;
  GatedDiscretisedDensity _spatial_transformation_y;
  GatedDiscretisedDensity _spatial_transformation_x;
  std::string _spline_level_number;
  bool _spatial_transformations_are_stored;
  BSpline::BSplineType _spline_type;
  std::string _time_gate_definition_filename;
  TimeGateDefinitions _gate_defs;
};

END_NAMESPACE_STIR
#endif
