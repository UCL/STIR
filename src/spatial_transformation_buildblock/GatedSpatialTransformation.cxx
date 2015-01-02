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
 \brief Implementations of inline functions of class stir::GatedSpatialTransformation
 \author Charalampos Tsoumpas
 \sa GatedSpatialTransformation.h and SpatialTransformation.h

*/

#include "stir/spatial_transformation/GatedSpatialTransformation.h"
#include "stir/spatial_transformation/warp_image.h"
#include "stir/info.h"
#include <boost/format.hpp>

START_NAMESPACE_STIR

void
GatedSpatialTransformation::
set_defaults()
{
  base_type::set_defaults();
  this->_transformation_filename_prefix="";
  this->_spline_type=static_cast<BSpline::BSplineType> (1);;
}

const char * const 
GatedSpatialTransformation::registered_name = "Gated Spatial Transformation";

//! default constructor
GatedSpatialTransformation::GatedSpatialTransformation()
{ 
  this->set_defaults();
}

GatedSpatialTransformation::~GatedSpatialTransformation()   //!< default destructor
{ }

Succeeded 
GatedSpatialTransformation::set_up()
{
  if (this->_spatial_transformations_are_stored==true)
    return Succeeded::yes;
  else
    return Succeeded::no;
}

const TimeGateDefinitions &
GatedSpatialTransformation::get_time_gate_definitions() const
{
  return this->_spatial_transformation_x.get_time_gate_definitions();
}

void
GatedSpatialTransformation::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Gated Spatial Transformation Parameters");
  this->parser.add_key("Gated Spatial Transformation Filenames Prefix", &this->_transformation_filename_prefix);
  this->parser.add_stop_key("end Gated Spatial Transformation Parameters");
}

bool
GatedSpatialTransformation::
post_processing()
{
  if (base_type::post_processing() == true)
    return true;
  if(this->_transformation_filename_prefix=="0")
    {
      warning("You need to specify a prefix for three files with transformation information.");
      return true;
    }
  else
    {
      GatedSpatialTransformation::read_from_files(this->_transformation_filename_prefix);
      this->_spatial_transformations_are_stored=true;
    }
  // Always linear interpolation for the moment
  this->_spline_type=static_cast<BSpline::BSplineType> (1);
  return false;
}

//! Implementation to read the transformation vectors will be moved to the IO directory because it should be general. For example it can be in ECAT7 image formant
void
GatedSpatialTransformation::read_from_files(const std::string input_string) 
{ 
  const std::string gate_defs_input_string=input_string + ".gdef";

  if (gate_defs_input_string.size()!=0)
    this->_gate_defs=TimeGateDefinitions(gate_defs_input_string);
  else {
    error("No Time Gates Definitions available!!!\n ");
  }

  const shared_ptr<GatedDiscretisedDensity> spatial_transformation_z_sptr (GatedDiscretisedDensity::read_from_files(input_string,"d1"));
  const GatedDiscretisedDensity & spatial_transformation_z(*spatial_transformation_z_sptr);
	
  const shared_ptr<GatedDiscretisedDensity> spatial_transformation_y_sptr(GatedDiscretisedDensity::read_from_files(input_string,"d2"));
  const GatedDiscretisedDensity & spatial_transformation_y(*spatial_transformation_y_sptr);
	
  const shared_ptr<GatedDiscretisedDensity> spatial_transformation_x_sptr (GatedDiscretisedDensity::read_from_files(input_string,"d3"));
  const GatedDiscretisedDensity & spatial_transformation_x(*spatial_transformation_x_sptr);
	
  const TimeGateDefinitions gate_defs(gate_defs_input_string);//This is not necessary as the defs are necessary for all the files and it should be one file... Think how to do this.
	
  this->_spatial_transformation_z= spatial_transformation_z; this->_spatial_transformation_y= spatial_transformation_y; this->_spatial_transformation_x= spatial_transformation_x; 
  this->_spatial_transformations_are_stored=true;
}     

//! Implementation to write the transformation vectors
void
GatedSpatialTransformation::write_to_files(const std::string output_string) 
{
  (this->_spatial_transformation_z).write_to_files(output_string,"d1");
  (this->_spatial_transformation_y).write_to_files(output_string,"d2");
  (this->_spatial_transformation_x).write_to_files(output_string,"d3");
  //	return Succeeded::yes; // add a no case if you cannot write
}  

void 
GatedSpatialTransformation::warp_image(GatedDiscretisedDensity & new_gated_image,
                          const GatedDiscretisedDensity & gated_image) const 
{
  std::string explanation;
  if (!(gated_image.get_densities()[0])->has_same_characteristics(*(gated_image.get_densities()[0]), explanation)){
        error(boost::format("GatedSpatialTransformation::warp_image needs the same sizes for input and output images: %1%") % explanation);
  }
  new_gated_image.set_time_gate_definitions(this->_gate_defs);
  assert(gated_image.get_time_gate_definitions().get_num_gates()==this->_spatial_transformation_x.get_time_gate_definitions().get_num_gates());
  new_gated_image.fill_with_zero();
  if (this->_spatial_transformations_are_stored)
    for(unsigned int gate_num=1 ; gate_num<=gated_image.get_time_gate_definitions().get_num_gates() ; ++gate_num)
      new_gated_image[gate_num]=stir::warp_image((gated_image.get_densities())[gate_num-1],
                                                 (this->_spatial_transformation_x.get_densities())[gate_num-1],
                                                 (this->_spatial_transformation_y.get_densities())[gate_num-1],
                                                 (this->_spatial_transformation_z.get_densities())[gate_num-1], 
                                                 BSpline::linear, false);
  else
    error("The transformation fields haven't been set properly yet.\n");
}

void
GatedSpatialTransformation::warp_image(DiscretisedDensity<3, float> & new_reference_image,
                          const GatedDiscretisedDensity & gated_image) const 
{
  new_reference_image.fill(0.F);
  this->accumulate_warp_image(new_reference_image, gated_image);
}

void
GatedSpatialTransformation::accumulate_warp_image(DiscretisedDensity<3, float> & new_reference_image,
                                     const GatedDiscretisedDensity & gated_image) const 
{
  GatedDiscretisedDensity new_gated_image(gated_image);
  new_gated_image.fill_with_zero();
  this->warp_image(new_gated_image,gated_image);
  //!todo This is not implemented as sum (or should it be the average?)
  for(unsigned int gate_num = 1;gate_num<=gated_image.get_time_gate_definitions().get_num_gates() ; ++gate_num)
    new_reference_image += new_gated_image[gate_num];
  //	new_reference_image /= gated_image.get_time_gate_definitions().get_num_gates();
}

void 
GatedSpatialTransformation::warp_image(GatedDiscretisedDensity & gated_image,
                          const DiscretisedDensity<3, float> & reference_image) const 
{
  if ((gated_image.get_densities())[0]->size_all()!=reference_image.size_all()){
    error("GatedSpatialTransformation::warp_image needs the same sizes for input and output images.\n");
  }
  if ((gated_image.get_densities())[0]->size_all()!=(this->_spatial_transformation_y.get_densities())[0]->size_all()){
    info(boost::format("Number of voxels in one gated image: %1%") % (gated_image.get_densities())[0]->size_all());
    info(boost::format("Number of voxels in one motion vector gated image: %1%") % (this->_spatial_transformation_y.get_densities())[0]->size_all());
    error("GatedSpatialTransformation::warp_image needs the same sizes for motion vectors and input/output images.\n");
  }
  const shared_ptr<DiscretisedDensity<3,float> > reference_image_sptr( reference_image.clone());
  gated_image.resize_densities(this->_gate_defs);
	
  if (this->_spatial_transformations_are_stored)
    for(unsigned int gate_num = 1 ; gate_num<=gated_image.get_time_gate_definitions().get_num_gates() ; ++gate_num)
      {
        const VoxelsOnCartesianGrid<float> density = stir::warp_image(reference_image_sptr,
                                                                      (this->_spatial_transformation_x.get_densities())[gate_num-1],
                                                                      (this->_spatial_transformation_y.get_densities())[gate_num-1],
                                                                      (this->_spatial_transformation_z.get_densities())[gate_num-1], 
                                                                      BSpline::linear, false);
        const shared_ptr<DiscretisedDensity<3,float> >  density_sptr(density.clone());
        gated_image.set_density_sptr(density_sptr,gate_num);
      }
  else
    error("The transformation fields haven't been set properly yet.");	
}

void
GatedSpatialTransformation::
set_spatial_transformations(const GatedDiscretisedDensity & transformation_z, 
                         const GatedDiscretisedDensity & transformation_y, 
                         const GatedDiscretisedDensity & transformation_x)
{
  this->_spatial_transformation_z=transformation_z;
  this->_spatial_transformation_y=transformation_y;
  this->_spatial_transformation_x=transformation_x;
  this->_spatial_transformations_are_stored=true;
}

void 
GatedSpatialTransformation::set_gate_defs(const TimeGateDefinitions & gate_defs)
{ this->_gate_defs=gate_defs; }
 
GatedDiscretisedDensity GatedSpatialTransformation::get_spatial_transformation_z() const
{ return this->_spatial_transformation_z; }
GatedDiscretisedDensity GatedSpatialTransformation::get_spatial_transformation_y() const
{ return this->_spatial_transformation_y; }
GatedDiscretisedDensity GatedSpatialTransformation::get_spatial_transformation_x() const
{ return this->_spatial_transformation_x; }


END_NAMESPACE_STIR
