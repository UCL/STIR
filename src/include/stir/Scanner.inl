//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2016, 2021 University College London
    Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock
  \brief implementation of inline functions of class Scanner

  \author Nikos Efthimiou
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Long Zhang (set*() functions)
  \author PARAPET project
  \author Parisa Khateri


*/
#include "stir/error.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

bool 
Scanner::operator !=(const Scanner& scanner) const
{
  return !(*this == scanner);
}

Scanner::Type
Scanner::get_type() const
{return type;}

int
Scanner::get_num_rings() const
{  return num_rings;}
int
Scanner::get_num_detectors_per_ring() const
{
  return num_detectors_per_ring;}
int
Scanner::get_max_num_non_arccorrected_bins() const
{ return max_num_non_arccorrected_bins;}

int 
Scanner::get_max_num_views() const
{ return get_num_detectors_per_ring()/2; }

int
Scanner::get_default_num_arccorrected_bins() const
{
  return default_num_arccorrected_bins;   
}

float
Scanner::get_inner_ring_radius() const
{ 
  return inner_ring_radius;
}

float
Scanner::get_outer_FOV_radius() const
{
  return outer_FOV_radius;
}

float
Scanner::get_average_depth_of_interaction() const
{
  return average_depth_of_interaction;
}


float
Scanner::get_effective_ring_radius() const
{
  return inner_ring_radius + average_depth_of_interaction;
}


float
Scanner::get_ring_spacing() const
{
  return ring_spacing;
}


float
Scanner::get_default_bin_size() const
{ return bin_size;}

float
Scanner::get_intrinsic_azimuthal_tilt() const
{
#ifdef STIR_LEGACY_IGNORE_VIEW_OFFSET
  return 0.F;
#else
  return intrinsic_tilt;
#endif
}

int 
Scanner::get_num_transaxial_blocks_per_bucket() const
{ 
  return num_transaxial_blocks_per_bucket;
}

int
Scanner::get_num_axial_blocks_per_bucket() const
{
  return num_axial_blocks_per_bucket;
}

int
Scanner::get_num_axial_crystals_per_block() const
{
  return num_axial_crystals_per_block;
}

int
Scanner::get_num_transaxial_crystals_per_block()const
{
  return num_transaxial_crystals_per_block;
}


int
Scanner::get_num_axial_crystals_per_bucket() const
{
  return
    get_num_axial_blocks_per_bucket() *
    get_num_axial_crystals_per_block();
}


int
Scanner::get_num_transaxial_crystals_per_bucket() const
{
  return
    get_num_transaxial_blocks_per_bucket() *
    get_num_transaxial_crystals_per_block();
}

int
Scanner::get_num_detector_layers() const
{
  return num_detector_layers;
}

int
Scanner::get_num_axial_blocks() const
{
  // when using virtual crystals between blocks, there won't be one at the end, so we
  // need to take this into account.
  return (num_rings+get_num_virtual_axial_crystals_per_block())/num_axial_crystals_per_block;
}

int
Scanner::get_num_transaxial_blocks() const
{
  return num_detectors_per_ring/num_transaxial_crystals_per_block;
}

int
Scanner::get_num_axial_buckets() const
{
  return get_num_axial_blocks()/num_axial_blocks_per_bucket;
}

int
Scanner::get_num_transaxial_buckets() const
{
  return get_num_transaxial_blocks()/num_transaxial_blocks_per_bucket;
}



int
Scanner::get_num_axial_crystals_per_singles_unit() const
{
  return num_axial_crystals_per_singles_unit;
}

int
Scanner::get_num_transaxial_crystals_per_singles_unit() const
{
  return num_transaxial_crystals_per_singles_unit;
}


int
Scanner::get_num_axial_singles_units() const
{
  if ( num_axial_crystals_per_singles_unit == 0 ) {
    return 0;
  } else {
    return (num_rings+get_num_virtual_axial_crystals_per_block()) / num_axial_crystals_per_singles_unit;
  }
}


int
Scanner::get_num_transaxial_singles_units() const
{
  if ( num_transaxial_crystals_per_singles_unit == 0 ) {
    return 0;
  } else {
    return num_detectors_per_ring / num_transaxial_crystals_per_singles_unit;
  }
}


int 
Scanner::get_num_singles_units  () const
{
  // TODO Accomodate more complex (multi-layer) geometries.
  return get_num_axial_singles_units() * get_num_transaxial_singles_units();
}

float
Scanner::get_energy_resolution() const
{
    return energy_resolution;
}

float
Scanner::get_reference_energy() const
{
    return reference_energy;
}

std::string
Scanner::get_scanner_geometry() const
{
  return scanner_geometry;
}

float
Scanner::get_axial_crystal_spacing() const
{
       return axial_crystal_spacing;
}

float
Scanner::get_transaxial_crystal_spacing() const
{
       return transaxial_crystal_spacing;
}

float
Scanner::get_transaxial_block_spacing() const
{
       return transaxial_block_spacing;
}

float
Scanner::get_axial_block_spacing() const
{
       return axial_block_spacing;
}

std::string
Scanner::get_crystal_map_file_name() const
{
  return crystal_map_file_name;
}

//************************ set ******************************8

void Scanner::set_type(const Type & new_type)
{
   type = new_type;
   _already_setup = false;
}

void Scanner::set_num_rings(const int & new_num)
{
  num_rings = new_num;
   _already_setup = false;
}
  
void Scanner::set_num_detectors_per_ring(const int & new_num) 
{
  num_detectors_per_ring = new_num;  
   _already_setup = false;
}

void Scanner::set_max_num_non_arccorrected_bins(const int&  new_num)
{
  max_num_non_arccorrected_bins = new_num;
   _already_setup = false;
}

void Scanner::set_default_num_arccorrected_bins(const int&  new_num)
{
  default_num_arccorrected_bins = new_num;
   _already_setup = false;
}


void Scanner::set_inner_ring_radius(const float & new_radius)
{
  inner_ring_radius = new_radius;
   _already_setup = false;
}

void Scanner::set_average_depth_of_interaction(const float & new_depth_of_interaction)
{
  average_depth_of_interaction = new_depth_of_interaction;
   _already_setup = false;
}

bool Scanner::has_energy_information() const
{
    return (energy_resolution <= 0.0 ||
            reference_energy <= 0.0) ? false : true;
}

void Scanner::set_ring_spacing(const float&  new_spacing)
{
  ring_spacing = new_spacing;
}

void Scanner::set_default_bin_size(const float  & new_size)
{
  bin_size = new_size;
   _already_setup = false;
}

void Scanner::set_intrinsic_azimuthal_tilt(const float new_tilt)
{
  intrinsic_tilt = new_tilt;
   _already_setup = false;
}

void Scanner::set_num_transaxial_blocks_per_bucket(const int&  new_num)
{
  num_transaxial_blocks_per_bucket = new_num;
   _already_setup = false;
}

void Scanner::set_num_axial_blocks_per_bucket(const int&  new_num)
{
  num_axial_blocks_per_bucket = new_num;
   _already_setup = false;
}

void Scanner::set_num_detector_layers(const int& new_num)
{
  num_detector_layers = new_num;
   _already_setup = false;
}


void Scanner::set_num_axial_crystals_per_block(const int&  new_num)
{
  num_axial_crystals_per_block = new_num;
   _already_setup = false;
}

void Scanner::set_num_transaxial_crystals_per_block(const int& new_num)
{
  num_transaxial_crystals_per_block = new_num;
   _already_setup = false;
}



void Scanner::set_num_axial_crystals_per_singles_unit(const int& new_num)
{
  num_axial_crystals_per_singles_unit = new_num;
   _already_setup = false;
}

void Scanner::set_num_transaxial_crystals_per_singles_unit(const int& new_num)
{
  num_transaxial_crystals_per_singles_unit = new_num;
   _already_setup = false;
}

void
Scanner::set_energy_resolution(const float new_num)
{
    energy_resolution = new_num;
   _already_setup = false;
}

void
Scanner::set_reference_energy(const float new_num)
{
    reference_energy = new_num;
   _already_setup = false;
}

void Scanner::set_axial_crystal_spacing(const float&  new_spacing)
{
  axial_crystal_spacing = new_spacing;
   _already_setup = false;
}

void Scanner::set_transaxial_crystal_spacing(const float&  new_spacing)
{
  transaxial_crystal_spacing = new_spacing;
   _already_setup = false;
}

void Scanner::set_transaxial_block_spacing(const float&  new_spacing)
{
  transaxial_block_spacing = new_spacing;
   _already_setup = false;
}

void Scanner::set_axial_block_spacing(const float&  new_spacing)
{
  axial_block_spacing = new_spacing;
   _already_setup = false;
}

void Scanner::set_crystal_map_file_name(const std::string& new_crystal_map_file_name)
{
  crystal_map_file_name = new_crystal_map_file_name;
   _already_setup = false;
}

/********    Calculate singles bin index from detection position    *********/


int
Scanner::get_singles_bin_index(int axial_index, int transaxial_index) const {
  // TODO: Accomodate more complex geometry.
  return(transaxial_index + (axial_index * get_num_transaxial_singles_units()));
}



int
Scanner::get_singles_bin_index(const DetectionPosition<>& det_pos) const {

  // TODO: Accomodate more complex geometry.
        
  int axial_index = det_pos.axial_coord() / get_num_axial_crystals_per_singles_unit();

  int transaxial_index = det_pos.tangential_coord() / 
                                get_num_transaxial_crystals_per_singles_unit();
  
  //return(transaxial_index + (axial_index * get_num_transaxial_singles_units()));
  return(get_singles_bin_index(axial_index, transaxial_index));

}



// Get the axial singles bin coordinate from a singles bin.
int 
Scanner::get_axial_singles_unit(int singles_bin_index) const {
  // TODO: Accomodate more complex geometry.
  return(singles_bin_index / get_num_transaxial_singles_units());
}



// Get the transaxial singles bin coordinate from a singles bin.
int 
Scanner::get_transaxial_singles_unit(int singles_bin_index) const {
  // TODO: Accomodate more complex geometry.
  return(singles_bin_index % get_num_transaxial_singles_units());
}

// For retrieving the coordinates / detector, ring id from the scanner
stir::DetectionPosition<>
Scanner::get_det_pos_for_index(const stir::DetectionPosition<> & det_pos) const
{
    if (!detector_map_sptr)
        stir::error("Scanner: detector_map not defined. Did you run set_up()?");

    return detector_map_sptr->get_det_pos_for_index(det_pos);
}

stir::CartesianCoordinate3D<float>
Scanner::get_coordinate_for_det_pos(const stir::DetectionPosition<>& det_pos) const
{
  if (!_already_setup)
    stir::error("Scanner: you forgot to call set_up().");
  if (!detector_map_sptr)
    stir::error("Scanner: detector_map not defined. Did you run set_up()?");

  return detector_map_sptr->get_coordinate_for_det_pos(det_pos);
}

stir::CartesianCoordinate3D<float>
Scanner::get_coordinate_for_index(const stir::DetectionPosition<>& index) const
{
  if (!_already_setup)
    stir::error("Scanner: you forgot to call set_up().");
  if (!detector_map_sptr)
    stir::error("Scanner: detector_map not defined. Did you run set_up()?");

  return detector_map_sptr->get_coordinate_for_index(index);
}

Succeeded
Scanner::find_detection_position_given_cartesian_coordinate(DetectionPosition<>& det_pos,
                                                            const CartesianCoordinate3D<float>& cart_coord) const
{
  if (!_already_setup)
    stir::error("Scanner: you forgot to call set_up().");
  if (!detector_map_sptr)
    stir::error("Scanner: detector_map not defined. Did you run set_up()?");

  return detector_map_sptr->find_detection_position_given_cartesian_coordinate(det_pos, cart_coord);
}

END_NAMESPACE_STIR

