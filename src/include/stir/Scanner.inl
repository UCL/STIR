//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2016, UCL
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
  \ingroup buildblock
  \brief implementation of inline functions of class Scanner

  \author Nikos Efthimiou
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Long Zhang (set*() functions)
  \author PARAPET project


*/

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
Scanner::get_default_intrinsic_tilt() const
{
  return intrinsic_tilt;}

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
  return num_rings/num_axial_crystals_per_block;
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
    return num_rings / num_axial_crystals_per_singles_unit;
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

int Scanner::get_num_max_of_timing_bins() const
{
    return max_num_of_timing_bins;
}

float Scanner::get_size_of_timing_bin() const
{
    return size_timing_bin;
}

float Scanner::get_timing_resolution() const
{
    return timing_resolution;
}

bool Scanner::is_tof_ready() const
{
    return (max_num_of_timing_bins > 0
            && timing_resolution > 0.0f
            && timing_resolution > 0.0f);
}

//************************ set ******************************8

void Scanner::set_type(const Type & new_type)
{
   type = new_type;
}

void Scanner::set_num_rings(const int & new_num)
{
  num_rings = new_num;
}
  
void Scanner::set_num_detectors_per_ring(const int & new_num) 
{
  num_detectors_per_ring = new_num;  
}

void Scanner::set_max_num_non_arccorrected_bins(const int&  new_num)
{
  max_num_non_arccorrected_bins = new_num;
}

void Scanner::set_default_num_arccorrected_bins(const int&  new_num)
{
  default_num_arccorrected_bins = new_num;
}


void Scanner::set_inner_ring_radius(const float & new_radius)
{
  inner_ring_radius = new_radius;
}

void Scanner::set_average_depth_of_interaction(const float & new_depth_of_interaction)
{
  average_depth_of_interaction = new_depth_of_interaction;
}



void Scanner::set_ring_spacing(const float&  new_spacing)
{
  ring_spacing = new_spacing;
}

void Scanner::set_default_bin_size(const float  & new_size)
{
  bin_size = new_size;
}

void Scanner::set_default_intrinsic_tilt(const float &  new_tilt)
{
  intrinsic_tilt = new_tilt;
}

void Scanner::set_num_transaxial_blocks_per_bucket(const int&  new_num)
{
  num_transaxial_blocks_per_bucket = new_num;
}

void Scanner::set_num_axial_blocks_per_bucket(const int&  new_num)
{
  num_axial_blocks_per_bucket = new_num;
}

void Scanner::set_num_detector_layers(const int& new_num)
{
  num_detector_layers = new_num;
}


void Scanner::set_num_axial_crystals_per_block(const int&  new_num)
{
  num_axial_crystals_per_block = new_num;
}

void Scanner::set_num_transaxial_crystals_per_block(const int& new_num)
{
  num_transaxial_crystals_per_block = new_num;
}



void Scanner::set_num_axial_crystals_per_singles_unit(const int& new_num)
{
  num_axial_crystals_per_singles_unit = new_num;
}

void Scanner::set_num_transaxial_crystals_per_singles_unit(const int& new_num)
{
  num_transaxial_crystals_per_singles_unit = new_num;
}

void
Scanner::set_energy_resolution(const float new_num)
{
    energy_resolution = new_num;
}

void
Scanner::set_reference_energy(const float new_num)
{
    reference_energy = new_num;
}

void Scanner::set_num_max_of_timing_bins(const int new_num)
{
    max_num_of_timing_bins = new_num;
}

void Scanner::set_size_of_timing_bin(const float new_num)
{
    size_timing_bin = new_num;
}

void Scanner::set_timing_resolution(const float new_num_in_ps)
{
    timing_resolution = new_num_in_ps;
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



END_NAMESPACE_STIR

