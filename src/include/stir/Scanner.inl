//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief implementation of inline functions of class Scanner

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Long Zhang (set*() functions)
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
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
Scanner::get_ring_radius() const

{ return ring_radius;}

float
Scanner::get_ring_spacing() const
{
  return ring_spacing;}

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


void Scanner::set_ring_radius(const float & new_radius)
{
  ring_radius = new_radius;
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

void Scanner::set_num_axial_crystals_per_block(const int&  new_num)
{
  num_axial_crystals_per_block = new_num;
}

void Scanner::set_num_transaxial_crystals_per_block(const int& new_num)
{
  num_transaxial_crystals_per_block = new_num;
}

END_NAMESPACE_STIR

