//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief implementation of inline functions of class Scanner

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
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
Scanner::get_trans_blocks_per_bucket() const
{ 
  return trans_blocks_per_bucket;
}

int
Scanner::get_axial_blocks_per_bucket() const
{
  return axial_blocks_per_bucket;
}

int
Scanner::get_axial_crystals_per_block() const
{
  return axial_crystals_per_block;
}

int
Scanner::get_angular_crystals_per_block()const
{
  return angular_crystals_per_block;
}

END_NAMESPACE_STIR

