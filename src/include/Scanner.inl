//
// $Id$: $Date$
//
/*!
  \file
  \ingroup buildblock
  \brief implementation of inline functions of class Scanner

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
START_NAMESPACE_TOMO

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
END_NAMESPACE_TOMO
