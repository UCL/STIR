//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementation of inline functions of class ProjDataInfoCylindrical

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

// for sqrt
#include <math.h>
#include "Bin.h"

START_NAMESPACE_TOMO


float
ProjDataInfoCylindrical::get_phi(const Bin& bin)const
{ return bin.view_num()*azimuthal_angle_sampling;}

float
ProjDataInfoCylindrical::get_t(const Bin& bin) const
{
  return 
    get_m(bin)/
    sqrt(1+square(get_tantheta(bin)));
}

/*!
  The 0 of the z-axis is chosen in the middle of the scanner.

  \warning Current implementation assumes that the axial positions are always 'centred',
  i.e. get_m(Bin(..., min_axial_pos_num,...)) == - get_m(Bin(..., max_axial_pos_num,...))
*/  
float
ProjDataInfoCylindrical::get_m(const Bin& bin) const
{ return 
    bin.axial_pos_num()*get_axial_sampling(bin.segment_num())
    - m_offset[bin.segment_num()];
}


float
ProjDataInfoCylindrical::get_tantheta(const Bin& bin) const
{
  return
    get_average_ring_difference(bin.segment_num())*
    ring_spacing/ 
    (2*sqrt(square(ring_radius)-square(get_s(bin))));
  
}



#ifdef SET
void
ProjDataInfoCylindrical::set_azimuthal_angle_sampling(const float angle_v)
{azimuthal_angle_sampling =  angle_v;}
#endif
//void
//ProjDataInfoCylindrical::set_axial_sampling(const float samp_v, int segment_num)
//{axial_sampling = samp_v;}

float
ProjDataInfoCylindrical::get_azimuthal_angle_sampling() const
{return azimuthal_angle_sampling;}

/*! 
   The implementation of this function currently assumes that the axial
   sampling is equal to the ring spacing for non-spanned data 
   (i.e. no axial compression), while it is half the 
   ring spacing for spanned data.
 */
float
ProjDataInfoCylindrical::get_axial_sampling(int segment_num) const
{
  if (max_ring_diff[segment_num] != min_ring_diff[segment_num])
    return ring_spacing/2;
  else
    return ring_spacing;
}

float 
ProjDataInfoCylindrical::get_average_ring_difference(int segment_num) const
{
  return (min_ring_diff[segment_num] + max_ring_diff[segment_num])/2;
}


int 
ProjDataInfoCylindrical::get_min_ring_difference(int segment_num) const
{ return min_ring_diff[segment_num]; }

int 
ProjDataInfoCylindrical::get_max_ring_difference(int segment_num) const
{ return max_ring_diff[segment_num]; }

float
ProjDataInfoCylindrical::get_ring_radius() const
{return ring_radius;}

float
ProjDataInfoCylindrical::get_ring_spacing() const
{ return ring_spacing;}


void
ProjDataInfoCylindrical::set_min_ring_difference( int min_ring_diff_v, int segment_num)
{
 min_ring_diff[segment_num] = min_ring_diff_v;
}

void
ProjDataInfoCylindrical::set_max_ring_difference( int max_ring_diff_v, int segment_num)
{
  max_ring_diff[segment_num] = max_ring_diff_v;
}

void
ProjDataInfoCylindrical::set_ring_spacing(float ring_spacing_v)
{
  ring_spacing = ring_spacing_v;
}




END_NAMESPACE_TOMO


  
