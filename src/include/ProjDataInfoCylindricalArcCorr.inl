//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementation of inline functions of class 
  ProjDataInfoCylindricalArcCorr

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#include "Bin.h"

START_NAMESPACE_TOMO
ProjDataInfoCylindricalArcCorr:: ProjDataInfoCylindricalArcCorr()

{}

ProjDataInfoCylindricalArcCorr:: ProjDataInfoCylindricalArcCorr(const shared_ptr<Scanner> scanner_ptr,float bin_size_v,								
								const  VectorWithOffset<int>& num_axial_pos_per_segment,
								const  VectorWithOffset<int>& min_ring_diff_v, 
								const  VectorWithOffset<int>& max_ring_diff_v,
								const int num_views,const int num_tangential_poss)
								:ProjDataInfoCylindrical(scanner_ptr,
								num_axial_pos_per_segment,
								min_ring_diff_v, max_ring_diff_v,
								num_views, num_tangential_poss),
								bin_size(bin_size_v)
								
								
								
{}


float
ProjDataInfoCylindricalArcCorr::get_s(const Bin& bin) const
{return bin.tangential_pos_num()*bin_size;}


float
ProjDataInfoCylindricalArcCorr::get_tangential_sampling() const
{return bin_size;}

#if 1
void
ProjDataInfoCylindricalArcCorr::set_tangential_sampling(const float new_tangential_sampling)
{bin_size = new_tangential_sampling;}
#endif



ProjDataInfo*
ProjDataInfoCylindricalArcCorr::clone() const
{
  return static_cast<ProjDataInfo*>(new ProjDataInfoCylindricalArcCorr(*this));
}


END_NAMESPACE_TOMO

