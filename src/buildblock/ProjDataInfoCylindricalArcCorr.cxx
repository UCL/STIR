//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementation of non-inline functions of class 
  ProjDataInfoCylindricalArcCorr

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/

#include "ProjDataInfoCylindricalArcCorr.h"
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#ifndef TOMO_NO_NAMESPACES
using std::endl;
using std::ends;
#endif


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



void
ProjDataInfoCylindricalArcCorr::set_tangential_sampling(const float new_tangential_sampling)
{bin_size = new_tangential_sampling;}



ProjDataInfo*
ProjDataInfoCylindricalArcCorr::clone() const
{
  return static_cast<ProjDataInfo*>(new ProjDataInfoCylindricalArcCorr(*this));
}

string
ProjDataInfoCylindricalArcCorr::parameter_info()  const
{

#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[50000];
  ostrstream s(str, 50000);
#else
  std::ostringstream s;
#endif  
  s << "ProjDataInfoCylindricalArcCorr := \n";
  s << ProjDataInfoCylindrical::parameter_info();
  s << "tangential sampling := " << get_tangential_sampling() << endl;
  s << "End :=\n";
  return s.str();
}

END_NAMESPACE_TOMO

