//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementation of non-inline functions of class 
  ProjDataInfoCylindricalNoArcCorr

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::ends;
#endif

START_NAMESPACE_STIR
ProjDataInfoCylindricalNoArcCorr:: 
ProjDataInfoCylindricalNoArcCorr()

{}

ProjDataInfoCylindricalNoArcCorr:: 
ProjDataInfoCylindricalNoArcCorr(const shared_ptr<Scanner> scanner_ptr,
                                 const float ring_radius_v, const float angular_increment_v,
				 const  VectorWithOffset<int>& num_axial_pos_per_segment,
                                 const  VectorWithOffset<int>& min_ring_diff_v, 
                                 const  VectorWithOffset<int>& max_ring_diff_v,
                                 const int num_views,const int num_tangential_poss)
: ProjDataInfoCylindrical(scanner_ptr,
                          num_axial_pos_per_segment,
                          min_ring_diff_v, max_ring_diff_v,
                          num_views, num_tangential_poss),
  ring_radius(ring_radius_v),
  angular_increment(angular_increment_v)
{}

ProjDataInfoCylindricalNoArcCorr:: 
ProjDataInfoCylindricalNoArcCorr(const shared_ptr<Scanner> scanner_ptr,
                                 const  VectorWithOffset<int>& num_axial_pos_per_segment,
                                 const  VectorWithOffset<int>& min_ring_diff_v, 
                                 const  VectorWithOffset<int>& max_ring_diff_v,
                                 const int num_views,const int num_tangential_poss)
: ProjDataInfoCylindrical(scanner_ptr,
                          num_axial_pos_per_segment,
                          min_ring_diff_v, max_ring_diff_v,
                          num_views, num_tangential_poss)
{
  assert(scanner_ptr.use_count()!=0);
  ring_radius = scanner_ptr->get_ring_radius();
  angular_increment = _PI/scanner_ptr->get_num_detectors_per_ring();
}




ProjDataInfo*
ProjDataInfoCylindricalNoArcCorr::clone() const
{
  return static_cast<ProjDataInfo*>(new ProjDataInfoCylindricalNoArcCorr(*this));
}

string
ProjDataInfoCylindricalNoArcCorr::parameter_info()  const
{

#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[50000];
  ostrstream s(str, 50000);
#else
  std::ostringstream s;
#endif  
  s << "ProjDataInfoCylindricalNoArcCorr := \n";
  s << ProjDataInfoCylindrical::parameter_info();
  s << "End :=\n";
  return s.str();
}


END_NAMESPACE_STIR

