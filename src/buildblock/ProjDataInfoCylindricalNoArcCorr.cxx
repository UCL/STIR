//
// $Id$
//
/*!

  \file
  \ingroup projdata

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
#include "stir/Bin.h"
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
{
  view_tangpos_to_det1det2_initialised = false;
  det1det2_to_view_tangpos_initialised = false;
}

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
  view_tangpos_to_det1det2_initialised = false;
  det1det2_to_view_tangpos_initialised = false;
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

/*
   TODO make compile time assert

   Warning:
   this code makes use of an implementation dependent feature:
   bit shifting negative ints to the right.
    -1 >> 1 should be -1
    -2 >> 1 should be -1
   This is ok on SUNs (gcc, but probably SUNs cc as well), Parsytec (gcc),
   Pentium (gcc, VC++) and probably every other system which uses
   the 2-complement convention.
*/

/*
  Go from sinograms to detectors.
  Because sinograms are not arc-corrected, tang_pos_num corresponds
  to an angle as well. Before interleaving we have that
  \verbatim
  det_angle_1 = LOR_angle + bin_angle
  det_angle_2 = LOR_angle + (Pi - bin_angle)
  \endverbatim
  (Hint: understand this first at LOR_angle=0, then realise that
  other LOR_angles follow just by rotation)

  Code gets slightly intricate because:
  - angles have to be defined modulo 2 Pi (so num_detectors)
  - interleaving
*/
void 
ProjDataInfoCylindricalNoArcCorr::
initialise_view_tangpos_to_det1det2() const
{
  const int num_detectors =
    get_scanner_ptr()->get_num_detectors_per_ring();

  assert(num_detectors%2 == 0);
  assert(get_min_view_num() == 0);
  assert(get_max_view_num() == num_detectors/2 - 1);
  // check views range from 0 to Pi
  assert(fabs(get_phi(Bin(0,0,0,0))) < 1.E-4);
  assert(fabs(get_phi(Bin(0,num_detectors/2,0,0)) - _PI) < 1.E-4);
  const int min_tang_pos_num = -(num_detectors/2)+1;
  const int max_tang_pos_num = -(num_detectors/2)+num_detectors;
  
  view_tangpos_to_det1det2.grow(0,num_detectors/2-1);
  for (int v_num=0; v_num<=num_detectors/2-1; ++v_num)
  {
    view_tangpos_to_det1det2[v_num].grow(min_tang_pos_num, max_tang_pos_num);

    for (int tp_num=min_tang_pos_num; tp_num<=max_tang_pos_num; ++tp_num)
    {
      /*
         adapted from CTI code
         Note for implementation: avoid using % with negative numbers
         so add num_detectors before doing modulo num_detectors)
        */
      view_tangpos_to_det1det2[v_num][tp_num].det1_num = 
        (v_num + (tp_num >> 1) + num_detectors) % num_detectors;
      view_tangpos_to_det1det2[v_num][tp_num].det2_num = 
        (v_num - ( (tp_num + 1) >> 1 ) + num_detectors/2) % num_detectors;
    }
  }
  view_tangpos_to_det1det2_initialised = true;
}

void 
ProjDataInfoCylindricalNoArcCorr::
initialise_det1det2_to_view_tangpos() const
{
  const int num_detectors =
    get_scanner_ptr()->get_num_detectors_per_ring();

  assert(num_detectors%2 == 0);
  assert(get_min_view_num() == 0);
  // check views range from 0 to Pi
  assert(fabs(get_phi(Bin(0,0,0,0))) < 1.E-4);
  assert(fabs(get_phi(Bin(0,get_max_view_num()+1,0,0)) - _PI) < 1.E-4);
  const int min_tang_pos_num = -(num_detectors/2);
  const int max_tang_pos_num = -(num_detectors/2)+num_detectors;
  const int max_num_views = num_detectors/2;

  det1det2_to_view_tangpos.grow(0,num_detectors-1);
  for (int det1_num=0; det1_num<num_detectors; ++det1_num)
  {
    det1det2_to_view_tangpos[det1_num].grow(0, num_detectors-1);

    for (int det2_num=0; det2_num<num_detectors; ++det2_num)
    {            
      if (det1_num == det2_num)
	  continue;
      /*
       This somewhat obscure formula was obtained by inverting the code for
       get_det_num_pair_for_view_tangential_pos_num()
       This can be simplified (especially all the branching later on), but
       as we execute this code only occasionally, it's probably not worth it.
      */
      int swap_detectors;
      /*
      Note for implementation: avoid using % with negative numbers
      so add num_detectors before doing modulo num_detectors
      */
      int tang_pos_num = (det1_num - det2_num +  3*num_detectors/2) % num_detectors;
      int view_num = (det1_num - (tang_pos_num >> 1) +  num_detectors) % num_detectors;
      
      /* Now adjust ranges for view_num, tang_pos_num.
      The next lines go only wrong in the singular (and irrelevant) case
      det_num1 == det_num2 (when tang_pos_num == num_detectors - tang_pos_num)
      
        We use the combinations of the following 'symmetries' of
        (tang_pos_num, view_num) == (tang_pos_num+2*num_views, view_num + num_views)
        == (-tang_pos_num, view_num + num_views)
        Using the latter interchanges det_num1 and det_num2, and this leaves
        the LOR the same in the 2D case. However, in 3D this interchanges the rings
        as well. So, we keep track of this in swap_detectors, and return its final
        value.
      */
      if (view_num <  max_num_views)
      {
        if (tang_pos_num >=  max_num_views)
        {
          tang_pos_num = num_detectors - tang_pos_num;
          swap_detectors = 1;
        }
        else
        {
          swap_detectors = 0;
        }
      }
      else
      {
        view_num -= max_num_views;
        if (tang_pos_num >=  max_num_views)
        {
          tang_pos_num -= num_detectors;
          swap_detectors = 0;
        }
        else
        {
          tang_pos_num *= -1;
          swap_detectors = 1;
        }
      }
      
      det1det2_to_view_tangpos[det1_num][det2_num].view_num = view_num;
      det1det2_to_view_tangpos[det1_num][det2_num].tang_pos_num = tang_pos_num;
      det1det2_to_view_tangpos[det1_num][det2_num].swap_detectors = swap_detectors==0;     
    }
  }
  det1det2_to_view_tangpos_initialised = true;
}
END_NAMESPACE_STIR

