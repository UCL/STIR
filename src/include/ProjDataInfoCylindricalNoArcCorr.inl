//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementation of inline functions of class 
  ProjDataInfoCylindricalNoArcCorr

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "Bin.h"
#include "tomo/Succeeded.h"
#include <math.h>

START_NAMESPACE_TOMO

float
ProjDataInfoCylindricalNoArcCorr::get_s(const Bin& bin) const
{
  return ring_radius * sin(bin.tangential_pos_num()*angular_increment);
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
get_det_num_pair_for_view_tangential_pos_num(
					     int& det_num1,
					     int& det_num2,
					     const int view_num,
					     const int tang_pos_num) const
{
  assert(get_view_mashing_factor() == 1);
  // TODO check on 180 degrees for views
  const int num_detectors =
    get_scanner_ptr()->get_num_detectors_per_ring();

  assert(num_detectors%2 == 0);
  /*
     adapted from CTI code
     Note for implementation: avoid using % with negative numbers
     so add num_detectors before doing modulo num_detectors)
  */
  det_num1 = (view_num + (tang_pos_num >> 1) + num_detectors) % num_detectors;
  det_num2 = (view_num - ( (tang_pos_num + 1) >> 1 ) + num_detectors/2) % num_detectors;
}


bool 
ProjDataInfoCylindricalNoArcCorr::
get_view_tangential_pos_num_for_det_num_pair(int& view_num,
					     int& tang_pos_num,
					     const int det_num1,
					     const int det_num2) const
{
  /*
     Note for implementation: avoid using % with negative numbers
     so add 2*nv before doing modulo 2*nv

     This somewhat obscure formula was obtained by inverting the code above
     TODO This can be simplified (especially all the branching later on).
 */
  // TODO check on 180 degrees for views
  const int num_detectors =
    get_scanner_ptr()->get_num_detectors_per_ring();

  assert(num_detectors%2 == 0);
  const int max_num_views = num_detectors/2;

  int swap_detectors;
  tang_pos_num = (det_num1 - det_num2 +  3*num_detectors/2) % num_detectors;
  int view = (det_num1 - (tang_pos_num >> 1) +  num_detectors) % num_detectors;

  /* Now adjust ranges for view, tang_pos_num.
     The next lines go only wrong in the singular (and irrelevant) case
     det_num1 == det_num2 (when tang_pos_num == num_detectors - tang_pos_num)

     We use the combinations of the following 'symmetries' of
     (tang_pos_num, view) == (tang_pos_num+2*num_views, view + num_views)
                 == (-tang_pos_num, view + num_views)
     Using the latter interchanges det_num1 and det_num2, and this leaves
     the LOR the same the 2D case. However, in 3D this interchanges the rings
     as well. So, we keep track of this in swap_detectors, and return its final
     value.
     */
  if (view <  max_num_views)
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
      view -= max_num_views;
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

  view_num = view/get_view_mashing_factor();
  return swap_detectors==0;
}


Succeeded 
ProjDataInfoCylindricalNoArcCorr::
get_bin_for_det_pair(Bin& bin,
		     const int det_num1, const int ring_num1,
		     const int det_num2, const int ring_num2) const
{  
  if (get_view_tangential_pos_num_for_det_num_pair(bin.view_num(), bin.tangential_pos_num(), det_num1, det_num2))
    return get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), ring_num1, ring_num2);
  else
    return get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), ring_num2, ring_num1);
}

void
ProjDataInfoCylindricalNoArcCorr::
get_det_pair_for_bin(
		     int& det_num1, int& ring_num1,
		     int& det_num2, int& ring_num2,
		     const Bin& bin) const
{
  get_det_num_pair_for_view_tangential_pos_num(det_num1, det_num2, bin.view_num(), bin.tangential_pos_num());
  get_ring_pair_for_segment_axial_pos_num( ring_num1, ring_num2, bin.segment_num(), bin.axial_pos_num());

}


END_NAMESPACE_TOMO

