//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Non-inline implementations of ProjDataInfoCylindrical

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInfoCylindrical.h"
#include <algorithm>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#include "stir/round.h"


#ifndef STIR_NO_NAMESPACES
using std::min_element;
using std::max_element;
using std::min;
using std::max;
using std::swap;
using std::endl;
using std::ends;
#endif

START_NAMESPACE_STIR

ProjDataInfoCylindrical::
ProjDataInfoCylindrical()
{}


ProjDataInfoCylindrical::
ProjDataInfoCylindrical(const shared_ptr<Scanner>& scanner_ptr,
                        const VectorWithOffset<int>& num_axial_pos_per_segment,
                        const VectorWithOffset<int>& min_ring_diff_v, 
                        const VectorWithOffset<int>& max_ring_diff_v,
                        const int num_views,const int num_tangential_poss)
  :ProjDataInfo(scanner_ptr,num_axial_pos_per_segment, 
                num_views,num_tangential_poss),
   min_ring_diff(min_ring_diff_v),
   max_ring_diff(max_ring_diff_v)
{
  
  azimuthal_angle_sampling = _PI/num_views;
  ring_radius = get_scanner_ptr()->get_ring_radius();
  ring_spacing= get_scanner_ptr()->get_ring_spacing() ;

  // TODO this info should probably be provided via the constructor, or at
  // least by Scanner.
  sampling_corresponds_to_physical_rings =
    scanner_ptr->get_type() != Scanner::HiDAC;
  

  assert(min_ring_diff.get_length() == max_ring_diff.get_length());
  assert(min_ring_diff.get_length() == num_axial_pos_per_segment.get_length());

  // check min,max ring diff
  {
    for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
      if (min_ring_diff[segment_num]> max_ring_diff[segment_num])
      {
        warning("ProjDataInfoCylindrical: min_ring_difference %d is larger than max_ring_difference %d for segment %d. "
          "Swapping them around\n", 
          min_ring_diff[segment_num], max_ring_diff[segment_num], segment_num);
        swap(min_ring_diff[segment_num], max_ring_diff[segment_num]);
      }
  }

  initialise_ring_diff_arrays();
}

void
ProjDataInfoCylindrical::
initialise_ring_diff_arrays() const
{

  // check min,max ring diff
  {
    // check is necessary here again because of set_min_ring_difference()
    // we do not swap here because that would require the min/max_ring_diff arrays to be mutable as well
    for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
      if (min_ring_diff[segment_num]> max_ring_diff[segment_num])
      {
        error("ProjDataInfoCylindrical: min_ring_difference %d is larger than "
	      "max_ring_difference %d for segment %d.\n",
          min_ring_diff[segment_num], max_ring_diff[segment_num], segment_num);        
      }
  }
  // initialise m_offset 
  { 
    m_offset = 
      VectorWithOffset<float>(get_min_segment_num(),get_max_segment_num());
    
    /* m_offsets are found by requiring
    get_m(..., min_axial_pos_num,...) == - get_m(..., max_axial_pos_num,...)
    */
    for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
    {
      m_offset[segment_num] =
        ((get_max_axial_pos_num(segment_num) + get_min_axial_pos_num(segment_num))
	 *get_axial_sampling(segment_num)
        )/2;
    }
  }
  // initialise ax_pos_num_offset 
  if (sampling_corresponds_to_physical_rings)
  { 
    const int num_rings = get_scanner_ptr()->get_num_rings();
    ax_pos_num_offset =
      VectorWithOffset<int>(get_min_segment_num(),get_max_segment_num());
    
    /* ax_pos_num will be determined by looking at ring1+ring2.
       This also works for axially compressed data (i.e. span) as
       ring1+ring2 is constant for all ring-pairs combined into 1
       segment,ax_pos.

       Ignoring the difficulties of axial compression for a second, it is clear that
       for a given bin, there will be 2 rings as follows:
         ring1 = get_m(bin)/ring_spacing  + ring_diff/2 + (num_rings-1)/2
         ring2 = get_m(bin)/ring_spacing  - ring_diff/2 + (num_rings-1)/2
       This follows from the fact that get_m() returns the z position
       in millimeter of the middle of the LOR w.r.t. the middle of the scanner.
       The (num_rings-1)/2 shifts the origin such that the first ring has 
       ring_num==0.

       From the above, it follows that
         ring1+ring2=2*get_m(bin)/ring_spacing + (num_rings-1)
       Finally, we use the formula for get_m to obtain
         ring1+ring2=2*ax_pos_num/get_num_axial_poss_per_ring_inc(segment_num)
	             -2*m_offset[segment_num]/ring_spacing + (num_rings-1)
       Solving this for ax_pos_num:
         ax_pos_num = (ring1+ring2-(num_rings-1)
                       + 2*m_offset[segment_num]/ring_spacing
		      ) * get_num_axial_poss_per_ring_inc(segment_num)/2

       We could plug m_offset in to obtain
         ax_pos_num = (ring1+ring2-(num_rings-1)
		      ) * get_num_axial_poss_per_ring_inc(segment_num)/2.
                      +
		      (get_max_axial_pos_num(segment_num) 
		        + get_min_axial_pos_num(segment_num) )/2.
       this formula is easy to understand, but we don't use it as
       at some point somebody might change m_offset
       and forget to change this code... 
       (also, the form above would need float division and then rounding)
       */
    for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
    {
      ax_pos_num_offset[segment_num] =
        round((num_rings-1) - 2*m_offset[segment_num]/ring_spacing);
      // check that it was integer
      assert(fabs(ax_pos_num_offset[segment_num] -
		  ((num_rings-1) - 2*m_offset[segment_num]/ring_spacing)) < 1E-4);

      if (get_num_axial_poss_per_ring_inc(segment_num)==1)
	{
	  // check that we'll get an integer ax_pos_num, i.e. 
	  // (ring1+ring2  - ax_pos_num_offset) has to be even, for any
          // ring1,ring2 in the segment, i.e ring1-ring2 = ring_diff, so
	  // ring1+ring2 = 2*ring2 + ring_diff
	  assert(get_min_ring_difference(segment_num) ==
		 get_max_ring_difference(segment_num));
	  if ((get_max_ring_difference(segment_num) -
	       ax_pos_num_offset[segment_num]) % 2 != 0)
	    warning("ProjDataInfoCylindrical: the number of axial positions in "
		    "segment %d is such that current conventions will place "
		    "the LORs shifted with respect to the physical rings.\n",
		    segment_num);
      }
    }
  }
  // initialise ring_diff_to_segment_num
  if (sampling_corresponds_to_physical_rings)
  {
    const int min_ring_difference = 
      *min_element(min_ring_diff.begin(), min_ring_diff.end());
    const int max_ring_difference = 
      *max_element(max_ring_diff.begin(), max_ring_diff.end());

    // set ring_diff_to_segment_num to appropriate size
    // in principle, the max ring difference would be scanner.num_rings-1, but 
    // in case someone is up to strange things, we take the max of this value 
    // with the max_ring_difference as given in the file
    ring_diff_to_segment_num =
      VectorWithOffset<int>(min(min_ring_difference, -(get_scanner_ptr()->get_num_rings()-1)),
                            max(max_ring_difference, get_scanner_ptr()->get_num_rings()-1));
    // first set all to impossible value
    // warning: get_segment_num_for_ring_difference relies on the fact that this value
    // is larger than get_max_segment_num()
    ring_diff_to_segment_num.fill(get_max_segment_num()+1);

    for(int ring_diff=min_ring_difference; ring_diff <= max_ring_difference; ++ring_diff) 
    {    
      int segment_num;
      for (segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
      {
        if (ring_diff >= min_ring_diff[segment_num] &&
            ring_diff <= max_ring_diff[segment_num])
        {
#if 0
          std::cerr << "ring diff " << ring_diff << " stored in s:" << segment_num << std::endl;
#endif
          ring_diff_to_segment_num[ring_diff] = segment_num;
          break;
        }
      }
      if (segment_num>get_max_segment_num())
      {
        warning("ProjDataInfoCylindrical: ring difference %d does not belong to a segment\n",
          ring_diff);
      }
    }
  }

  ring_diff_arrays_computed = true;
}

void
ProjDataInfoCylindrical::
get_ring_pair_for_segment_axial_pos_num(int& ring1,
					int& ring2,
					const int segment_num,
					const int axial_pos_num) const
{
  if (!sampling_corresponds_to_physical_rings)
    error("ProjDataInfoCylindrical::get_ring_pair_for_segment_axial_pos_num does not work for this type of sampled data\n");
  // can do only span=1 at the moment
  if (get_min_ring_difference(segment_num) != get_max_ring_difference(segment_num))
    error("ProjDataInfoCylindrical::get_ring_pair_for_segment_axial_pos_num does not work for data with axial compression\n");

  // see documentation above for formulas
        
  const int ring1_plus_ring2 =
    round(2*axial_pos_num/get_num_axial_poss_per_ring_inc(segment_num)
	  -2*m_offset[segment_num]/ring_spacing + (get_scanner_ptr()->get_num_rings()-1));
  // check that it was integer
  assert(fabs(
	      ring1_plus_ring2 -
	      (2*axial_pos_num/get_num_axial_poss_per_ring_inc(segment_num)
	       -2*m_offset[segment_num]/ring_spacing + (get_scanner_ptr()->get_num_rings()-1))
	      ) < 1E-4) ;

  const int ring_diff = get_max_ring_difference(segment_num);

  // KT 01/08/2002 swapped rings
  ring1 = (ring1_plus_ring2 - ring_diff)/2;
  ring2 = (ring1_plus_ring2 + ring_diff)/2;
  assert((ring1_plus_ring2 + ring_diff)%2 == 0);
  assert((ring1_plus_ring2 - ring_diff)%2 == 0);
}

/*
void
ProjDataInfoCylindrical::
set_azimuthal_angle_sampling(const float angle_v)
{azimuthal_angle_sampling =  angle_v;}

void
ProjDataInfoCylindrical::
set_axial_sampling(const float samp_v, int segment_num)
{axial_sampling = samp_v;}
*/

void
ProjDataInfoCylindrical::
set_min_ring_difference( int min_ring_diff_v, int segment_num)
{
  ring_diff_arrays_computed = false;
  min_ring_diff[segment_num] = min_ring_diff_v;
}

void
ProjDataInfoCylindrical::
set_max_ring_difference( int max_ring_diff_v, int segment_num)
{
  ring_diff_arrays_computed = false;
  max_ring_diff[segment_num] = max_ring_diff_v;
}

void
ProjDataInfoCylindrical::
set_ring_spacing(float ring_spacing_v)
{
  ring_diff_arrays_computed = false;
  ring_spacing = ring_spacing_v;
}


string
ProjDataInfoCylindrical::parameter_info()  const
{

#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[30000];
  ostrstream s(str, 30000);
#else
  std::ostringstream s;
#endif  
  s << ProjDataInfo::parameter_info();

  s << "ring differences per segment: \n";
  for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
  {
    s << '(' << min_ring_diff[segment_num]  << ',' << max_ring_diff[segment_num] <<')';
  }
  s << "\n";
  s << ends;
  return s.str();
}

END_NAMESPACE_STIR
