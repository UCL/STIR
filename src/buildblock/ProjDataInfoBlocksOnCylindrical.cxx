
/*

TODO copyright and License

*/

/*!
  \file
  \ingroup projdata

  \brief  Non-inline implementations of stir::ProjDataInfoBlocksOnCylindrical

  \author Parisa Khateri

*/
#include "stir/ProjDataInfoBlocksOnCylindrical.h"
#include "stir/LORCoordinates.h"
#include <algorithm>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#include "stir/round.h"
#include <math.h>

#ifndef STIR_NO_NAMESPACES
using std::min_element;
using std::max_element;
using std::min;
using std::max;
using std::swap;
using std::endl;
using std::string;
using std::pair;
using std::vector;
#endif

START_NAMESPACE_STIR

ProjDataInfoBlocksOnCylindrical::
ProjDataInfoBlocksOnCylindrical()
{}


ProjDataInfoBlocksOnCylindrical::
ProjDataInfoBlocksOnCylindrical(const shared_ptr<Scanner>& scanner_ptr,
                        const VectorWithOffset<int>& num_axial_pos_per_segment,
                        const VectorWithOffset<int>& min_ring_diff_v,
                        const VectorWithOffset<int>& max_ring_diff_v,
                        const int num_views,const int num_tangential_poss)
  :ProjDataInfo(scanner_ptr,num_axial_pos_per_segment,
                num_views,num_tangential_poss),
   min_ring_diff(min_ring_diff_v),
   max_ring_diff(max_ring_diff_v)
{

  azimuthal_angle_sampling = static_cast<float>(_PI/num_views);
  ring_radius.resize(0,0);
  ring_radius[0] = get_scanner_ptr()->get_effective_ring_radius();
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
        warning("ProjDataInfoBlocksOnCylindrical: min_ring_difference %d is larger than max_ring_difference %d for segment %d. "
          "Swapping them around",
          min_ring_diff[segment_num], max_ring_diff[segment_num], segment_num);
        swap(min_ring_diff[segment_num], max_ring_diff[segment_num]);
      }
  }

  initialise_ring_diff_arrays();
}

/* warning In cylindrical geometry m_offset is calculated based on axial_spacing,
 then it is used to caculate ax_pos_num_offset and segment_axial_pos_to_ring1_plus_ring2.
 For block geometry, m_offset has been removed and the above mentioned variables are
 calculated independant of m_offset.*/
void
ProjDataInfoBlocksOnCylindrical::
initialise_ring_diff_arrays() const
{

  // check min,max ring diff
  {
    // check is necessary here again because of set_min_ring_difference()
    // we do not swap here because that would require the min/max_ring_diff arrays to be mutable as well
    for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
      if (min_ring_diff[segment_num]> max_ring_diff[segment_num])
      {
        error("ProjDataInfoBlocksOnCylindrical: min_ring_difference %d is larger than "
	      "max_ring_difference %d for segment %d.",
          min_ring_diff[segment_num], max_ring_diff[segment_num], segment_num);
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
         ring2 = get_m(bin)/ring_spacing  + ring_diff/2 + (num_rings-1)/2
         ring1 = get_m(bin)/ring_spacing  - ring_diff/2 + (num_rings-1)/2
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
		const float ax_pos_num_offset_float = (num_rings-1) -
				(get_max_axial_pos_num(segment_num) +
				get_min_axial_pos_num(segment_num))
				/get_num_axial_poss_per_ring_inc(segment_num);
        ax_pos_num_offset[segment_num] = round(ax_pos_num_offset_float);
        // check that it was integer
        if (fabs(ax_pos_num_offset[segment_num] - ax_pos_num_offset_float) > 1E-4)
		{
 	       error("ProjDataInfoBlocksOnCylindrical: in segment %d, the axial positions\n"
 	       "do not correspond to the usual locations between physical rings.\n"
 	       "This is suspicious and can make things go wrong in STIR, so I abort.\n"
 	       "Check the number of axial positions in this segment.",
 	       segment_num);
		}

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
                warning("ProjDataInfoBlocksOnCylindrical: the number of axial positions in "
                        "segment %d is such that current conventions will place "
                        "the LORs shifted with respect to the physical rings.",
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
        warning("ProjDataInfoBlocksOnCylindrical: ring difference %d does not belong to a segment",
          ring_diff);
      }
    }
  }
  // initialise segment_axial_pos_to_ring1_plus_ring2
  if (sampling_corresponds_to_physical_rings)
  {
    segment_axial_pos_to_ring1_plus_ring2 =
      VectorWithOffset<VectorWithOffset<int> >(get_min_segment_num(), get_max_segment_num());
    for (int s_num=get_min_segment_num(); s_num<=get_max_segment_num(); ++s_num)
    {
      const int min_ax_pos_num = get_min_axial_pos_num(s_num);
      const int max_ax_pos_num = get_max_axial_pos_num(s_num);
      segment_axial_pos_to_ring1_plus_ring2[s_num].grow(min_ax_pos_num, max_ax_pos_num);
      for (int ax_pos_num=min_ax_pos_num; ax_pos_num<=max_ax_pos_num; ++ax_pos_num)
      {
         // see documentation above for formulas
        const float ring1_plus_ring2_float =
					(2*ax_pos_num -
					 get_max_axial_pos_num(s_num) + get_min_axial_pos_num(s_num))
					 /get_num_axial_poss_per_ring_inc(s_num) +
					 get_scanner_ptr()->get_num_rings()-1;

        const int ring1_plus_ring2 =
          round(ring1_plus_ring2_float);
        // check that it was integer
        assert(fabs(ring1_plus_ring2 - ring1_plus_ring2_float) < 1E-4) ;
        segment_axial_pos_to_ring1_plus_ring2[s_num][ax_pos_num] = ring1_plus_ring2;
      }
    }
  }

  if (sampling_corresponds_to_physical_rings)
    allocate_segment_axial_pos_to_ring_pair();

  ring_diff_arrays_computed = true;
}

/*! Default implementation checks common variables. Needs to be overloaded.
 */
bool
ProjDataInfoBlocksOnCylindrical::
blindly_equals(const root_type * const that) const
{
  if (!base_type::blindly_equals(that))
    return false;

  const self_type& proj_data_info = static_cast<const self_type&>(*that);
  return
    this->azimuthal_angle_sampling == proj_data_info.azimuthal_angle_sampling &&
    this->ring_radius == proj_data_info.ring_radius &&
    this->sampling_corresponds_to_physical_rings == proj_data_info.sampling_corresponds_to_physical_rings &&
    this->ring_spacing == proj_data_info.ring_spacing &&
    this->min_ring_diff == proj_data_info.min_ring_diff &&
    this->max_ring_diff == proj_data_info.max_ring_diff;
}

void
ProjDataInfoBlocksOnCylindrical::
get_ring_pair_for_segment_axial_pos_num(int& ring1,
					int& ring2,
					const int segment_num,
					const int axial_pos_num) const
{
  if (!sampling_corresponds_to_physical_rings)
    error("ProjDataInfoBlocksOnCylindrical::get_ring_pair_for_segment_axial_pos_num does not work for this type of sampled data");
  // can do only span=1 at the moment
  if (get_min_ring_difference(segment_num) != get_max_ring_difference(segment_num))
    error("ProjDataInfoBlocksOnCylindrical::get_ring_pair_for_segment_axial_pos_num does not work for data with axial compression");

  if (!ring_diff_arrays_computed)
    initialise_ring_diff_arrays();

  const int ring_diff = get_max_ring_difference(segment_num);
  const int ring1_plus_ring2= segment_axial_pos_to_ring1_plus_ring2[segment_num][axial_pos_num];

  // KT 01/08/2002 swapped rings
  ring1 = (ring1_plus_ring2 - ring_diff)/2;
  ring2 = (ring1_plus_ring2 + ring_diff)/2;
  assert((ring1_plus_ring2 + ring_diff)%2 == 0);
  assert((ring1_plus_ring2 - ring_diff)%2 == 0);
}


void
ProjDataInfoBlocksOnCylindrical::
set_azimuthal_angle_sampling(const float angle_v)
{
	azimuthal_angle_sampling =  angle_v;
}

//void
//ProjDataInfoBlocksOnCylindrical::
//set_axial_sampling(const float samp_v, int segment_num)
//{axial_sampling = samp_v;}


void
ProjDataInfoBlocksOnCylindrical::
set_num_views(const int new_num_views)
{
  const float old_azimuthal_angle_range =
    this->get_azimuthal_angle_sampling() * this->get_num_views();
  base_type::set_num_views(new_num_views);
  this->azimuthal_angle_sampling = old_azimuthal_angle_range/this->get_num_views();
}

void
ProjDataInfoBlocksOnCylindrical::
set_min_ring_difference( int min_ring_diff_v, int segment_num)
{
  ring_diff_arrays_computed = false;
  min_ring_diff[segment_num] = min_ring_diff_v;
}

void
ProjDataInfoBlocksOnCylindrical::
set_max_ring_difference( int max_ring_diff_v, int segment_num)
{
  ring_diff_arrays_computed = false;
  max_ring_diff[segment_num] = max_ring_diff_v;
}

void
ProjDataInfoBlocksOnCylindrical::
set_ring_spacing(float ring_spacing_v)
{
  ring_diff_arrays_computed = false;
  ring_spacing = ring_spacing_v;
}

void
ProjDataInfoBlocksOnCylindrical::
allocate_segment_axial_pos_to_ring_pair() const
{
  segment_axial_pos_to_ring_pair =
    VectorWithOffset<VectorWithOffset<shared_ptr<RingNumPairs> > >
    (get_min_segment_num(), get_max_segment_num());

  for (int segment_num = get_min_segment_num();
       segment_num <= get_max_segment_num();
       ++segment_num)
    {
      segment_axial_pos_to_ring_pair[segment_num].grow(get_min_axial_pos_num(segment_num),
						       get_max_axial_pos_num(segment_num));
    }
}

void
ProjDataInfoBlocksOnCylindrical::
compute_segment_axial_pos_to_ring_pair(const int segment_num, const int axial_pos_num) const
{
  shared_ptr<RingNumPairs> new_el(new RingNumPairs);
  segment_axial_pos_to_ring_pair[segment_num][axial_pos_num] = new_el;

  RingNumPairs& table =
    *segment_axial_pos_to_ring_pair[segment_num][axial_pos_num];
  table.reserve(get_max_ring_difference(segment_num) -
		get_min_ring_difference(segment_num) + 1);

  /* We compute the lookup-table in a fancy way.
     We could just as well have a simple loop over all ring pairs and check
     if it belongs to this segment/axial_pos.
     The current way is a lot faster though.
  */
  const int min_ring_diff = get_min_ring_difference(segment_num);
  const int max_ring_diff = get_max_ring_difference(segment_num);
  const int num_rings = get_scanner_ptr()->get_num_rings();

  /* ring1_plus_ring2 is the same for any ring pair that contributes to
     this particular segment_num, axial_pos_num.
  */
  const int ring1_plus_ring2=
    segment_axial_pos_to_ring1_plus_ring2[segment_num][axial_pos_num];

  /*
    The ring_difference increments with 2 as the other ring differences do
    not give a ring pair with this axial_position. This is because
    ring1_plus_ring2%2 == ring_diff%2
    (which easily follows by plugging in ring1+ring2 and ring1-ring2).
    The starting ring_diff is determined such that the above condition
    is satisfied. You can check it by noting that the
      start_ring_diff%2
        == (min_ring_diff + (min_ring_diff+ring1_plus_ring2)%2)%2
	== (2*min_ring_diff+ring1_plus_ring2)%2
	== ring1_plus_ring2%2
  */
  for(int ring_diff = min_ring_diff + (min_ring_diff+ring1_plus_ring2)%2;
      ring_diff <= max_ring_diff;
      ring_diff+=2 )
    {
      const int ring1 = (ring1_plus_ring2 - ring_diff)/2;
      const int ring2 = (ring1_plus_ring2 + ring_diff)/2;
      if (ring1<0 || ring2 < 0 || ring1>=num_rings || ring2 >= num_rings)
	continue;
      assert((ring1_plus_ring2 + ring_diff)%2 == 0);
      assert((ring1_plus_ring2 - ring_diff)%2 == 0);
      table.push_back(pair<int,int>(ring1, ring2));
 #ifndef NDEBUG
      int check_segment_num = 0, check_axial_pos_num = 0;
      assert(get_segment_axial_pos_num_for_ring_pair(check_segment_num,
						     check_axial_pos_num,
						     ring1,
						     ring2) ==
	     Succeeded::yes);
      assert(check_segment_num == segment_num);
      assert(check_axial_pos_num == axial_pos_num);
 #endif
    }
}

void
ProjDataInfoBlocksOnCylindrical::
set_num_axial_poss_per_segment(const VectorWithOffset<int>& num_axial_poss_per_segment)
{
  ProjDataInfo::set_num_axial_poss_per_segment(num_axial_poss_per_segment);
  ring_diff_arrays_computed = false;
}

void
ProjDataInfoBlocksOnCylindrical::
set_min_axial_pos_num(const int min_ax_pos_num, const int segment_num)
{
  ProjDataInfo::set_min_axial_pos_num(min_ax_pos_num, segment_num);
  ring_diff_arrays_computed = false;
}

void ProjDataInfoBlocksOnCylindrical::
set_max_axial_pos_num(const int max_ax_pos_num, const int segment_num)
{
  ProjDataInfo::set_max_axial_pos_num(max_ax_pos_num, segment_num);
  ring_diff_arrays_computed = false;
}

void
ProjDataInfoBlocksOnCylindrical::
reduce_segment_range(const int min_segment_num, const int max_segment_num)
{
  ProjDataInfo::reduce_segment_range(min_segment_num, max_segment_num);
  // reduce ring_diff arrays to new valid size
  VectorWithOffset<int> new_min_ring_diff(min_segment_num, max_segment_num);
  VectorWithOffset<int> new_max_ring_diff(min_segment_num, max_segment_num);

  for (int segment_num = min_segment_num; segment_num<= max_segment_num; ++segment_num)
  {
    new_min_ring_diff[segment_num] = this->min_ring_diff[segment_num];
    new_max_ring_diff[segment_num] = this->max_ring_diff[segment_num];
  }

  this->min_ring_diff = new_min_ring_diff;
  this->max_ring_diff = new_max_ring_diff;

  // make sure other arrays will be updated if/when necessary
  this->ring_diff_arrays_computed = false;
}

//! warning Find lor from cartesian coordinates of detector pair
void
ProjDataInfoBlocksOnCylindrical::
get_LOR(LORInAxialAndNoArcCorrSinogramCoordinates<float>& lor,
	const Bin& bin) const
{
	CartesianCoordinate3D<float> _p1;
	CartesianCoordinate3D<float> _p2;
	find_cartesian_coordinates_of_detection(_p1, _p2, bin);

	LORAs2Points<float> lor_as_2_points(_p1, _p2);
	const double R = get_ring_radius();
  lor_as_2_points.change_representation_for_block(lor, R);
}

string
ProjDataInfoBlocksOnCylindrical::parameter_info()  const
{

#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[30000];
  ostrstream s(str, 30000);
#else
  std::ostringstream s;
#endif
  s << ProjDataInfo::parameter_info();
  s << "Azimuthal angle increment (deg):   " << get_azimuthal_angle_sampling()*180/_PI << '\n';
  s << "Azimuthal angle extent (deg):      " << fabs(get_azimuthal_angle_sampling())*get_num_views()*180/_PI << '\n';

  s << "ring differences per segment: \n";
  for (int segment_num=get_min_segment_num(); segment_num<=get_max_segment_num(); ++segment_num)
  {
    s << '(' << min_ring_diff[segment_num]  << ',' << max_ring_diff[segment_num] <<')';
  }
  s << std::endl;
  return s.str();
}

END_NAMESPACE_STIR
