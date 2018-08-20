
/*

TODO copyright and License

*/

/*!
  \file
  \ingroup projdata

  \brief Implementation of inline functions of class stir::ProjDataInfoBlocksOnCylindrical

  \author Parisa Khateri

*/
// for sqrt
#include <math.h>
#include "stir/Bin.h"
#include "stir/Succeeded.h"
#include "stir/LORCoordinates.h"
#include "stir/is_null_ptr.h"
#include <algorithm>

START_NAMESPACE_STIR

void
ProjDataInfoBlocksOnCylindrical::
initialise_ring_diff_arrays_if_not_done_yet() const
{
  // for efficiency reasons, use "Double-Checked-Locking(DCL) pattern" with OpenMP atomic operation
  // OpenMP v3.1 or later required
  // thanks to yohjp: http://stackoverflow.com/questions/27975737/how-to-handle-cached-data-structures-with-multi-threading-e-g-openmp
#if defined(STIR_OPENMP) &&  _OPENMP >=201012
  bool initialised;
#pragma omp atomic read
  initialised = ring_diff_arrays_computed;

  if (!initialised)
#endif
    {
#if defined(STIR_OPENMP)
#pragma omp critical(PROJDATAINFOCYLINDRICALRINGDIFFARRAY)
#endif
      {
        if (!ring_diff_arrays_computed)
          initialise_ring_diff_arrays();
      }
    }
}

//! find phi from correspoding lor
float
ProjDataInfoBlocksOnCylindrical::get_phi(const Bin& bin)const
{
	LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
	get_LOR(lor, bin);
	if (bin.view_num()==0 && lor.phi()>0.1)
		return lor.phi()-_PI;
	return lor.phi();
}

/*! warning In block geometry m is calculated directly from lor while in
	cylindrical geometry m is calculated using m_offset and axial_sampling
*/
float
ProjDataInfoBlocksOnCylindrical::get_m(const Bin& bin) const
{
	LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
	get_LOR(lor, bin);
	//Parisa to Check
	//std::cout<<"seg_ax_m_z1 = "<<bin.segment_num()<<"\t"<<bin.axial_pos_num()<<"\t"<<lor.z1()<<"\t"<<(lor.z1() + lor.z2())/2.<<std::endl;
	return (lor.z1() + lor.z2())/2.;
}

float
ProjDataInfoBlocksOnCylindrical::get_t(const Bin& bin) const
{
  return
    get_m(bin)*get_costheta(bin);
}

/*
	theta is copolar angle of normal to projection plane with z axis, i.e. copolar angle of lor with z axis.
	tan (theta) = dz/sqrt(dx2+dy2)
	cylindrical geometry:
		delta_z = delta_ring * ring spacing
	Block geometry:
		delta_z is calculated from lor
*/
float
ProjDataInfoBlocksOnCylindrical::get_tantheta(const Bin& bin) const
{
	LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
	get_LOR(lor, bin);
	const float delta_z = lor.z2() - lor.z1();
  if (fabs(delta_z)<0.0001F)
		return 0;
	const float R=get_ring_radius(bin.view_num());
	assert(R>=fabs(get_s(bin)));
	return delta_z/(2*sqrt(square(R)-square(get_s(bin))));
}

float
ProjDataInfoBlocksOnCylindrical::get_sampling_in_m(const Bin& bin) const
{
  return get_axial_sampling(bin.segment_num());
}

/*
	warning Theta is not uniform anymore, so sampling in t does not make sense anymore.
	Sampling parameters remained unchanged to be consistent with cylindrical version.
*/
float
ProjDataInfoBlocksOnCylindrical::get_sampling_in_t(const Bin& bin) const
{
  return get_axial_sampling(bin.segment_num())*get_costheta(bin);
}

int
ProjDataInfoBlocksOnCylindrical::
get_num_axial_poss_per_ring_inc(const int segment_num) const
{
  return
    max_ring_diff[segment_num] != min_ring_diff[segment_num] ?
    2 : 1;
}


float
ProjDataInfoBlocksOnCylindrical::get_azimuthal_angle_sampling() const
{return azimuthal_angle_sampling;}


float
ProjDataInfoBlocksOnCylindrical::get_axial_sampling(int segment_num) const
{
  return ring_spacing/get_num_axial_poss_per_ring_inc(segment_num);
}

float
ProjDataInfoBlocksOnCylindrical::get_average_ring_difference(int segment_num) const
{
  // KT 05/07/2001 use float division here.
  // In any reasonable case, min+max_ring_diff will be even.
  // But some day, an unreasonable case will walk in.
  return (min_ring_diff[segment_num] + max_ring_diff[segment_num])/2.F;
}

int
ProjDataInfoBlocksOnCylindrical::get_min_ring_difference(int segment_num) const
{ return min_ring_diff[segment_num]; }

int
ProjDataInfoBlocksOnCylindrical::get_max_ring_difference(int segment_num) const
{ return max_ring_diff[segment_num]; }

float
ProjDataInfoBlocksOnCylindrical::get_ring_radius() const
{
  if (this->ring_radius.get_min_index()!=0 || this->ring_radius.get_max_index()!=0)
    {
      // check if all elements are equal
      for (VectorWithOffset<float>::const_iterator iter=this->ring_radius.begin(); iter!= this->ring_radius.end(); ++iter)
	{
	  if (*iter != *this->ring_radius.begin())
	    error("get_ring_radius called for non-circular ring");
	}
    }
  return *this->ring_radius.begin();
}

void
ProjDataInfoBlocksOnCylindrical::set_ring_radii_for_all_views(const VectorWithOffset<float>& new_ring_radius)
{
  if (new_ring_radius.get_min_index() != this->get_min_view_num() ||
      new_ring_radius.get_max_index() != this->get_max_view_num())
    {
      error("error set_ring_radii_for_all_views: you need to use correct range of view numbers");
    }

  this->ring_radius = new_ring_radius;
}

VectorWithOffset<float>
ProjDataInfoBlocksOnCylindrical::get_ring_radii_for_all_views() const
{
  if (this->ring_radius.get_min_index()==0 && this->ring_radius.get_max_index()==0)
    {
      VectorWithOffset<float> out(this->get_min_view_num(), this->get_max_view_num());
      out.fill(this->ring_radius[0]);
      return out;
    }
  else
    return this->ring_radius;
}

float
ProjDataInfoBlocksOnCylindrical::get_ring_radius( const int view_num) const
{
  if (this->ring_radius.get_min_index()==0 && this->ring_radius.get_max_index()==0)
    return ring_radius[0];
  else
    return ring_radius[view_num];
}

float
ProjDataInfoBlocksOnCylindrical::get_ring_spacing() const
{ return ring_spacing;}

int
ProjDataInfoBlocksOnCylindrical::
get_view_mashing_factor() const
{
  // KT 10/05/2002 new assert
  assert(get_scanner_ptr()->get_num_detectors_per_ring() > 0);
  // KT 10/05/2002 moved assert here from constructor
  assert(get_scanner_ptr()->get_num_detectors_per_ring() % (2*get_num_views()) == 0);
  // KT 28/11/2001 do not pre-store anymore as set_num_views would invalidate it
  return get_scanner_ptr()->get_num_detectors_per_ring() / (2*get_num_views());
}

Succeeded
ProjDataInfoBlocksOnCylindrical::
get_segment_num_for_ring_difference(int& segment_num, const int ring_diff) const
{
  if (!sampling_corresponds_to_physical_rings)
    return Succeeded::no;

  // check currently necessary as reduce_segment does not reduce the size of the ring_diff arrays
  if (ring_diff > get_max_ring_difference(get_max_segment_num()) ||
      ring_diff < get_min_ring_difference(get_min_segment_num()))
    return Succeeded::no;

  if (!ring_diff_arrays_computed)
    initialise_ring_diff_arrays();

  segment_num = ring_diff_to_segment_num[ring_diff];
  // warning: relies on initialise_ring_diff_arrays to set invalid ring_diff to a too large segment_num
  if (segment_num <= get_max_segment_num())
    return Succeeded::yes;
  else
    return Succeeded::no;
}


Succeeded
ProjDataInfoBlocksOnCylindrical::
get_segment_axial_pos_num_for_ring_pair(int& segment_num,
                                        int& ax_pos_num,
                                        const int ring1,
                                        const int ring2) const
{
  assert(0<=ring1);
  assert(ring1<get_scanner_ptr()->get_num_rings());
  assert(0<=ring2);
  assert(ring2<get_scanner_ptr()->get_num_rings());

  // KT 01/08/2002 swapped rings
  if (get_segment_num_for_ring_difference(segment_num, ring2-ring1) == Succeeded::no)
    return Succeeded::no;

  // see initialise_ring_diff_arrays() for some info
  ax_pos_num = (ring1 + ring2 - ax_pos_num_offset[segment_num])*
               get_num_axial_poss_per_ring_inc(segment_num)/2;
  return Succeeded::yes;
}

const ProjDataInfoBlocksOnCylindrical::RingNumPairs&
ProjDataInfoBlocksOnCylindrical::
get_all_ring_pairs_for_segment_axial_pos_num(const int segment_num,
					     const int axial_pos_num) const
{
  if (!ring_diff_arrays_computed)
    initialise_ring_diff_arrays();
  if (is_null_ptr(segment_axial_pos_to_ring_pair[segment_num][axial_pos_num]))
    compute_segment_axial_pos_to_ring_pair(segment_num, axial_pos_num);
  return *segment_axial_pos_to_ring_pair[segment_num][axial_pos_num];
}

unsigned
ProjDataInfoBlocksOnCylindrical::
get_num_ring_pairs_for_segment_axial_pos_num(const int segment_num,
					     const int axial_pos_num) const
{
  return
    static_cast<unsigned>(
       this->get_all_ring_pairs_for_segment_axial_pos_num(segment_num,axial_pos_num).size());
}

END_NAMESPACE_STIR
