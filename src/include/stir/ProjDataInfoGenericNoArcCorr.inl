/*
    Copyright (C) 2000- 2005, Hammersmith Imanet Ltd
    Copyright (C) 2017, ETH Zurich, Institute of Particle Physics and Astrophysics
    Copyright (C) 2021, University of Leeds
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!

  \file
  \ingroup projdata

  \brief Implementation of inline functions of class stir::ProjDataInfoGenericNoArcCorr
	
  \author Kris Thielemans
  \author Parisa Khateri
  \author Michael Roethlisberger
  \author Viet Ahn Dao
*/


#include "stir/Bin.h"
#include "stir/Succeeded.h"
#include "stir/LORCoordinates.h"
#include <math.h>

START_NAMESPACE_STIR

void
ProjDataInfoGenericNoArcCorr::
initialise_uncompressed_view_tangpos_to_det1det2_if_not_done_yet() const
{
  // for efficiency reasons, use "Double-Checked-Locking(DCL) pattern" with OpenMP atomic operation
  // OpenMP v3.1 or later required
  // thanks to yohjp: http://stackoverflow.com/questions/27975737/how-to-handle-cached-data-structures-with-multi-threading-e-g-openmp
#if defined(STIR_OPENMP) &&  _OPENMP >=201012
  bool initialised;
#pragma omp atomic read
  initialised = uncompressed_view_tangpos_to_det1det2_initialised;

  if (!initialised)
#endif
    {
#if defined(STIR_OPENMP)
#pragma omp critical(PROJDATAINFOCYLINDRICALNOARCCORR_VIEWTANGPOS_TO_DETS)
#endif
          {
            if (!uncompressed_view_tangpos_to_det1det2_initialised)
              initialise_uncompressed_view_tangpos_to_det1det2();
          }
    }
}

void
ProjDataInfoGenericNoArcCorr::
initialise_det1det2_to_uncompressed_view_tangpos_if_not_done_yet() const
{
  // as above
#if defined(STIR_OPENMP) &&  _OPENMP >=201012
  bool initialised;
#pragma omp atomic read
  initialised = det1det2_to_uncompressed_view_tangpos_initialised;

  if (!initialised)
#endif
    {
#if defined(STIR_OPENMP)
#pragma omp critical(PROJDATAINFOCYLINDRICALNOARCCORR_DETS_TO_VIEWTANGPOS)
#endif
          {
            if (!det1det2_to_uncompressed_view_tangpos_initialised)
              initialise_det1det2_to_uncompressed_view_tangpos();
          }
    }
}

/*! warning In cylindrical s is found from bin: sin(beta) = sin(tang_pos*angular_increment)
	In block it is calculated directly from corresponding lor
*/
float
ProjDataInfoGenericNoArcCorr::
get_s(const Bin& bin) const
{
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
	get_LOR(lor, bin);
  // REQUIREMENT: Euclidean coordinate of 3 points, a,b and c.
  // CALCULATION: // Equation of a line, in parametric form, given two point a, b: p(t) = a + (b-a)*t
  //              // Let a,b,c be points in 2D and let a,b form a line then shortest distance between c and line ab is:
  //              // || (p0-p1) - [(p0-p1)*(p2-p1)/||p2-p1||^2]*(p2-p1) ||
  //              p1 = _p1, p2=_p2 and p0=(0,0,0)
  // OUTPUT:      replace the ring_radius*sin(lor.beta()) with distance from o to ab.

  // Can't access coordinate of detection from this class so we have to recalculate where it is:
  
  CartesianCoordinate3D<float> _p1;
	CartesianCoordinate3D<float> _p2;
	find_cartesian_coordinates_of_detection(_p1, _p2, bin);

  // // Get p1-p0 and p2-p1 vector.
  CartesianCoordinate3D<float> p2_minus_p1 = _p2 - _p1;
  CartesianCoordinate3D<float> p0_minus_p1 = CartesianCoordinate3D<float>(0,0,0) - _p1;
  float p0_minus_p1_dot_p2_minus_p1 = p0_minus_p1.x()* p2_minus_p1.x() + p0_minus_p1.y()*p2_minus_p1.y();
  float p2_minus_p1_magitude = square(p2_minus_p1.x()) + square(p2_minus_p1.y());
  float x = 0;
  float y = 0;
  float sign = sin(lor.beta());
  if (p2_minus_p1_magitude > 0.01)
  {
    x = p0_minus_p1.x() - (p0_minus_p1_dot_p2_minus_p1/p2_minus_p1_magitude)*p2_minus_p1.x();
    y = p0_minus_p1.y() - (p0_minus_p1_dot_p2_minus_p1/p2_minus_p1_magitude)*p2_minus_p1.y();
  }
  else
  {
     error("get_s(): 2 detection points are too close to each other. This indicates an internal error.");
  }
  float s = sqrt(square(x) + square(y));

  if(sign < 0.0){
    return -s;
  }else{
    return s;
  }
}

void
ProjDataInfoGenericNoArcCorr::
get_det_num_pair_for_view_tangential_pos_num(
					     int& det1_num,
					     int& det2_num,
					     const int view_num,
					     const int tang_pos_num) const
{
  assert(get_view_mashing_factor() == 1);
	this->initialise_uncompressed_view_tangpos_to_det1det2_if_not_done_yet();

  det1_num = uncompressed_view_tangpos_to_det1det2[view_num][tang_pos_num].det1_num;
  det2_num = uncompressed_view_tangpos_to_det1det2[view_num][tang_pos_num].det2_num;
}

bool
ProjDataInfoGenericNoArcCorr::
get_view_tangential_pos_num_for_det_num_pair(int& view_num,
					     int& tang_pos_num,
					     const int det1_num,
					     const int det2_num) const
{
	assert(det1_num!=det2_num);
	this->initialise_det1det2_to_uncompressed_view_tangpos_if_not_done_yet();

  view_num =
    det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].view_num/get_view_mashing_factor();
  tang_pos_num =
    det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].tang_pos_num;
  return
    det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].swap_detectors;
}

Succeeded
ProjDataInfoGenericNoArcCorr::
get_bin_for_det_pair(Bin& bin,
		     const int det_num1, const int ring_num1,
		     const int det_num2, const int ring_num2) const
{
  if (det_num1 == det_num2)  // this scenario is undefined: which view should we use here?
    return Succeeded::no;
  if (get_view_tangential_pos_num_for_det_num_pair(bin.view_num(), bin.tangential_pos_num(), det_num1, det_num2))
    return get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), ring_num1, ring_num2);
  else
    return get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), ring_num2, ring_num1);
}

Succeeded
ProjDataInfoGenericNoArcCorr::
get_bin_for_det_pos_pair(Bin& bin,
                         const DetectionPositionPair<>& dp) const
{
  return
    get_bin_for_det_pair(bin,
                         dp.pos1().tangential_coord(),
                         dp.pos1().axial_coord(),
		         		 dp.pos2().tangential_coord(),
                         dp.pos2().axial_coord());
}

void
ProjDataInfoGenericNoArcCorr::
get_det_pair_for_bin(
		     int& det_num1, int& ring_num1,
		     int& det_num2, int& ring_num2,
		     const Bin& bin) const
{
  get_det_num_pair_for_view_tangential_pos_num(det_num1, det_num2, bin.view_num(), bin.tangential_pos_num());
  get_ring_pair_for_segment_axial_pos_num( ring_num1, ring_num2, bin.segment_num(), bin.axial_pos_num());
}

void
ProjDataInfoGenericNoArcCorr::
get_det_pos_pair_for_bin(
		     DetectionPositionPair<>& dp,
		     const Bin& bin) const
{
  //lousy work around because types don't match TODO remove!
#if 1
  int t1=dp.pos1().tangential_coord(),
    a1=dp.pos1().axial_coord(),
    t2=dp.pos2().tangential_coord(),
    a2=dp.pos2().axial_coord();
  get_det_pair_for_bin(t1, a1, t2, a2, bin);
  dp.pos1().tangential_coord()=t1;
  dp.pos1().axial_coord()=a1;
  dp.pos2().tangential_coord()=t2;
  dp.pos2().axial_coord()=a2;

#else

  get_det_pair_for_bin(dp.pos1().tangential_coord(),
                       dp.pos1().axial_coord(),
		       dp.pos2().tangential_coord(),
                       dp.pos2().axial_coord(),
                       bin);
#endif
}

END_NAMESPACE_STIR
