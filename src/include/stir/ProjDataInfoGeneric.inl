/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2003, Hammersmith Imanet Ltd
    Copyright (C) 2013, University College London
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2017, ETH Zurich, Institute of Particle Physics and Astrophysics
    Copyright (C) 2018, 2021 University of Leeds

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

/*!

  \file
  \ingroup projdata

  \brief Implementation of inline functions of class stir::ProjDataInfoGeneric
	
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Palak Wadhwa
  \author Berta Marti Fuster
  \author PARAPET project
  \author Parisa Khateri
  \author Michael Roethlisberger
  \author Viet Ahn Dao
*/

// for sqrt
#include <math.h>
#include "stir/Bin.h"
#include "stir/Succeeded.h"
#include "stir/LORCoordinates.h"
#include "stir/is_null_ptr.h"
#include <algorithm>
#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR


//! find phi from correspoding lor
float
ProjDataInfoGeneric::get_phi(const Bin& bin)const
{
	LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
	get_LOR(lor, bin);
	return lor.phi();
}

/*! warning In generic geometry m is calculated directly from lor while in
	cylindrical geometry m is calculated using m_offset and axial_sampling
*/
float
ProjDataInfoGeneric::get_m(const Bin& bin) const
{
	LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
	get_LOR(lor, bin);
	return (static_cast<float>(lor.z1() + lor.z2()))/2.F;
}

float
ProjDataInfoGeneric::get_t(const Bin& bin) const
{
  return
    get_m(bin)*get_costheta(bin);
}

/*
	theta is copolar angle of normal to projection plane with z axis, i.e. copolar angle of lor with z axis.
	tan (theta) = dz/sqrt(dx2+dy2)
	cylindrical geometry:
		delta_z = delta_ring * ring spacing
	generic geometry:
		delta_z is calculated from lor
*/
float
ProjDataInfoGeneric::get_tantheta(const Bin& bin) const
{
  CartesianCoordinate3D<float> _p1;
  CartesianCoordinate3D<float> _p2;
  find_cartesian_coordinates_of_detection(_p1, _p2, bin);
  CartesianCoordinate3D<float> p2_minus_p1 = _p2 - _p1;
  return p2_minus_p1.z() / (sqrt(square(p2_minus_p1.x())+square(p2_minus_p1.y()))); 
}

float
ProjDataInfoGeneric::get_sampling_in_m(const Bin& bin) const
{
  // TODOBLOCK
  return get_scanner_ptr()->get_ring_spacing(); // TODO currently restricted to span=1 get_num_axial_poss_per_ring_inc(segment_num);
  //return get_axial_sampling(bin.segment_num());
}

float
ProjDataInfoGeneric::get_sampling_in_t(const Bin& bin) const
{
  // TODOBLOCK
  return get_scanner_ptr()->get_ring_spacing()*get_costheta(bin); // TODO currently restricted to span=1 get_num_axial_poss_per_ring_inc(segment_num);
  //return get_axial_sampling(bin.segment_num())*get_costheta(bin);
}

float
ProjDataInfoGeneric::get_axial_sampling(int segment_num) const
{
  // TODOBLOCK should check sampling
  return get_ring_spacing(); // TODO currently restricted to span=1 get_num_axial_poss_per_ring_inc(segment_num);
}

bool ProjDataInfoGeneric::axial_sampling_is_uniform() const
{
  // TODOBLOCK should check sampling
  return true;
}

float
ProjDataInfoGeneric::get_ring_spacing() const
{ return get_scanner_ptr()->get_ring_spacing();}


END_NAMESPACE_STIR
