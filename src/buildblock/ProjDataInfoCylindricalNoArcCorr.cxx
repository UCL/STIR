//
//
/*
    Copyright (C) 2000- 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata

  \brief Implementation of non-inline functions of class 
  stir::ProjDataInfoCylindricalNoArcCorr

  \author Kris Thielemans

*/

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/LORCoordinates.h"
#include "stir/round.h"

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
  uncompressed_view_tangpos_to_det1det2_initialised = false;
  det1det2_to_uncompressed_view_tangpos_initialised = false;
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
  assert(!is_null_ptr(scanner_ptr));
  ring_radius = scanner_ptr->get_effective_ring_radius();
  angular_increment = static_cast<float>(_PI/scanner_ptr->get_num_detectors_per_ring());
  uncompressed_view_tangpos_to_det1det2_initialised = false;
  det1det2_to_uncompressed_view_tangpos_initialised = false;
}




ProjDataInfo*
ProjDataInfoCylindricalNoArcCorr::clone() const
{
  return static_cast<ProjDataInfo*>(new ProjDataInfoCylindricalNoArcCorr(*this));
}

bool
ProjDataInfoCylindricalNoArcCorr::
operator==(const self_type& that) const
{
  if (!base_type::blindly_equals(&that))
    return false;
  return
    this->ring_radius == that.ring_radius &&
    this->angular_increment == that.angular_increment;
}

bool
ProjDataInfoCylindricalNoArcCorr::
blindly_equals(const root_type * const that_ptr) const
{
  assert(dynamic_cast<const self_type * const>(that_ptr) != 0);
  return
    this->operator==(static_cast<const self_type&>(*that_ptr));
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

/*!
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
initialise_uncompressed_view_tangpos_to_det1det2() const
{
  assert(-1 >> 1 == -1);
  assert(-2 >> 1 == -1);

  const int num_detectors =
    get_scanner_ptr()->get_num_detectors_per_ring();

  assert(num_detectors%2 == 0);
  // check views range from 0 to Pi
  assert(fabs(get_phi(Bin(0,0,0,0))) < 1.E-4);
  assert(fabs(get_phi(Bin(0,get_num_views(),0,0)) - _PI) < 1.E-4);
  const int min_tang_pos_num = -(num_detectors/2)+1;
  const int max_tang_pos_num = -(num_detectors/2)+num_detectors;
  
  if (this->get_min_tangential_pos_num() < min_tang_pos_num ||
      this->get_max_tangential_pos_num() > max_tang_pos_num)
    {
      error("The tangential_pos range (%d to %d) for this projection data is too large.\n"
	    "Maximum supported range is from %d to %d",  
	    this->get_min_tangential_pos_num(), this->get_max_tangential_pos_num(),
	    min_tang_pos_num, max_tang_pos_num);
    }

  uncompressed_view_tangpos_to_det1det2.grow(0,num_detectors/2-1);
  for (int v_num=0; v_num<=num_detectors/2-1; ++v_num)
  {
    uncompressed_view_tangpos_to_det1det2[v_num].grow(min_tang_pos_num, max_tang_pos_num);

    for (int tp_num=min_tang_pos_num; tp_num<=max_tang_pos_num; ++tp_num)
    {
      /*
         adapted from CTI code
         Note for implementation: avoid using % with negative numbers
         so add num_detectors before doing modulo num_detectors)
        */
      uncompressed_view_tangpos_to_det1det2[v_num][tp_num].det1_num = 
        (v_num + (tp_num >> 1) + num_detectors) % num_detectors;
      uncompressed_view_tangpos_to_det1det2[v_num][tp_num].det2_num = 
        (v_num - ( (tp_num + 1) >> 1 ) + num_detectors/2) % num_detectors;
    }
  }
  uncompressed_view_tangpos_to_det1det2_initialised = true;
}

void 
ProjDataInfoCylindricalNoArcCorr::
initialise_det1det2_to_uncompressed_view_tangpos() const
{
  assert(-1 >> 1 == -1);
  assert(-2 >> 1 == -1);

  const int num_detectors =
    get_scanner_ptr()->get_num_detectors_per_ring();

  if (num_detectors%2 != 0)
    {
      error("Number of detectors per ring should be even but is %d", num_detectors);
    }
  if (this->get_min_view_num() != 0)
    {
      error("Minimum view number should currently be zero to be able to use get_view_tangential_pos_num_for_det_num_pair()");
    }
  // check views range from 0 to Pi
  assert(fabs(get_phi(Bin(0,0,0,0))) < 1.E-4);
  assert(fabs(get_phi(Bin(0,get_max_view_num()+1,0,0)) - _PI) < 1.E-4);
  //const int min_tang_pos_num = -(num_detectors/2);
  //const int max_tang_pos_num = -(num_detectors/2)+num_detectors;
  const int max_num_views = num_detectors/2;

  det1det2_to_uncompressed_view_tangpos.grow(0,num_detectors-1);
  for (int det1_num=0; det1_num<num_detectors; ++det1_num)
  {
    det1det2_to_uncompressed_view_tangpos[det1_num].grow(0, num_detectors-1);

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
      
      det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].view_num = view_num;
      det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].tang_pos_num = tang_pos_num;
      det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].swap_detectors = swap_detectors==0;     
    }
  }
  det1det2_to_uncompressed_view_tangpos_initialised = true;
}

unsigned int
ProjDataInfoCylindricalNoArcCorr::
get_num_det_pos_pairs_for_bin(const Bin& bin) const
{
  return
    get_num_ring_pairs_for_segment_axial_pos_num(bin.segment_num(),
						 bin.axial_pos_num())*
    get_view_mashing_factor();
}

void
ProjDataInfoCylindricalNoArcCorr::
get_all_det_pos_pairs_for_bin(vector<DetectionPositionPair<> >& dps,
			      const Bin& bin) const
{
  if (!uncompressed_view_tangpos_to_det1det2_initialised)
    initialise_uncompressed_view_tangpos_to_det1det2();

  dps.resize(get_num_det_pos_pairs_for_bin(bin));

  const ProjDataInfoCylindrical::RingNumPairs& ring_pairs =
    get_all_ring_pairs_for_segment_axial_pos_num(bin.segment_num(),
						 bin.axial_pos_num());
  // not sure how to handle mashing with non-zero view offset...
  assert(get_min_view_num()==0);

  unsigned int current_dp_num=0;
  for (int uncompressed_view_num=bin.view_num()*get_view_mashing_factor();
       uncompressed_view_num<(bin.view_num()+1)*get_view_mashing_factor();
       ++uncompressed_view_num)
    {
      const int det1_num =
	uncompressed_view_tangpos_to_det1det2[uncompressed_view_num][bin.tangential_pos_num()].det1_num;
      const int det2_num = 
	uncompressed_view_tangpos_to_det1det2[uncompressed_view_num][bin.tangential_pos_num()].det2_num;
      for (ProjDataInfoCylindrical::RingNumPairs::const_iterator rings_iter = ring_pairs.begin();
	   rings_iter != ring_pairs.end();
	   ++rings_iter)
	{
	  assert(current_dp_num < get_num_det_pos_pairs_for_bin(bin));
	  dps[current_dp_num].pos1().tangential_coord() = det1_num;     
	  dps[current_dp_num].pos1().axial_coord() = rings_iter->first;
	  dps[current_dp_num].pos2().tangential_coord() = det2_num;     
	  dps[current_dp_num].pos2().axial_coord() = rings_iter->second;
	  ++current_dp_num;
	}
    }
  assert(current_dp_num == get_num_det_pos_pairs_for_bin(bin));
}

Succeeded
ProjDataInfoCylindricalNoArcCorr::
find_scanner_coordinates_given_cartesian_coordinates(int& det1, int& det2, int& ring1, int& ring2,
					             const CartesianCoordinate3D<float>& c1,
						     const CartesianCoordinate3D<float>& c2) const
{
  const int num_detectors=get_scanner_ptr()->get_num_detectors_per_ring();
  const float ring_spacing=get_scanner_ptr()->get_ring_spacing();
  const float ring_radius=get_scanner_ptr()->get_effective_ring_radius();

#if 0
  const CartesianCoordinate3D<float> d = c2 - c1;
  /* parametrisation of LOR is 
     c = l*d+c1
     l has to be such that c.x^2 + c.y^2 = R^2
     i.e.
     (l*d.x+c1.x)^2+(l*d.y+c1.y)^2==R^2
     l^2*(d.x^2+d.y^2) + 2*l*(d.x*c1.x + d.y*c1.y) + c1.x^2+c2.y^2-R^2==0
     write as a*l^2+2*b*l+e==0
     l = (-b +- sqrt(b^2-a*e))/a
     argument of sqrt simplifies to
     R^2*(d.x^2+d.y^2)-(d.x*c1.y-d.y*c1.x)^2
  */
  const float dxy2 = (square(d.x())+square(d.y()));
  assert(dxy2>0); // otherwise parallel to z-axis, which is gives ill-defined bin-coordinates
  const float argsqrt=
    (square(ring_radius)*dxy2-square(d.x()*c1.y()-d.y()*c1.x()));
  if (argsqrt<=0)
    return Succeeded::no; // LOR is outside detector radius
  const float root = sqrt(argsqrt);

  const float l1 = (- (d.x()*c1.x() + d.y()*c1.y())+root)/dxy2;
  const float l2 = (- (d.x()*c1.x() + d.y()*c1.y())-root)/dxy2;
  const CartesianCoordinate3D<float> coord_det1 = d*l1 + c1;
  const CartesianCoordinate3D<float> coord_det2 = d*l2 + c1;
  assert(fabs(square(coord_det1.x())+square(coord_det1.y())-square(ring_radius))<square(ring_radius)*10.E-5);
  assert(fabs(square(coord_det2.x())+square(coord_det2.y())-square(ring_radius))<square(ring_radius)*10.E-5);

  det1 = modulo(round(atan2(coord_det1.x(),-coord_det1.y())/(2.*_PI/num_detectors)), num_detectors);
  det2 = modulo(round(atan2(coord_det2.x(),-coord_det2.y())/(2.*_PI/num_detectors)), num_detectors);
  ring1 = round(coord_det1.z()/ring_spacing);
  ring2 = round(coord_det2.z()/ring_spacing);
#else
  LORInCylinderCoordinates<float> cyl_coords;
  if (find_LOR_intersections_with_cylinder(cyl_coords,
					   LORAs2Points<float>(c1, c2),
					   ring_radius)
      == Succeeded::no)
    return Succeeded::no;

  det1 = modulo(round(cyl_coords.p1().psi()/(2.*_PI/num_detectors)), num_detectors);
  det2 = modulo(round(cyl_coords.p2().psi()/(2.*_PI/num_detectors)), num_detectors);
  ring1 = round(cyl_coords.p1().z()/ring_spacing);
  ring2 = round(cyl_coords.p2().z()/ring_spacing);

#endif

  assert(det1 >=0 && det1<get_scanner_ptr()->get_num_detectors_per_ring());
  assert(det2 >=0 && det2<get_scanner_ptr()->get_num_detectors_per_ring());

  return 
    (ring1 >=0 && ring1<get_scanner_ptr()->get_num_rings() &&
     ring2 >=0 && ring2<get_scanner_ptr()->get_num_rings()) 
     ? Succeeded::yes : Succeeded::no;
}


void 
ProjDataInfoCylindricalNoArcCorr::
find_cartesian_coordinates_of_detection(
					CartesianCoordinate3D<float>& coord_1,
					CartesianCoordinate3D<float>& coord_2,
					const Bin& bin) const
{
 // find detectors
  int det_num_a;
  int det_num_b;
  int ring_a;
  int ring_b;
  get_det_pair_for_bin(det_num_a, ring_a,
                       det_num_b, ring_b, bin);
  
  // find corresponding cartesian coordinates
  find_cartesian_coordinates_given_scanner_coordinates(coord_1,coord_2,
    ring_a,ring_b,det_num_a,det_num_b);
}


void
ProjDataInfoCylindricalNoArcCorr::
find_cartesian_coordinates_given_scanner_coordinates (CartesianCoordinate3D<float>& coord_1,
				 CartesianCoordinate3D<float>& coord_2,
				 const int Ring_A,const int Ring_B, 
				 const int det1, const int det2) const
{
  const int num_detectors_per_ring = 
    get_scanner_ptr()->get_num_detectors_per_ring();

#if 0
  const float df1 = (2.*_PI/num_detectors_per_ring)*(det1);
  const float df2 = (2.*_PI/num_detectors_per_ring)*(det2);
  const float x1 = get_scanner_ptr()->get_effective_ring_radius()*cos(df1);
  const float y1 = get_scanner_ptr()->get_effective_ring_radius()*sin(df1);
  const float x2 = get_scanner_ptr()->get_effective_ring_radius()*cos(df2);
  const float y2 = get_scanner_ptr()->get_effective_ring_radius()*sin(df2);
  const float z1 = Ring_A*get_scanner_ptr()->get_ring_spacing();
  const float z2 = Ring_B*get_scanner_ptr()->get_ring_spacing();
  // make sure the return values are in STIR coordinates
  coord_1.z() = z1;
  coord_1.y() = -x1;
  coord_1.x() = y1;

  coord_2.z() = z2;
  coord_2.y() = -x2;
  coord_2.x() = y2; 
#else
  // although code maybe doesn't really need the following, 
  // asserts in the LOR code will break if these conditions are not satisfied.
  assert(0<=det1);
  assert(det1<num_detectors_per_ring);
  assert(0<=det2);
  assert(det2<num_detectors_per_ring);

  LORInCylinderCoordinates<float> cyl_coords(get_scanner_ptr()->get_effective_ring_radius());
  cyl_coords.p1().psi() = static_cast<float>((2.*_PI/num_detectors_per_ring)*(det1));
  cyl_coords.p2().psi() = static_cast<float>((2.*_PI/num_detectors_per_ring)*(det2));
  cyl_coords.p1().z() = Ring_A*get_scanner_ptr()->get_ring_spacing();
  cyl_coords.p2().z() = Ring_B*get_scanner_ptr()->get_ring_spacing();
  LORAs2Points<float> lor(cyl_coords);  
  coord_1 = lor.p1();
  coord_2 = lor.p2();
  
#endif
}


void 
ProjDataInfoCylindricalNoArcCorr::
find_bin_given_cartesian_coordinates_of_detection(Bin& bin,
						  const CartesianCoordinate3D<float>& coord_1,
						  const CartesianCoordinate3D<float>& coord_2) const
{
  int det_num_a;
  int det_num_b;
  int ring_a;
  int ring_b;
  
  // given two CartesianCoordinates find the intersection     
  if (find_scanner_coordinates_given_cartesian_coordinates(det_num_a,det_num_b,
							   ring_a, ring_b,
							   coord_1,
							   coord_2) ==
      Succeeded::no)
  {
    bin.set_bin_value(-1);
    return;
  }

  // check rings are in valid range
  // this should have been done by find_scanner_coordinates_given_cartesian_coordinates
  assert(!(ring_a<0 ||
	   ring_a>=get_scanner_ptr()->get_num_rings() ||
	   ring_b<0 ||
	   ring_b>=get_scanner_ptr()->get_num_rings()));

  if (get_bin_for_det_pair(bin,
			   det_num_a, ring_a,
			   det_num_b, ring_b) == Succeeded::no ||
      bin.tangential_pos_num() < get_min_tangential_pos_num() ||
      bin.tangential_pos_num() > get_max_tangential_pos_num())
    bin.set_bin_value(-1);
}

Bin
ProjDataInfoCylindricalNoArcCorr::
get_bin(const LOR<float>& lor) const
{
  Bin bin;
#ifndef STIR_DEVEL
  // find nearest bin by going to nearest detectors first
  LORInCylinderCoordinates<float> cyl_coords;
  if (lor.change_representation(cyl_coords, get_ring_radius()) == Succeeded::no)
    {
      bin.set_bin_value(-1);
      return bin;
    }
  const int num_detectors_per_ring = 
    get_scanner_ptr()->get_num_detectors_per_ring();
  const int num_rings = 
    get_scanner_ptr()->get_num_rings();

  const int det1 = modulo(round(cyl_coords.p1().psi()/(2.*_PI/num_detectors_per_ring)),num_detectors_per_ring);
  const int det2 = modulo(round(cyl_coords.p2().psi()/(2.*_PI/num_detectors_per_ring)),num_detectors_per_ring);
  // TODO WARNING LOR coordinates are w.r.t. centre of scanner, but the rings are numbered with the first ring at 0
  const int ring1 = round(cyl_coords.p1().z()/get_ring_spacing() + (num_rings-1)/2.F);
  const int ring2 = round(cyl_coords.p2().z()/get_ring_spacing() + (num_rings-1)/2.F);

  assert(det1 >=0 && det1<num_detectors_per_ring);
  assert(det2 >=0 && det2<num_detectors_per_ring);

  if (ring1 >=0 && ring1<num_rings &&
      ring2 >=0 && ring2<num_rings &&
      get_bin_for_det_pair(bin,
			   det1, ring1, det2, ring2) == Succeeded::yes &&
      bin.tangential_pos_num() >= get_min_tangential_pos_num() &&
      bin.tangential_pos_num() <= get_max_tangential_pos_num())
    {
      bin.set_bin_value(1);
      return bin;
    }
  else
    {
      bin.set_bin_value(-1);
      return bin;
    }


#else
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor_coords;
  if (lor.change_representation(lor_coords, get_ring_radius()) == Succeeded::no)
    {
      bin.set_bin_value(-1);
      return bin;
    }

  // first find view 
  // unfortunately, phi ranges from [0,Pi[, but the rounding can
  // map this to a view which corresponds to Pi anyway.
  bin.view_num() = round(lor_coords.phi() / get_azimuthal_angle_sampling());
  assert(bin.view_num()>=0);
  assert(bin.view_num()<=get_num_views());
  const bool swap_direction =
    bin.view_num() > get_max_view_num();
  if (swap_direction)
    bin.view_num()-=get_num_views();

  bin.tangential_pos_num() = round(lor_coords.beta() / angular_increment);
  if (swap_direction)
    bin.tangential_pos_num() *= -1;

  if (bin.tangential_pos_num() < get_min_tangential_pos_num() ||
      bin.tangential_pos_num() > get_max_tangential_pos_num())
    {
      bin.set_bin_value(-1);
      return bin;
    }

#if 0
  const int num_rings = 
    get_scanner_ptr()->get_num_rings();
  // TODO WARNING LOR coordinates are w.r.t. centre of scanner, but the rings are numbered with the first ring at 0
  int ring1, ring2;
  if (!swap_direction)
    {
      ring1 = round(lor_coords.z1()/get_ring_spacing() + (num_rings-1)/2.F);
      ring2 = round(lor_coords.z2()/get_ring_spacing() + (num_rings-1)/2.F);
    }
  else
    {
      ring2 = round(lor_coords.z1()/get_ring_spacing() + (num_rings-1)/2.F);
      ring1 = round(lor_coords.z2()/get_ring_spacing() + (num_rings-1)/2.F);
    }

  if (!(ring1 >=0 && ring1<get_scanner_ptr()->get_num_rings() &&
	ring2 >=0 && ring2<get_scanner_ptr()->get_num_rings() &&
	get_segment_axial_pos_num_for_ring_pair(bin.segment_num(),
						bin.axial_pos_num(),
						ring1,
						ring2) == Succeeded::yes)
      )
    {
      bin.set_bin_value(-1);
      return bin;
    }
#else
  // find nearest segment
  {
    const float delta =
      (swap_direction 
       ? lor_coords.z1()-lor_coords.z2()
       : lor_coords.z2()-lor_coords.z1()
       )/get_ring_spacing();
    // check if out of acquired range
    // note the +1 or -1, which takes the size of the rings into account
    if (delta>get_max_ring_difference(get_max_segment_num())+1 ||
	delta<get_min_ring_difference(get_min_segment_num())-1)
      {
	bin.set_bin_value(-1);
	return bin;
      } 
    if (delta>=0)
      {
	for (bin.segment_num()=0; bin.segment_num()<get_max_segment_num(); ++bin.segment_num())
	  {
	    if (delta < get_max_ring_difference(bin.segment_num())+.5)
	      break;
	  }
      }
    else
      {
	// delta<0
	for (bin.segment_num()=0; bin.segment_num()>get_min_segment_num(); --bin.segment_num())
	  {
	    if (delta > get_min_ring_difference(bin.segment_num())-.5)
	      break;
	  }
      }
  }
  // now find nearest axial position
  {
    const float m = (lor_coords.z2()+lor_coords.z1())/2;
#if 0
    // this uses private member of ProjDataInfoCylindrical
    // enable when moved
    if (!ring_diff_arrays_computed)
      initialise_ring_diff_arrays();
#ifndef NDEBUG
    bin.axial_pos_num()=0;
    assert(get_m(bin)==- m_offset[bin.segment_num()]);
#endif
    bin.axial_pos_num() =
      round((m + m_offset[bin.segment_num()])/
	    get_axial_sampling(bin.segment_num()));
#else
    bin.axial_pos_num()=0;
    bin.axial_pos_num() =
      round((m - get_m(bin))/
	    get_axial_sampling(bin.segment_num()));
#endif
    if (bin.axial_pos_num() < get_min_axial_pos_num(bin.segment_num()) ||
	bin.axial_pos_num() > get_max_axial_pos_num(bin.segment_num()))
      {
	bin.set_bin_value(-1);
	return bin;
      }
  }
#endif

  bin.set_bin_value(1);
  return bin;
#endif
}

 
END_NAMESPACE_STIR

