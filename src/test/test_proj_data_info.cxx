//
//
/*!

  \file
  \ingroup test

  \brief Test program for stir::ProjDataInfo hierarchy

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2018, Palak Wadhwa and University of Leeds
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

#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "stir/LORCoordinates.h"
#include "stir/round.h"
#include "stir/num_threads.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <math.h>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::setw;
using std::endl;
using std::min;
using std::max;
using std::size_t;
#endif

START_NAMESPACE_STIR

static inline
int intabs(const int x)
{ return x>=0?x:-x; }

// prints a michelogram to the screen
#if 0
// TODO move somewhere else
void michelogram(const ProjDataInfoCylindrical& proj_data_info)
{
  cerr << '{';
  for (int ring1=0; ring1<proj_data_info.get_scanner_ptr()->get_num_rings(); ++ring1)
  {
    cerr << '{';
    for (int ring2=0; ring2<proj_data_info.get_scanner_ptr()->get_num_rings(); ++ring2)
    {
      int segment_num=0;
      int ax_pos_num = 0;
      if (proj_data_info.get_segment_axial_pos_num_for_ring_pair(segment_num, ax_pos_num, ring1, ring2)
        == Succeeded::yes)
        cerr << '{' << setw(3) << segment_num << ',' << setw(2) << ax_pos_num << "}";
      else
        cerr << "{}      ";
      if (ring2 != proj_data_info.get_scanner_ptr()->get_num_rings()-1)
        cerr << ',';

    }
    cerr << '}';
    if (ring1 != proj_data_info.get_scanner_ptr()->get_num_rings()-1)
        cerr << ',';
    cerr << endl;
  }
  cerr << '}' << endl;
}
#endif

/*!
  \ingroup test
  \brief Test class for ProjDataInfo
*/
class ProjDataInfoTests: public RunTests
{
protected:  
  void test_generic_proj_data_info(ProjDataInfo& proj_data_info);
};

void
ProjDataInfoTests::
test_generic_proj_data_info(ProjDataInfo& proj_data_info)
{
  cerr << "\tTests on get_LOR/get_bin\n";
  int max_diff_segment_num=0;
  int max_diff_view_num=0;
  int max_diff_axial_pos_num=0;
  int max_diff_tangential_pos_num=0;
  float view_offset=proj_data_info.get_scanner_ptr()->get_default_intrinsic_tilt();
  if (view_offset==0)
  {
#ifdef STIR_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int segment_num=proj_data_info.get_min_segment_num();
       segment_num<=proj_data_info.get_max_segment_num();
	 ++segment_num)
      {
	for (int view_num=proj_data_info.get_min_view_num();
	     view_num<=proj_data_info.get_max_view_num();
	     view_num+=3)
	  {
	    // loop over axial_positions. Avoid using first and last positions, as 
	    // if there is axial compression, the central LOR of a bin might actually not
	    // fall within the scanner. In this case, the get_bin(get_LOR(org_bin)) code 
	    // will return an out-of-range bin (i.e. value<0).
	    int axial_pos_num_margin=0;
	    const ProjDataInfoCylindrical* const proj_data_info_cyl_ptr =
	      dynamic_cast<const ProjDataInfoCylindrical* const>(&proj_data_info);
	    if (proj_data_info_cyl_ptr!=0)
	      {
		axial_pos_num_margin =
		  std::max(
			   round(ceil(proj_data_info_cyl_ptr->get_average_ring_difference(segment_num) -
				      proj_data_info_cyl_ptr->get_min_ring_difference(segment_num))),
			   round(ceil(proj_data_info_cyl_ptr->get_max_ring_difference(segment_num) -
				      proj_data_info_cyl_ptr->get_average_ring_difference(segment_num))));		
	      }
	    for (int axial_pos_num=proj_data_info.get_min_axial_pos_num(segment_num)+axial_pos_num_margin ;
		 axial_pos_num<=proj_data_info.get_max_axial_pos_num(segment_num)-axial_pos_num_margin;
		 axial_pos_num+=3)
	      {
		for (int tangential_pos_num=proj_data_info.get_min_tangential_pos_num()+1;
		     tangential_pos_num<=proj_data_info.get_max_tangential_pos_num()-1;
		     tangential_pos_num+=1)
		  {
		    const Bin org_bin(segment_num,view_num,axial_pos_num,tangential_pos_num, /* value*/1);
		    LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
		    proj_data_info.get_LOR(lor, org_bin);
		    {
		      const Bin new_bin = proj_data_info.get_bin(lor);
#if 1
		      const int diff_segment_num =
			intabs(org_bin.segment_num() - new_bin.segment_num());
		      const int diff_view_num = 
			intabs(org_bin.view_num() - new_bin.view_num());
		      const int diff_axial_pos_num = 
			intabs(org_bin.axial_pos_num() - new_bin.axial_pos_num());
		      const int diff_tangential_pos_num = 
			intabs(org_bin.tangential_pos_num() - new_bin.tangential_pos_num());
		      if (new_bin.get_bin_value()>0)
			{
			  if (diff_segment_num>max_diff_segment_num)
			    max_diff_segment_num=diff_segment_num;
			  if (diff_view_num>max_diff_view_num)
			    max_diff_view_num=diff_view_num;
			  if (diff_axial_pos_num>max_diff_axial_pos_num)
			    max_diff_axial_pos_num=diff_axial_pos_num;
			  if (diff_tangential_pos_num>max_diff_tangential_pos_num)
			    max_diff_tangential_pos_num=diff_tangential_pos_num;
			}
		      if (!check(org_bin.get_bin_value() == new_bin.get_bin_value(), "round-trip get_LOR then get_bin: value") ||
			  !check(diff_segment_num<=0, "round-trip get_LOR then get_bin: segment") ||
			  !check(diff_view_num<=1, "round-trip get_LOR then get_bin: view") ||
			  !check(diff_axial_pos_num<=1, "round-trip get_LOR then get_bin: axial_pos") ||
			  !check(diff_tangential_pos_num<=1, "round-trip get_LOR then get_bin: tangential_pos"))

#else
		      if (!check(org_bin == new_bin, "round-trip get_LOR then get_bin"))
#endif
			{
			  cerr << "\tProblem at    segment = " << org_bin.segment_num() 
			       << ", axial pos " << org_bin.axial_pos_num()
			       << ", view = " << org_bin.view_num() 
			       << ", tangential_pos_num = " << org_bin.tangential_pos_num() << "\n";
			  if (new_bin.get_bin_value()>0)
			    cerr << "\tround-trip to segment = " << new_bin.segment_num() 
				 << ", axial pos " << new_bin.axial_pos_num()
				 << ", view = " << new_bin.view_num() 
				 << ", tangential_pos_num = " << new_bin.tangential_pos_num() 
				 <<'\n';
			}
		    }
		    // repeat test but with different type of LOR
		    {
		      LORAs2Points<float> lor_as_points;
		      lor.get_intersections_with_cylinder(lor_as_points, lor.radius());
		      const Bin new_bin = proj_data_info.get_bin(lor_as_points);
#if 1
		      const int diff_segment_num =
			intabs(org_bin.segment_num() - new_bin.segment_num());
		      const int diff_view_num = 
			  intabs(org_bin.view_num() - new_bin.view_num());
		      const int diff_axial_pos_num = 
			intabs(org_bin.axial_pos_num() - new_bin.axial_pos_num());
		      const int diff_tangential_pos_num = 
			intabs(org_bin.tangential_pos_num() - new_bin.tangential_pos_num());
		      if (new_bin.get_bin_value()>0)
			{
			  if (diff_segment_num>max_diff_segment_num)
			    max_diff_segment_num=diff_segment_num;
			  if (diff_view_num>max_diff_view_num)
			    max_diff_view_num=diff_view_num;
			  if (diff_axial_pos_num>max_diff_axial_pos_num)
			    max_diff_axial_pos_num=diff_axial_pos_num;
			  if (diff_tangential_pos_num>max_diff_tangential_pos_num)
			    max_diff_tangential_pos_num=diff_tangential_pos_num;
			}
		      if (!check(org_bin.get_bin_value() == new_bin.get_bin_value(), "round-trip get_LOR then get_bin (LORAs2Points): value") ||
			  !check(diff_segment_num<=0, "round-trip get_LOR then get_bin (LORAs2Points): segment") ||
			  !check(diff_view_num<=1, "round-trip get_LOR then get_bin (LORAs2Points): view") ||
			  !check(diff_axial_pos_num<=1, "round-trip get_LOR then get_bin (LORAs2Points): axial_pos") ||
			  !check(diff_tangential_pos_num<=1, "round-trip get_LOR then get_bin (LORAs2Points): tangential_pos"))
			
#else
		      if (!check(org_bin == new_bin, "round-trip get_LOR then get_bin"))
#endif
			{
			  cerr << "\tProblem at    segment = " << org_bin.segment_num() 
			       << ", axial pos " << org_bin.axial_pos_num()
			       << ", view = " << org_bin.view_num() 
			       << ", tangential_pos_num = " << org_bin.tangential_pos_num() << "\n";
			  if (new_bin.get_bin_value()>0)
			    cerr << "\tround-trip to segment = " << new_bin.segment_num() 
				 << ", axial pos " << new_bin.axial_pos_num()
				 << ", view = " << new_bin.view_num() 
				 << ", tangential_pos_num = " << new_bin.tangential_pos_num() 
				 <<'\n';
			}
		    }
		  }
	      }
	  }
      }
    cerr << "Max Deviation:  segment = " << max_diff_segment_num 
	 << ", axial pos " << max_diff_axial_pos_num
	 << ", view = " << max_diff_view_num 
	 << ", tangential_pos_num = " << max_diff_tangential_pos_num << "\n";
}
}

/*!
  \ingroup test
  \brief Test class for ProjDataInfoCylindrical
*/
class ProjDataInfoCylindricalTests: public ProjDataInfoTests
{
protected:  
  void test_cylindrical_proj_data_info(ProjDataInfoCylindrical& proj_data_info);
};


void
ProjDataInfoCylindricalTests::
test_cylindrical_proj_data_info(ProjDataInfoCylindrical& proj_data_info) 
{
  cerr << "\tTesting consistency between different implementations of geometric info\n";
  {
    const Bin bin(proj_data_info.get_max_segment_num(),
	    1,
	    proj_data_info.get_max_axial_pos_num(proj_data_info.get_max_segment_num())/2,
	    1);
    check_if_equal(proj_data_info.get_sampling_in_m(bin),
		   proj_data_info.ProjDataInfo::get_sampling_in_m(bin),
		   "test consistency get_sampling_in_m");
    check_if_equal(proj_data_info.get_sampling_in_t(bin),
		   proj_data_info.ProjDataInfo::get_sampling_in_t(bin),
		   "test consistency get_sampling_in_t");
#if 0
    // ProjDataInfo has no default implementation for get_tantheta
    // I just leave the code here to make this explicit
    check_if_equal(proj_data_info.get_tantheta(bin),
		   proj_data_info.ProjDataInfo::get_tantheta(bin),
		   "test consistency get_tantheta");
#endif
    check_if_equal(proj_data_info.get_costheta(bin),
		   proj_data_info.ProjDataInfo::get_costheta(bin),
		   "test consistency get_costheta");

    check_if_equal(proj_data_info.get_costheta(bin),
		   cos(atan(proj_data_info.get_tantheta(bin))),
		   "cross check get_costheta and get_tantheta");

    // try the same with a non-standard ring spacing
    const float old_ring_spacing = proj_data_info.get_ring_spacing();
    proj_data_info.set_ring_spacing(2.1F);

    check_if_equal(proj_data_info.get_sampling_in_m(bin),
		   proj_data_info.ProjDataInfo::get_sampling_in_m(bin),
		   "test consistency get_sampling_in_m");
    check_if_equal(proj_data_info.get_sampling_in_t(bin),
		   proj_data_info.ProjDataInfo::get_sampling_in_t(bin),
		   "test consistency get_sampling_in_t");
#if 0
    check_if_equal(proj_data_info.get_tantheta(bin),
		   proj_data_info.ProjDataInfo::get_tantheta(bin),
		   "test consistency get_tantheta");
#endif
    check_if_equal(proj_data_info.get_costheta(bin),
		   proj_data_info.ProjDataInfo::get_costheta(bin),
		   "test consistency get_costheta");

    check_if_equal(proj_data_info.get_costheta(bin),
		   cos(atan(proj_data_info.get_tantheta(bin))),
		   "cross check get_costheta and get_tantheta");
    // set back to usual value
    proj_data_info.set_ring_spacing(old_ring_spacing);
  }

  if (proj_data_info.get_max_ring_difference(0) == proj_data_info.get_min_ring_difference(0)
      && proj_data_info.get_max_ring_difference(1) == proj_data_info.get_min_ring_difference(1)
      && proj_data_info.get_max_ring_difference(2) == proj_data_info.get_min_ring_difference(2)
      )
    {
      // these tests work only without axial compression
      cerr << "\tTest ring pair to segment,ax_pos (span 1)\n";
#ifdef STIR_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int ring1=0; ring1<proj_data_info.get_scanner_ptr()->get_num_rings(); ++ring1)
      for (int ring2=0; ring2<proj_data_info.get_scanner_ptr()->get_num_rings(); ++ring2)
	{
	  int segment_num = 0, axial_pos_num = 0;
	  check(proj_data_info.
		get_segment_axial_pos_num_for_ring_pair(segment_num,
							axial_pos_num,
							ring1,
							ring2) ==
		    Succeeded::yes,
		"test if segment,ax_pos_num found for a ring pair");
	  check_if_equal(segment_num, ring2-ring1,
			 "test if segment_num is equal to ring difference\n");
	  check_if_equal(axial_pos_num, min(ring2,ring1),
			 "test if segment_num is equal to ring difference\n");
	  
	  int check_ring1 = 0, check_ring2 = 0;
	  proj_data_info.
	    get_ring_pair_for_segment_axial_pos_num(check_ring1,
						    check_ring2,
						    segment_num,
						    axial_pos_num);
	  check_if_equal(ring1, check_ring1,
			 "test ring1 equal after going to segment/ax_pos and returning\n");
	  check_if_equal(ring2, check_ring2,
			 "test ring2 equal after going to segment/ax_pos and returning\n");

	  const ProjDataInfoCylindrical::RingNumPairs& ring_pairs =
	    proj_data_info.
	    get_all_ring_pairs_for_segment_axial_pos_num(segment_num,
							 axial_pos_num);

	  check_if_equal(ring_pairs.size(), static_cast<size_t>(1),
			 "test total number of ring-pairs for 1 segment/ax_pos should be 1 for span=1\n");
	  check_if_equal(ring1, ring_pairs[0].first,
			 "test ring1 equal after going to segment/ax_pos and returning (version with all ring_pairs)\n");
	  check_if_equal(ring2, ring_pairs[0].second,
			 "test ring2 equal after going to segment/ax_pos and returning (version with all ring_pairs)\n");
	}
  }

  cerr << "\tTest ring pair to segment,ax_pos and vice versa (for any axial compression)\n";
  {
#ifdef STIR_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int segment_num=proj_data_info.get_min_segment_num(); 
	 segment_num<=proj_data_info.get_max_segment_num(); 
	 ++segment_num)
      for (int axial_pos_num=proj_data_info.get_min_axial_pos_num(segment_num); 
	   axial_pos_num<=proj_data_info.get_max_axial_pos_num(segment_num); 
	   ++axial_pos_num)
	{
	  const ProjDataInfoCylindrical::RingNumPairs& ring_pairs =
	    proj_data_info.
	    get_all_ring_pairs_for_segment_axial_pos_num(segment_num,
							 axial_pos_num);
	  for (ProjDataInfoCylindrical::RingNumPairs::const_iterator iter = ring_pairs.begin();
	       iter != ring_pairs.end();
	       ++iter)
	    {
	      int check_segment_num = 0, check_axial_pos_num = 0;
	      check(proj_data_info.
		    get_segment_axial_pos_num_for_ring_pair(check_segment_num,
							    check_axial_pos_num,
							    iter->first,
							    iter->second) ==
		    Succeeded::yes,
		    "test if segment,ax_pos_num found for a ring pair");
	      check_if_equal(check_segment_num, segment_num,
			     "test if segment_num is consistent\n");
	      check_if_equal(check_axial_pos_num, axial_pos_num,
			     "test if axial_pos_num is consistent\n");
	    }
	}	      
  }

  test_generic_proj_data_info(proj_data_info);
}

/*!
  \ingroup test
  \brief Test class for ProjDataInfoCylindricalArcCorr
*/

class ProjDataInfoCylindricalArcCorrTests: public ProjDataInfoCylindricalTests
{
public:  
  void run_tests();
};


void
ProjDataInfoCylindricalArcCorrTests::run_tests()

{ 
  cerr << "-------- Testing ProjDataInfoCylindricalArcCorr --------\n";
  {
    // Test on the empty constructor
    
    ProjDataInfoCylindricalArcCorr ob1;
    
    // Test on set.* & get.* + constructor
    const float test_tangential_sampling = 1.5;
    //const float test_azimuthal_angle_sampling = 10.1;
    
    ob1.set_tangential_sampling(test_tangential_sampling);
    // Set_azimuthal_angle_sampling
    // ob1.set_azimuthal_angle_sampling(test_azimuthal_angle_sampling);
    
    
    check_if_equal( ob1.get_tangential_sampling(), test_tangential_sampling,"test on tangential_sampling");
    //check_if_zero( ob1.get_azimuthal_angle_sampling() - test_azimuthal_angle_sampling, " test on azimuthal_angle_sampling");
    
  }
  {
    shared_ptr<Scanner> scanner_ptr(new Scanner(Scanner::E953));
    
    VectorWithOffset<int> num_axial_pos_per_segment(-1,1);
    VectorWithOffset<int> min_ring_diff(-1,1); 
    VectorWithOffset<int> max_ring_diff(-1,1);
    // simulate span=3 for segment 0, span=1 for segment 2
    num_axial_pos_per_segment[-1]=14;
    num_axial_pos_per_segment[0]=31;
    num_axial_pos_per_segment[1]=14;
    // KT 28/11/2001 corrected typo (bug): min_ring_diff[-1] was initialised twice, and max_ring_diff[-1] wasn't
    min_ring_diff[-1] = max_ring_diff[-1] = -2;
    min_ring_diff[ 0] = -1; max_ring_diff[ 0] = 1;
    min_ring_diff[+1] = max_ring_diff[+1] = +2;
    const int num_views = 96;
    const int num_tangential_poss = 128;
    
    const float bin_size = 1.2F;
    
    
    //Test on the constructor
    ProjDataInfoCylindricalArcCorr
      ob2(scanner_ptr, bin_size, 
      num_axial_pos_per_segment, min_ring_diff, max_ring_diff,
      num_views,num_tangential_poss);
    
    check_if_equal( ob2.get_tangential_sampling(), bin_size,"test on tangential_sampling");
    check_if_equal( ob2.get_azimuthal_angle_sampling() , _PI/num_views, " test on azimuthal_angle_sampling");
    check_if_equal( ob2.get_axial_sampling(1),  scanner_ptr->get_ring_spacing(), "test on axial_sampling");
    check_if_equal( ob2.get_axial_sampling(0), scanner_ptr->get_ring_spacing()/2, "test on axial_sampling for segment0");
    
    {
      // segment 0
      Bin bin(0,10,10,20);
      float theta = ob2.get_tantheta(bin);
      float phi = ob2.get_phi(bin); 
      // Get t
      float t = ob2.get_t(bin);
      //! Get s
      float s = ob2.get_s(bin);
      
      check_if_equal( theta, 0.F,"test on get_tantheta, seg 0");
      check_if_equal( phi, 10*ob2.get_azimuthal_angle_sampling()+scanner_ptr->get_default_intrinsic_tilt(), " get_phi , seg 0");
      // KT 25/10/2000 adjust to new convention
      const float ax_pos_origin =
	(ob2.get_min_axial_pos_num(0) + ob2.get_max_axial_pos_num(0))/2.F;
      check_if_equal( t, (10-ax_pos_origin)*ob2.get_axial_sampling(0) , "get_t, seg 0");
      check_if_equal( s, 20*ob2.get_tangential_sampling() , "get_s, seg 0");
    }
    {
      // Segment 1
      Bin bin (1,10,10,20);
      float theta = ob2.get_tantheta(bin);
      float phi = ob2.get_phi(bin); 
      // Get t
      float t = ob2.get_t(bin);
      // Get s
      float s = ob2.get_s(bin);
      
      float thetatest = 2*ob2.get_axial_sampling(1)/(2*sqrt(square(scanner_ptr->get_effective_ring_radius())-square(s)));
      
      check_if_equal( theta, thetatest,"test on get_tantheta, seg 1");
      check_if_equal( phi, 10*ob2.get_azimuthal_angle_sampling()+scanner_ptr->get_default_intrinsic_tilt(), " get_phi , seg 1");
      // KT 25/10/2000 adjust to new convention
      const float ax_pos_origin =
	(ob2.get_min_axial_pos_num(1) + ob2.get_max_axial_pos_num(1))/2.F;
      check_if_equal( t, (10-ax_pos_origin)/sqrt(1+square(thetatest))*ob2.get_axial_sampling(1) , "get_t, seg 1");
      check_if_equal( s, 20*ob2.get_tangential_sampling() , "get_s, seg 1");
    }

#if 0    
  // disabled to get noninteractive test
  michelogram(ob2);
  cerr << endl;
#endif
  }

#if 0    

  // disabled to get noninteractive test
  {
    shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E953);
    
    shared_ptr<ProjDataInfo> proj_data_info_ptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
		                    /*span*/1, 10,/*views*/ 96, /*tang_pos*/128, /*arc_corrected*/ true);
    michelogram(dynamic_cast<const ProjDataInfoCylindrical&>(*proj_data_info_ptr));
    cerr << endl;
  }
 {
    shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E953);
    
    shared_ptr<ProjDataInfo> proj_data_info_ptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
		                    /*span*/7, 10,/*views*/ 96, /*tang_pos*/128, /*arc_corrected*/ true);
    michelogram(dynamic_cast<const ProjDataInfoCylindrical&>(*proj_data_info_ptr));
    cerr << endl;
  }
#endif


 shared_ptr<Scanner> scanner_ptr(new Scanner(Scanner::E953));
  cerr << "Tests with proj_data_info without mashing and axial compression\n\n";
  // Note: test without axial compression requires that all ring differences 
  // are in some segment, so use maximum ring difference
  shared_ptr<ProjDataInfo> proj_data_info_ptr(
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				  /*span*/1, scanner_ptr->get_num_rings()-1,
				  /*views*/ scanner_ptr->get_num_detectors_per_ring()/2, 
				  /*tang_pos*/64, 
				  /*arc_corrected*/ true));
test_cylindrical_proj_data_info(dynamic_cast<ProjDataInfoCylindricalArcCorr &>(*proj_data_info_ptr));

  cerr << "\nTests with proj_data_info with mashing and axial compression\n\n";
  proj_data_info_ptr.reset(
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				  /*span*/5, scanner_ptr->get_num_rings()-1,
				  /*views*/ scanner_ptr->get_num_detectors_per_ring()/2/8, 
				  /*tang_pos*/64, 
				  /*arc_corrected*/ true));
    test_cylindrical_proj_data_info(dynamic_cast<ProjDataInfoCylindricalArcCorr &>(*proj_data_info_ptr));
}


/*!
  \ingroup test
  \brief Test class for ProjDataInfoCylindricalNoArcCorr
*/

class ProjDataInfoCylindricalNoArcCorrTests: public ProjDataInfoCylindricalTests
{
public:  
  void run_tests();
private:
  void test_proj_data_info(ProjDataInfoCylindricalNoArcCorr& proj_data_info);
};


void
ProjDataInfoCylindricalNoArcCorrTests::
run_tests()
{ 
  cerr << "\n-------- Testing ProjDataInfoCylindricalNoArcCorr --------\n";
  shared_ptr<Scanner> scanner_ptr(new Scanner(Scanner::E953));
  cerr << "Tests with proj_data_info without mashing and axial compression\n\n";
  // Note: test without axial compression requires that all ring differences 
  // are in some segment, so use maximum ring difference
  shared_ptr<ProjDataInfo> proj_data_info_ptr(
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				  /*span*/1, scanner_ptr->get_num_rings()-1,
				  /*views*/ scanner_ptr->get_num_detectors_per_ring()/2, 
				  /*tang_pos*/64, 
				  /*arc_corrected*/ false));
    test_proj_data_info(dynamic_cast<ProjDataInfoCylindricalNoArcCorr &>(*proj_data_info_ptr));

  cerr << "\nTests with proj_data_info with mashing and axial compression\n\n";
  proj_data_info_ptr.reset(
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				  /*span*/5, scanner_ptr->get_num_rings()-1,
				  /*views*/ scanner_ptr->get_num_detectors_per_ring()/2/8, 
				  /*tang_pos*/64, 
				  /*arc_corrected*/ false));
    test_proj_data_info(dynamic_cast<ProjDataInfoCylindricalNoArcCorr &>(*proj_data_info_ptr));
}

void
ProjDataInfoCylindricalNoArcCorrTests::
test_proj_data_info(ProjDataInfoCylindricalNoArcCorr& proj_data_info)
{
  test_cylindrical_proj_data_info(proj_data_info);

  const int num_detectors = proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();

#ifndef TEST_ONLY_GET_BIN
  if (proj_data_info.get_view_mashing_factor()==1)
    {
      // these tests work only without mashing

      cerr << "\n\tTest code for sinogram <-> detector conversions.";
      
#ifdef STIR_OPENMP
      #pragma omp parallel for schedule(dynamic)
#endif
      for (int det_num_a = 0; det_num_a < num_detectors; det_num_a++)
	for (int det_num_b = 0; det_num_b < num_detectors; det_num_b++)
	  {
	    int det1, det2;
	    bool positive_segment;
            int tang_pos_num;
            int view;
	    
	    // skip case of equal detectors (as this is a singular LOR)
	    if (det_num_a == det_num_b)
	      continue;
	    
	    positive_segment = 
	      proj_data_info.get_view_tangential_pos_num_for_det_num_pair(view, tang_pos_num, det_num_a, det_num_b);
	    proj_data_info.get_det_num_pair_for_view_tangential_pos_num(det1, det2, view, tang_pos_num);
	    
	    if (!check((det_num_a == det1 && det_num_b == det2 && positive_segment) ||
		       (det_num_a == det2 && det_num_b == det1 && !positive_segment)))
	      {
		cerr << "Problem at det1 = " << det_num_a << ", det2 = " << det_num_b
		     << "\n  dets -> sino -> dets gives new detector numbers "
		     << det1 << ", " << det2 << endl;
		continue;
	      }
	    if (!check(view < num_detectors/2))
	      {
		cerr << "Problem at det1 = " << det_num_a << ", det2 = " << det_num_b
		     << ":\n  view is too big : " << view << endl;
	      }
	    if (!check(tang_pos_num < num_detectors/2 && tang_pos_num >= -(num_detectors/2)))
	      {
		cerr << "Problem at det1 = " << det_num_a << ", det2 = " << det_num_b
		     << ":\n  tang_pos_num is out of range : " << tang_pos_num << endl;
	      }
	  } // end of detectors_to_sinogram, sinogram_to_detector test

	
#ifdef STIR_OPENMP
#pragma omp parallel for
#endif
      for (int view = 0; view < num_detectors/2; ++view)
	for (int tang_pos_num = -(num_detectors/2)+1; tang_pos_num < num_detectors/2; ++tang_pos_num)
	  {
	    int new_tang_pos_num, new_view;
	    bool positive_segment;
            int det_num_a;
            int det_num_b;

	    proj_data_info.get_det_num_pair_for_view_tangential_pos_num(det_num_a, det_num_b, view, tang_pos_num);
	    positive_segment = 
	      proj_data_info.get_view_tangential_pos_num_for_det_num_pair(new_view, new_tang_pos_num, det_num_a, det_num_b);

	    if (tang_pos_num != new_tang_pos_num || view != new_view || !positive_segment)
	      {
		cerr << "Problem at view = " << view << ", tang_pos_num = " << tang_pos_num
		     << "\n   sino -> dets -> sino gives new view, tang_pos_num :"
		     << new_view << ", " << new_tang_pos_num 
		     << " with detector swapping " << positive_segment
		     << endl;
	      }
	  } // end of sinogram_to_detector, detectors_to_sinogram test
	
    } // end of tests that work only without mashing

  if (proj_data_info.get_view_mashing_factor()==1
      && proj_data_info.get_max_ring_difference(0) == proj_data_info.get_min_ring_difference(0)
      && proj_data_info.get_max_ring_difference(1) == proj_data_info.get_min_ring_difference(1)
      && proj_data_info.get_max_ring_difference(2) == proj_data_info.get_min_ring_difference(2)
      )
    {
      // these tests work only without mashing and axial compression
	
      cerr << "\n\tTest code for detector,ring -> bin and back conversions.";
      
      DetectionPositionPair<> det_pos_pair;
      for (det_pos_pair.pos1().axial_coord() = 0; 
           det_pos_pair.pos1().axial_coord() <= 2;
	   det_pos_pair.pos1().axial_coord()++)
	for (det_pos_pair.pos2().axial_coord() = 0; 
	     det_pos_pair.pos2().axial_coord() <= 2; 
	     det_pos_pair.pos2().axial_coord()++)
#ifdef STIR_OPENMP
            // insert a parallel for here for testing.
            // we do it at this level to avoid too much overhead for the thread creation, while still having enough jobs to do
            // note: for-loop writing somewhat awkwardly as openmp needs int variables for the loop
#pragma omp parallel for firstprivate(det_pos_pair)
#endif
          for (int tangential_coord1 = 0; 
               tangential_coord1 < num_detectors; 
               tangential_coord1++)
            for (det_pos_pair.pos2().tangential_coord() = 0; 
                 det_pos_pair.pos2().tangential_coord() < (unsigned)num_detectors;
                 det_pos_pair.pos2().tangential_coord()++)
	      {
                // set from for-loop variable
                det_pos_pair.pos1().tangential_coord() = (unsigned)tangential_coord1;
		// skip case of equal detector numbers (as this is either a singular LOR)
		// or an LOR parallel to the scanner axis
		if (det_pos_pair.pos1().tangential_coord() == det_pos_pair.pos2().tangential_coord())
		  continue;
		Bin bin;
		DetectionPositionPair<> new_det_pos_pair;
		const bool there_is_a_bin =
		  proj_data_info.get_bin_for_det_pos_pair(bin, det_pos_pair) ==
		    Succeeded::yes;       
		if (there_is_a_bin)
		  proj_data_info.get_det_pos_pair_for_bin(new_det_pos_pair, bin);
		if (!check(there_is_a_bin, "checking if there is a bin for this det_pos_pair") ||
		    !check(det_pos_pair == new_det_pos_pair, "checking if we round-trip to the same detection positions"))
		    {
		      cerr << "Problem at det1 = " << det_pos_pair.pos1().tangential_coord() 
			   << ", det2 = " << det_pos_pair.pos2().tangential_coord()
			   << ", ring1 = " << det_pos_pair.pos1().axial_coord() 
			   << ", ring2 = " << det_pos_pair.pos2().axial_coord()
			   << endl;
		      if (there_is_a_bin)
			 cerr << "  dets,rings -> bin -> dets,rings, gives new numbers:\n\t"
			      << "det1 = " << new_det_pos_pair.pos1().tangential_coord() 
			   
			      << ", det2 = " << new_det_pos_pair.pos2().tangential_coord()
			      << ", ring1 = " << new_det_pos_pair.pos1().axial_coord() 
			      << ", ring2 = " << new_det_pos_pair.pos2().axial_coord()
			 << endl;
		  }
	
	      } // end of get_bin_for_det_pos_pair and vice versa code

      cerr << "\n\tTest code for bin -> detector,ring and back conversions. (This might take a while...)";

      {
	Bin bin;
	// set value for comparison later on
	bin.set_bin_value(0);
	for (bin.segment_num() = max(-5,proj_data_info.get_min_segment_num()); 
	     bin.segment_num() <= min(5,proj_data_info.get_max_segment_num()); 
	     ++bin.segment_num())
	  for (bin.axial_pos_num() = proj_data_info.get_min_axial_pos_num(bin.segment_num());
	       bin.axial_pos_num() <= proj_data_info.get_max_axial_pos_num(bin.segment_num());
	       ++bin.axial_pos_num())
#ifdef STIR_OPENMP
            // insert a parallel for here for testing.
            // we do it at this level to avoid too much overhead for the thread creation, while still having enough jobs to do
            // Note that the omp construct needs an int loop variable
#pragma omp parallel for firstprivate(bin)
#endif
            for (int tangential_pos_num = -(num_detectors/2)+1; 
                 tangential_pos_num < num_detectors/2; 
                 ++tangential_pos_num)
              for (bin.view_num() = 0; bin.view_num() < num_detectors/2; ++bin.view_num())
		{
                  // set from for-loop variable
                  bin.tangential_pos_num() = tangential_pos_num;
		  Bin new_bin;
		  // set value for comparison with bin
		  new_bin.set_bin_value(0);
		  DetectionPositionPair<> det_pos_pair;
		  proj_data_info.get_det_pos_pair_for_bin(det_pos_pair, bin);
	
		  const bool there_is_a_bin =
		    proj_data_info.get_bin_for_det_pos_pair(new_bin, 
							    det_pos_pair) ==
		    Succeeded::yes;
		  if (!check(there_is_a_bin, "checking if there is a bin for this det_pos_pair") ||
		      !check(bin == new_bin, "checking if we round-trip to the same bin"))
		    {
		      cerr << "Problem at  segment = " << bin.segment_num() 
			   << ", axial pos " << bin.axial_pos_num()
			   << ", view = " << bin.view_num() 
			   << ", tangential_pos_num = " << bin.tangential_pos_num() << "\n";
		      if (there_is_a_bin)
			cerr  << "  bin -> dets -> bin, gives new numbers:\n\t"
			      << "segment = " << new_bin.segment_num() 
			      << ", axial pos " << new_bin.axial_pos_num()
			      << ", view = " << new_bin.view_num() 
			      << ", tangential_pos_num = " << new_bin.tangential_pos_num()
			      << endl;
		    }
	
		} // end of get_det_pos_pair_for_bin and back code
      }
    } // end of tests which require no mashing nor axial compression

  {
    cerr << "\n\tTest code for bins <-> detectors routines that work with any mashing and axial compression";

    Bin bin;
    // set value for comparison later on
    bin.set_bin_value(0);
    std::vector<DetectionPositionPair<> > det_pos_pairs;
#ifdef STIR_OPENMP
    //#pragma omp parallel for schedule(dynamic)
#endif
    for (bin.segment_num() = proj_data_info.get_min_segment_num(); 
	 bin.segment_num() <= proj_data_info.get_max_segment_num(); 
	 ++bin.segment_num())
      for (bin.axial_pos_num() = proj_data_info.get_min_axial_pos_num(bin.segment_num());
	   bin.axial_pos_num() <= proj_data_info.get_max_axial_pos_num(bin.segment_num());
	   ++bin.axial_pos_num())
	for (bin.view_num() = proj_data_info.get_min_view_num(); 
	     bin.view_num() <= proj_data_info.get_max_view_num(); 
	     ++bin.view_num())
	  for (bin.tangential_pos_num() = proj_data_info.get_min_tangential_pos_num();
	       bin.tangential_pos_num() <= proj_data_info.get_max_tangential_pos_num();
	       ++bin.tangential_pos_num())
	    {
	      proj_data_info.get_all_det_pos_pairs_for_bin(det_pos_pairs, bin);
	      Bin new_bin;
	      // set value for comparison with bin
	      new_bin.set_bin_value(0);
	      for (std::vector<DetectionPositionPair<> >::const_iterator det_pos_pair_iter = det_pos_pairs.begin();
		   det_pos_pair_iter != det_pos_pairs.end();
		   ++det_pos_pair_iter)
		{
		  const bool there_is_a_bin =
		    proj_data_info.get_bin_for_det_pos_pair(new_bin, 
							    *det_pos_pair_iter) ==
		    Succeeded::yes;
		  if (!check(there_is_a_bin, "checking if there is a bin for this det_pos_pair") ||
		      !check(bin == new_bin, "checking if we round-trip to the same bin"))
		    {
		      cerr << "Problem at  segment = " << bin.segment_num() 
			   << ", axial pos " << bin.axial_pos_num()
			   << ", view = " << bin.view_num() 
			   << ", tangential_pos_num = " << bin.tangential_pos_num() << "\n";
		      if (there_is_a_bin)
			cerr << "  bin -> dets -> bin, gives new numbers:\n\t"
			     << "segment = " << new_bin.segment_num() 
			     << ", axial pos " << new_bin.axial_pos_num()
			     << ", view = " << new_bin.view_num() 
			     << ", tangential_pos_num = " << new_bin.tangential_pos_num()
			     << endl;
		    }
		} // end of iteration of det_pos_pairs
	    } // end of loop over all bins
  } // end of get_all_det_pairs_for_bin and back code
#endif //TEST_ONLY_GET_BIN

  {
    cerr << endl;
    cerr << "\tTesting find scanner coordinates given cartesian and vice versa." << endl;
    {   
      const int num_detectors_per_ring =
	proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();
      const int num_rings =
	proj_data_info.get_scanner_ptr()->get_num_rings();

#ifdef STIR_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for ( int Ring_A = 0; Ring_A < num_rings; Ring_A+=num_rings/3)
	for ( int Ring_B = 0; Ring_B < num_rings; Ring_B+=num_rings/3)
	  for ( int det1 =0; det1 < num_detectors_per_ring; ++det1)
	    for ( int det2 =0; det2 < num_detectors_per_ring; ++det2)	  
	      {
		if (det1==det2)
		  continue;
		CartesianCoordinate3D<float> coord_1;
		CartesianCoordinate3D<float> coord_2;
	  
		proj_data_info.find_cartesian_coordinates_given_scanner_coordinates (coord_1,coord_2,
										     Ring_A,Ring_B, 
										     det1,det2);
	  
		const CartesianCoordinate3D<float> coord_1_new = coord_1 + (coord_2-coord_1)*5;
		const CartesianCoordinate3D<float> coord_2_new = coord_1 + (coord_2-coord_1)*2;
	  
		int det1_f, det2_f,ring1_f, ring2_f;
	  
		check(proj_data_info.find_scanner_coordinates_given_cartesian_coordinates(det1_f, det2_f, ring1_f, ring2_f,
										    coord_1_new, coord_2_new) ==
		      Succeeded::yes);
		if (det1_f == det1 && Ring_A == ring1_f)
		  { 
		    check_if_equal( det1_f, det1, "test on det1");
		    check_if_equal( Ring_A, ring1_f, "test on ring1");
		    check_if_equal( det2_f, det2, "test on det2");
		    check_if_equal( Ring_B, ring2_f, "test on ring1");
		  }
		else
		  {
		    check_if_equal( det2_f, det1, "test on det1");
		    check_if_equal( Ring_B, ring1_f, "test on ring1");
		    check_if_equal( det1_f, det2, "test on det2");
		    check_if_equal( Ring_A, ring2_f, "test on ring1");
		  }
	      }
    }
  }
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  set_default_num_threads();

  ProjDataInfoCylindricalArcCorrTests tests;
  tests.run_tests();
  ProjDataInfoCylindricalNoArcCorrTests tests1;
  tests1.run_tests();
  return tests.main_return_value();
}
