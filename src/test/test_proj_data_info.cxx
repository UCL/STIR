//
// $Id$
//
/*!

  \file
  \ingroup test

  \brief Test program for ProjDataInfo hierarchy

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <math.h>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::setw;
using std::endl;
using std::min;
#endif

START_NAMESPACE_STIR
// prints a michelogram to the screen
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

/*!
  \ingroup test
  \brief Test class for ProjDataInfoCylindrical
*/

class ProjDataInfoCylindricalTests: public RunTests
{
public:  
  void run_tests();
};


void
ProjDataInfoCylindricalTests::
run_tests()
{
  shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E953);

  cerr << "Testing consistency between different implementations\n";
  {
    shared_ptr<ProjDataInfo> proj_data_info_ptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				    /*span*/5, 10,/*views*/ scanner_ptr->get_num_detectors_per_ring()/2, /*tang_pos*/128, /*arc_corrected*/ false);
    
    ProjDataInfoCylindrical& proj_data_info_cyl =
      dynamic_cast<ProjDataInfoCylindrical&>(*proj_data_info_ptr);

    Bin bin(proj_data_info_ptr->get_max_segment_num(),
	    1,
	    proj_data_info_ptr->get_max_axial_pos_num(proj_data_info_ptr->get_max_segment_num())/2,
	    1);
    check_if_equal(proj_data_info_ptr->get_sampling_in_m(bin),
		   proj_data_info_cyl.get_sampling_in_m(bin),
		   "test consistency get_sampling_in_m");
    check_if_equal(proj_data_info_ptr->get_sampling_in_t(bin),
		   proj_data_info_cyl.get_sampling_in_t(bin),
		   "test consistency get_sampling_in_t");

    check_if_equal(proj_data_info_ptr->get_tantheta(bin),
		   proj_data_info_cyl.get_tantheta(bin),
		   "test consistency get_tantheta");
    check_if_equal(proj_data_info_ptr->get_costheta(bin),
		   proj_data_info_cyl.get_costheta(bin),
		   "test consistency get_costheta");

    check_if_equal(proj_data_info_ptr->get_costheta(bin),
		   cos(atan(proj_data_info_ptr->get_tantheta(bin))),
		   "cross check get_costheta and get_tantheta");

    // try the same with a non-standard ring spacing
    proj_data_info_cyl.set_ring_spacing(2);

    check_if_equal(proj_data_info_ptr->get_sampling_in_m(bin),
		   proj_data_info_cyl.get_sampling_in_m(bin),
		   "test consistency get_sampling_in_m");
    check_if_equal(proj_data_info_ptr->get_sampling_in_t(bin),
		   proj_data_info_cyl.get_sampling_in_t(bin),
		   "test consistency get_sampling_in_t");

    check_if_equal(proj_data_info_ptr->get_tantheta(bin),
		   proj_data_info_cyl.get_tantheta(bin),
		   "test consistency get_tantheta");
    check_if_equal(proj_data_info_ptr->get_costheta(bin),
		   proj_data_info_cyl.get_costheta(bin),
		   "test consistency get_costheta");

    check_if_equal(proj_data_info_ptr->get_costheta(bin),
		   cos(atan(proj_data_info_ptr->get_tantheta(bin))),
		   "cross check get_costheta and get_tantheta");
  }

  cerr << "Test ring pair to segment,ax_pos (span 1)\n";
  {
    shared_ptr<Scanner> small_scanner_ptr = 
      new Scanner(Scanner::Unknown_Scanner,
		  "Small scanner",
		  /*num_detectors_per_ring*/ 80, 
		  /* num_rings */ 3, 
		  /*max_num_non_arccorrected_bins*/ 40, 
		  /*RingRadius*/ 400.F, /*RingSpacing*/ 5.F, 
		  /*BinSize*/ 4.5F, /*intrTilt*/ 0);

    shared_ptr<ProjDataInfo> proj_data_info_ptr =
      ProjDataInfo::ProjDataInfoCTI(small_scanner_ptr,
				    /*span*/1, small_scanner_ptr->get_num_rings()-1,
				    /*views*/ scanner_ptr->get_num_detectors_per_ring()/2, /*tang_pos*/128, /*arc_corrected*/ false);
    
    ProjDataInfoCylindrical& proj_data_info =
      dynamic_cast<ProjDataInfoCylindrical&>(*proj_data_info_ptr);

    for (int ring1=0; ring1<small_scanner_ptr->get_num_rings(); ++ring1)
      for (int ring2=0; ring2<small_scanner_ptr->get_num_rings(); ++ring2)
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

	  check_if_equal(ring_pairs.size(), 1,
			 "test total number of ring-pairs for 1 segment/ax_pos should be 1 for span=1\n");
	  check_if_equal(ring1, ring_pairs[0].first,
			 "test ring1 equal after going to segment/ax_pos and returning (version with all ring_pairs)\n");
	  check_if_equal(ring2, ring_pairs[0].second,
			 "test ring2 equal after going to segment/ax_pos and returning (version with all ring_pairs)\n");
	}
  }

  cerr << "Test ring pair to segment,ax_pos and vice versa (span 5)\n";
  {
    shared_ptr<ProjDataInfo> proj_data_info_ptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				    /*span*/5, 10,/*views*/ scanner_ptr->get_num_detectors_per_ring()/2, /*tang_pos*/128, /*arc_corrected*/ false);
    
    ProjDataInfoCylindrical& proj_data_info =
      dynamic_cast<ProjDataInfoCylindrical&>(*proj_data_info_ptr);

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
  cerr << "Testing ProjDataInfoCylindricalArcCorr\n";
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
    shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E953);
    
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
      check_if_equal( phi, 10*ob2.get_azimuthal_angle_sampling(), " get_phi , seg 0");
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
      
      float thetatest = 2*ob2.get_axial_sampling(1)/(2*sqrt(square(scanner_ptr->get_ring_radius())-square(s)));
      
      check_if_equal( theta, thetatest,"test on get_tantheta, seg 1");
      check_if_equal( phi, 10*ob2.get_azimuthal_angle_sampling(), " get_phi , seg 1");
      // KT 25/10/2000 adjust to new convention
      const float ax_pos_origin =
	(ob2.get_min_axial_pos_num(1) + ob2.get_max_axial_pos_num(1))/2.F;
      check_if_equal( t, (10-ax_pos_origin)/sqrt(1+square(thetatest))*ob2.get_axial_sampling(1) , "get_t, seg 1");
      check_if_equal( s, 20*ob2.get_tangential_sampling() , "get_s, seg 1");
    }

    ProjDataInfoCylindricalTests::run_tests();

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

}


/*!
  \ingroup test
  \brief Test class for ProjDataInfoCylindricalNoArcCorr
*/

class ProjDataInfoCylindricalNoArcCorrTests: public RunTests
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
  cerr << "Testing ProjDataInfoCylindricalNoArcCorr\n";
  shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E953);
  shared_ptr<ProjDataInfo> proj_data_info_ptr =
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				  /*span*/1, 10,/*views*/ scanner_ptr->get_num_detectors_per_ring()/2, /*tang_pos*/128, /*arc_corrected*/ false);
    test_proj_data_info(dynamic_cast<ProjDataInfoCylindricalNoArcCorr &>(*proj_data_info_ptr));
}

void
ProjDataInfoCylindricalNoArcCorrTests::
test_proj_data_info(ProjDataInfoCylindricalNoArcCorr& proj_data_info)
{
  int det_num_a;
  int det_num_b;
  int ring_a;
  int ring_b;
  int tang_pos_num;
  int view;
  
  const int num_detectors = proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();

  cerr << "\nTest code for sinogram <-> detector conversions. \n"
       << "No output means everything is fine." << endl;

  for (det_num_a = 0; det_num_a < num_detectors; det_num_a++)
    for (det_num_b = 0; det_num_b < num_detectors; det_num_b++)
      {
	int det1, det2;
	bool positive_segment;

        // skip case of equal detectors (as this is a singular LOR)
	if (det_num_a == det_num_b)
	  continue;

	positive_segment = 
	  proj_data_info.get_view_tangential_pos_num_for_det_num_pair(view, tang_pos_num, det_num_a, det_num_b);
	proj_data_info.get_det_num_pair_for_view_tangential_pos_num(det1, det2, view, tang_pos_num);

	if (!(det_num_a == det1 && det_num_b == det2 && positive_segment) &&
	    !(det_num_a == det2 && det_num_b == det1 && !positive_segment))
	  {
	    cerr << "Problem at det1 = " << det_num_a << ", det2 = " << det_num_b
		 << endl
		 << "  dets -> sino -> dets gives new detector numbers "
		 << det1 << ", " << det2 << endl;
	    continue;
	  }
	if (view >= num_detectors/2)
	  {
	    cerr << "Problem at det1 = " << det_num_a << ", det2 = " << det_num_b
		 << endl
		 << "  view is too big : " << view << endl;
	  }
	if (tang_pos_num >= num_detectors/2 || tang_pos_num < -(num_detectors/2))
	  {
	    cerr << "Problem at det1 = " << det_num_a << ", det2 = " << det_num_b
		 << endl
		 << "  tang_pos_num is out of range : " << tang_pos_num << endl;
	  }
      } // end of detectors_to_sinogram, sinogram_to_detector test

	
  for (view = 0; view < num_detectors/2; ++view)
    for (tang_pos_num = -(num_detectors/2)+1; tang_pos_num < num_detectors/2; ++tang_pos_num)
      {
	int new_tang_pos_num, new_view;
	bool positive_segment;

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
	
	
  cerr << "\nTest code for detector,ring -> bin and back conversions. \n"
       << "No output means everything is fine." << endl;

  for (ring_a = 0; ring_a <= 2; ring_a++)
    for (ring_b = 0; ring_b <= 2; ring_b++)
      for (det_num_a = 0; det_num_a < num_detectors; det_num_a++)
	for (det_num_b = 0; det_num_b < num_detectors; det_num_b++)
	  {
            // skip case of equal detector numbers (as this is either a singular LOR)
            // or an LOR parallel to the scanner axis
            if (det_num_a == det_num_b)
              continue;
            Bin bin;
            int det1;
            int det2;
            int ring1;
            int ring2;
            
	    proj_data_info.get_bin_for_det_pair(bin,
						det_num_a, ring_a, 
						det_num_b, ring_b);
	
	    proj_data_info.get_det_pair_for_bin(det1, ring1,
						det2, ring2,
						bin);
	
	
	    if (!(det_num_a == det1 && det_num_b == det2 &&
		  ring_a == ring1 && ring_b == ring2) &&
		!(det_num_a == det2 && det_num_b == det1 &&
		  ring_a == ring2 && ring_b == ring1))
	      {
		cerr << "Problem at det1 = " << det_num_a << ", det2 = " << det_num_b
		     << ", ring1 = " << ring_a << ", ring2 = " << ring_b
		     << endl
		     << "  dets,rings -> bin -> dets,rings, gives new numbers:\n\t"
		     << "det1 = " << det1 << ", det2 = " << det2
		     << ", ring1 = " << ring1 << ", ring2 = " << ring2
		     << endl;
	      }
	
	  } // end of get_bin_for_det_pair and vice versa code

  cerr << "\nTest code for bin -> detector,ring and back conversions. (This might take a while...)\n"
       << "No output means everything is fine." << endl;

  {
    Bin bin;
    // set value for comparison later on
    bin.set_bin_value(0);
    for (bin.segment_num() = proj_data_info.get_min_segment_num(); 
	 bin.segment_num() <= proj_data_info.get_max_segment_num(); 
	 ++bin.segment_num())
      for (bin.axial_pos_num() = proj_data_info.get_min_axial_pos_num(bin.segment_num());
	   bin.axial_pos_num() <= proj_data_info.get_max_axial_pos_num(bin.segment_num());
	   ++bin.axial_pos_num())
	for (bin.view_num() = 0; bin.view_num() < num_detectors/2; ++bin.view_num())
	  for (bin.tangential_pos_num() = -(num_detectors/2)+1; 
	       bin.tangential_pos_num() < num_detectors/2; 
	       ++bin.tangential_pos_num())
	    {

	      Bin new_bin;
	      // set value for comparison with bin
	      new_bin.set_bin_value(0);
	      int det1, det2, ring1, ring2;
	      proj_data_info.get_det_pair_for_bin(det1, ring1,
						  det2, ring2,
						  bin);
	      proj_data_info.get_bin_for_det_pair(new_bin,
						  det1, ring1, 
						  det2, ring2);
	
	
	
	      if (bin != new_bin)
	      {
		cerr << "Problem at  segment = " << bin.segment_num() 
		     << ", axial pos " << bin.axial_pos_num()
		     << ", view = " << bin.view_num() 
		     << ", tangential_pos_num = " << bin.tangential_pos_num() << "\n"
		     << "  bin -> dets -> bin, gives new numbers:\n\t"
                     << "segment = " << new_bin.segment_num() 
		     << ", axial pos " << new_bin.axial_pos_num()
		     << ", view = " << new_bin.view_num() 
		     << ", tangential_pos_num = " << new_bin.tangential_pos_num()
		     << endl;
	      }
	
	  } // end of get_det_pair_for_bin and back code
  }
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  ProjDataInfoCylindricalArcCorrTests tests;
  tests.run_tests();
  ProjDataInfoCylindricalNoArcCorrTests tests1;
  tests1.run_tests();
  return tests.main_return_value();
}
