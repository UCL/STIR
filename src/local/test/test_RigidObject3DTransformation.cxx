//
// $Id$
//
/*!
  \file
  \ingroup test
  \brief Test program for RigidObject3DTransformation functions
  \author Sanida Mustafovic
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$ , Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/RunTests.h"
#include "local/stir/Quaternion.h"
#include <iostream>
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/listmode/TimeFrameDefinitions.h"
#include "stir/shared_ptr.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#ifdef DO_TIMINGS
#include "stir/CPUTimer.h"
#endif

//#include "local/stir/motion/Polaris_MT_File.h"
//#include "local/stir/motion/RigidObject3DMotionFromPolaris.h"
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR

class RigidObject3DTransformationTests: public RunTests
{
public:  
  void run_tests();
};

void
RigidObject3DTransformationTests::run_tests()
{
  // testing inverse
  {
    Quaternion<float> quat(1,-2,3,8);
    quat.normalise();
    const CartesianCoordinate3D<float> translation(111,-12,152);
    
    const RigidObject3DTransformation ro3dtrans(quat, translation);
    
    RigidObject3DTransformation ro3dtrans_inverse =ro3dtrans;
    ro3dtrans_inverse =ro3dtrans_inverse.inverse();
    {
      const Quaternion<float> quat_original = ro3dtrans.get_quaternion();
      const Quaternion<float> quat_inverse = ro3dtrans_inverse.get_quaternion();
      
      const Quaternion<float> unity = quat_original * quat_inverse;
      
      check_if_equal(unity[1], 1.F, "test on inverse quat -- scalar");
      check_if_equal(unity[2], 0.F, "test on inverse quat -- vector1");
      check_if_equal(unity[3], 0.F, "test on inverse quat -- vector2");
      check_if_equal(unity[4], 0.F, "test on inverse quat -- vector3");
    }      

    
    for (int i=0; i<1000; ++i)
    {
      const CartesianCoordinate3D<float> point(210*i,-55-i,2+2*i);
      const CartesianCoordinate3D<float> transformed_point =ro3dtrans.transform_point(point);
      //Testing norm of the original and transformed point 
      {
	const float norm_original = sqrt(square(point.z()) +square(point.y())+square(point.x()));
	const float norm_transformed = sqrt(square(point.z()) +square(point.y())+square(point.x()));
	
	check_if_equal(norm_original, norm_transformed, "test on norm");
      }
      // Testing to see if inverse gets us back
      {
	
	const CartesianCoordinate3D<float> transformed_back_point =
	    ro3dtrans_inverse.transform_point(transformed_point);
	// compare with original by checking norm of difference
	// divide by norm(point) such that we're looking at a relative measure
	check_if_zero(norm(point-transformed_back_point)/norm(point), 
	              "test on inverse transformation of transformed point");
	
      }
    }
  }

#if 0
  cerr << "Testing reading of mt files" <<endl;
  
  const string fdef_filename = "H09990.fdef";
  TimeFrameDefinitions tfdef(fdef_filename);
  const float polaris_time_offset =  3241;
  const string mt_filename = "H09990.mt";
  
  shared_ptr<Polaris_MT_File> mt_file_ptr = 
    new Polaris_MT_File(mt_filename);
  RigidObject3DMotionFromPolaris ro3dmfromp(mt_filename,mt_file_ptr);
  ro3dmfromp.set_polaris_time_offset(polaris_time_offset);
  
  int number_of_frames = tfdef.get_num_frames();
  for ( int frame = 1; frame<=number_of_frames;frame++)
  {
    float start = tfdef.get_start_time(frame);
    float end  = tfdef.get_end_time(frame);
    
    RigidObject3DTransformation ro3dtrans;
    
    ro3dmfromp.get_motion(ro3dtrans,start); //+polaris_time_offset);
    const Quaternion<float> quat_s = ro3dtrans.get_quaternion();
    const CartesianCoordinate3D<float> trans_s =ro3dtrans.get_translation();
    cerr << " Quaternion is " << quat_s << endl;
    cerr << " Translation is " << trans_s << endl;
    int i = 1;
    while (i<= end)
    {
      RigidObject3DTransformation ro3dtrans_test;
      ro3dmfromp.get_motion(ro3dtrans_test,start+i);
      const Quaternion<float> quat = ro3dtrans.get_quaternion();
      const CartesianCoordinate3D<float> trans =ro3dtrans.get_translation();
      
      check_if_equal( quat_s[1], quat[1],"test on -scalar");
      check_if_equal( quat_s[2], quat[2],"test on -vector-1");
      check_if_equal( quat_s[3], quat[3],"test on -vector-2");
      check_if_equal( quat_s[4], quat[4],"test on -vector-3");
      i+=100;
    }  
  }
#endif
  // cerr << " Testing compose " << endl;
  {
    Quaternion<float> quat_1(1,-2,3,8);
    quat_1.normalise();
    const CartesianCoordinate3D<float> translation_1(111,-12,152);    
    const RigidObject3DTransformation ro3dtrans_1(quat_1, translation_1);
    
    Quaternion<float> quat_2(1,-3,12,4);
    quat_2.normalise();
    const CartesianCoordinate3D<float> translation_2(1,-54,12);    
    const RigidObject3DTransformation ro3dtrans_2(quat_2, translation_2);
    
    Quaternion<float> quat_3(2,-7,24,1);
    quat_3.normalise();
    const CartesianCoordinate3D<float> translation_3(9,4,34);
    const RigidObject3DTransformation ro3dtrans_3(quat_3, translation_3);
#ifdef DO_TIMINGS    
    CPUTimer timer;
    timer.reset();
    CPUTimer compose_timer;
    compose_timer.reset();
#endif
    const RigidObject3DTransformation composed_ro3dtrans1=
      compose(ro3dtrans_3,
              compose(ro3dtrans_2,ro3dtrans_1));
    
    for (int i=0; i<1000; ++i)
    {
      const CartesianCoordinate3D<float> point(210*i,-55-i,2+2*i);
#ifdef DO_TIMINGS
      timer.start();
#endif
      const CartesianCoordinate3D<float> transformed_point_3 =
	ro3dtrans_3.
	transform_point(ro3dtrans_2.
	                transform_point(
	                                ro3dtrans_1.
	                                transform_point(point)));
#ifdef DO_TIMINGS      
      timer.stop();
      
      compose_timer.start();
#endif            
      const CartesianCoordinate3D<float> transformed_point_composed =
	composed_ro3dtrans1.transform_point(point);
#ifdef DO_TIMINGS
      compose_timer.stop();
#endif
      check_if_zero(norm(transformed_point_3-transformed_point_composed)/norm(transformed_point_3),
	            "test on compose");
    }
#ifdef DO_TIMINGS
    cerr << " Individual multiplications: " <<  timer.value() << " s CPU time"<<endl;
    cerr << "Combined: " <<  compose_timer.value() << " s CPU time"<<endl;
#endif
  }

  // testing transform_lor
  {
    shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E953);
    shared_ptr<ProjDataInfo> proj_data_info_ptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				    /*span*/1, scanner_ptr->get_num_rings()-1,
				    /*views*/ scanner_ptr->get_num_detectors_per_ring()/2, 
				    /*tang_pos*/scanner_ptr->get_num_detectors_per_ring()/2, 
				    /*arc_corrected*/ false);
    ProjDataInfoCylindricalNoArcCorr& proj_data_info =
      dynamic_cast<ProjDataInfoCylindricalNoArcCorr &>(*proj_data_info_ptr);

    Quaternion<float> quat(1,-2,3,8);
    quat.normalise();
    const CartesianCoordinate3D<float> translation(11,-12,15);
    
    const RigidObject3DTransformation ro3dtrans(quat, translation);
    
    RigidObject3DTransformation ro3dtrans_inverse =ro3dtrans;
    ro3dtrans_inverse =ro3dtrans_inverse.inverse();

    unsigned num_bins_checked = 0;

    for (int segment_num=proj_data_info.get_min_segment_num();
	 segment_num<=proj_data_info.get_max_segment_num();
	 ++segment_num)
      {
	for (int view_num=proj_data_info.get_min_view_num();
	     view_num<=proj_data_info.get_max_view_num();
	     view_num+=5)
	  {
	    // loop over axial_positions. Avoid using first and last position, as 
	    // the discretisation error can easily bring the transformed_bin back
	    // outside the range. We could test for that, but it would make
	    // the code much more complicated, and not give anything useful back.
	    for (int axial_pos_num=proj_data_info.get_min_axial_pos_num(segment_num)+1;
		 axial_pos_num<=proj_data_info.get_max_axial_pos_num(segment_num)-1;
		 axial_pos_num+=3)
	      {
		for (int tangential_pos_num=proj_data_info.get_min_tangential_pos_num()+1;
		     tangential_pos_num<=proj_data_info.get_max_tangential_pos_num()-1;
		     tangential_pos_num+=17)
		  {
		    ++num_bins_checked;

		    const Bin org_bin(segment_num,view_num,axial_pos_num,tangential_pos_num, /* value*/1);
	
		    Bin transformed_bin = org_bin;
		    ro3dtrans.transform_bin(transformed_bin, proj_data_info, proj_data_info);
	    
		    if (transformed_bin.get_bin_value()>0) // only check when the transformed_bin is within the range
		      {
			ro3dtrans_inverse.transform_bin(transformed_bin, proj_data_info, proj_data_info);
			if (!check(org_bin.get_bin_value() == transformed_bin.get_bin_value(), "transform_bin_with_inverse: value") ||
			    !check(std::abs(org_bin.segment_num() - transformed_bin.segment_num())<=1, "transform_bin_with_inverse: segment") ||
			    !check(std::abs(org_bin.view_num() - transformed_bin.view_num())<=1, "transform_bin_with_inverse: view") ||
			    !check(std::abs(org_bin.axial_pos_num() - transformed_bin.axial_pos_num())<=1, "transform_bin_with_inverse: axial_pos") ||
			    !check(std::abs(org_bin.tangential_pos_num() - transformed_bin.tangential_pos_num())<=1, "transform_bin_with_inverse: tangential_pos"))
			  {
			    cerr << "Problem at  segment = " << org_bin.segment_num() 
				 << ", axial pos " << org_bin.axial_pos_num()
				 << ", view = " << org_bin.view_num() 
				 << ", tangential_pos_num = " << org_bin.tangential_pos_num() << "\n";
			    cerr << "round-trip to  segment = " << transformed_bin.segment_num() 
				 << ", axial pos " << transformed_bin.axial_pos_num()
				 << ", view = " << transformed_bin.view_num() 
				 << ", tangential_pos_num = " << transformed_bin.tangential_pos_num() 
				 << " value=" << transformed_bin.get_bin_value()
				 <<"\n";
			  }
		      }
		  } // tangential_pos
	      } // axial_pos
	  } // view
      } //segment
    cerr << num_bins_checked << " num_bins checked\n";
  }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  RigidObject3DTransformationTests tests;
  tests.run_tests();
  
  return tests.main_return_value();
}
