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
#include "stir/CPUTimer.h"

#include "local/stir/motion/Polaris_MT_File.h"
#include "local/stir/motion/RigidObject3DMotionFromPolaris.h"
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
#if 0
  {
  Quaternion<float> quat(1,-2,3,8);
  quat.normalise();
  const CartesianCoordinate3D<float> translation(111,-12,152);
  
  RigidObject3DTransformation ro3dtrans(quat, translation);

  cerr << "Testing norm of the original and transformed point " << endl;

  const CartesianCoordinate3D<float> point(210,-55,2);
  const CartesianCoordinate3D<float> transformed_point =ro3dtrans.transform_point(point);
  {
    const float norm_original = sqrt(square(point.z()) +square(point.y())+square(point.x()));
    const float norm_transformed = sqrt(square(point.z()) +square(point.y())+square(point.x()));
    
    check_if_equal(norm_original, norm_transformed, "test on norm");
  }
  cerr << " Testing inverse of RO3DTRANSFORMATION " << endl;
  {
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
      //CartesianCoordinate3D<float> trans = ro3dtrans.get_translation();
      //CartesianCoordinate3D<float> trans_inverse = ro3dtrans_inverse.get_translation();
      
      //cerr << trans.z()<<"   "<< trans.y()<< "   "<< trans.x()<< endl;
      //cerr << trans_inverse.z()<<"   "<< trans_inverse.y()<< "   "<< trans_inverse.x()<< endl;
    }      
    const CartesianCoordinate3D<float> transformed_back_point =ro3dtrans_inverse.transform_point(transformed_point);
    check_if_zero(norm(point-transformed_back_point), "test on inverse transformation of transformed point");
    cerr << point <<transformed_point<< transformed_back_point<<endl;
  }
  }
#endif
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
  cerr << " Testing compose " << endl;

  Quaternion<float> quat_1(1,-2,3,8);
  quat_1.normalise();
  const CartesianCoordinate3D<float> translation_1(111,-12,152);
  
  RigidObject3DTransformation ro3dtrans_1(quat_1, translation_1);

  Quaternion<float> quat_2(1,-3,12,4);
  quat_2.normalise();
  const CartesianCoordinate3D<float> translation_2(1,-54,12);
  
  RigidObject3DTransformation ro3dtrans_2(quat_2, translation_2);

  Quaternion<float> quat_3(2,-7,24,1);
  quat_3.normalise();
  const CartesianCoordinate3D<float> translation_3(9,4,34);
 
  RigidObject3DTransformation ro3dtrans_3(quat_3, translation_3);

  const CartesianCoordinate3D<float> point(210,-55,2);
  CPUTimer timer;
  timer.reset();
  timer.start();

  //const CartesianCoordinate3D<float> transformed_point_1 =ro3dtrans_1.transform_point(point);
  //const CartesianCoordinate3D<float> transformed_point_2 =ro3dtrans_2.transform_point(transformed_point_1);
  
  const CartesianCoordinate3D<float> transformed_point_3 =
    ro3dtrans_3.
    transform_point(ro3dtrans_2.
		    transform_point(
				ro3dtrans_1.
				transform_point(point)));

  timer.stop();
  cerr << " Individual multiplications: " <<  timer.value() << " s CPU time"<<endl;
  //cerr << transformed_point_2<< endl;
  timer.reset();
  timer.start();
  RigidObject3DTransformation composed_ro3dtrans=compose(ro3dtrans_2,ro3dtrans_1);
  RigidObject3DTransformation composed_ro3dtrans1=compose(ro3dtrans_3,
						  compose(ro3dtrans_2,ro3dtrans_1));
    //composed_ro3dtrans);


  const CartesianCoordinate3D<float> transformed_point_composed =
    composed_ro3dtrans1.transform_point(point);
  timer.stop();
  cerr << "Combined: " <<  timer.value() << " s CPU time"<<endl;
    check_if_equal( transformed_point_3.z(), transformed_point_composed.z(),"test on z");
   check_if_equal( transformed_point_3.y(), transformed_point_composed.y(),"test on y");
   check_if_equal( transformed_point_3.x(), transformed_point_composed.x(),"test on x");
  
  
 




}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()//int argc, char *argv[])
{
  /*if (argc!=3) 
  {
    cerr<<"Usage: " << argv[0] << " filename polaris offset mt_filename\n"
       	<< endl; 
    return EXIT_FAILURE;
  }*/

  RigidObject3DTransformationTests tests;
  tests.run_tests();

 
  //cerr << " Quat   " << quat[1] << " " << quat[2] << " "<<quat[3] << " "<<quat[4] << endl;
  //cerr << " Trans  " << trans.z() << " " << trans.y() << " " << trans.x ()<< endl;
 
  return tests.main_return_value();
}