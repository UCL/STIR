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

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  RigidObject3DTransformationTests tests;
  tests.run_tests();
  return tests.main_return_value();
}