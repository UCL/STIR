//
//
/*
    Copyright (C) 2003- 2005 , Hammersmith Imanet Ltd
    For GE Internal use only
*/
/*!
\file
\ingroup test

\brief Test program for class Quaternion 
\author Sanida Mustafovic    
*/
#include "stir/RunTests.h"
#include "local/stir/Quaternion.h"
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR
/*!
  \ingroup test
  \brief Test class for Quaternion
*/
class QuaternionTests: public RunTests
{
public:  
  void run_tests();
};


void
QuaternionTests::run_tests()
{
  
  cerr << "Tests for Quaternions\n";
  Quaternion<float> test(0.00525584F, -0.999977F, -0.00166456F, 0.0039961F);
  
  
  cerr << "Testing multiplication with a const" << endl;
  {
    const float factor =5.0;
    Quaternion<float> multiplywithconst =test;
    multiplywithconst *=factor;
    
    check_if_equal( test[1]*factor, multiplywithconst[1],"test on multiplication with a const (operator *=())--scalar");
    check_if_equal( test[2]*factor, multiplywithconst[2],"test on multiplication with a const (operator *=())--vector 1st");
    check_if_equal( test[3]*factor, multiplywithconst[3],"test on multiplication with a const (operator *=())--vector 2nd");
    check_if_equal( test[4]*factor, multiplywithconst[4],"test on multiplication with a const (operator *=())--vector 3rd");
  }
  {
    const float factor =5.0;
    const Quaternion<float> multiplywithconst =test*factor;
    
    check_if_equal( test[1]*factor, multiplywithconst[1],"test on multiplication with a const (operator *())--scalar");
    check_if_equal( test[2]*factor, multiplywithconst[2],"test on multiplication with a const (operator *())--vector 1st");
    check_if_equal( test[3]*factor, multiplywithconst[3],"test on multiplication with a const (operator *())--vector 2nd");
    check_if_equal( test[4]*factor, multiplywithconst[4],"test on multiplication with a const (operator *())--vector 3rd");
  }
  
  cerr << "Testing division by a const" << endl;
  {
    const float factor =5.0;
    Quaternion<float> dividewithconst =test;
    dividewithconst /=factor;
    
    check_if_equal( test[1]/factor, dividewithconst[1],"test on division with a const (operator /=())-- scalar");
    check_if_equal( test[2]/factor, dividewithconst[2],"test on division with a const (operator /=())-- vector 1st");
    check_if_equal( test[3]/factor, dividewithconst[3],"test on division with a const (operator /=())-- vector 2nd");
    check_if_equal( test[4]/factor, dividewithconst[4],"test on division with a const (operator /=())-- vector 3rd");
  }
  {
    const float factor =5.0;
    Quaternion<float> dividewithconst = test/factor;
    
    check_if_equal( test[1]/factor, dividewithconst[1],"test on division with a const (operator /())-- scalar");
    check_if_equal( test[2]/factor, dividewithconst[2],"test on division with a const (operator /())-- vector 1st");
    check_if_equal( test[3]/factor, dividewithconst[3],"test on division with a const (operator /())-- vector 2nd");
    check_if_equal( test[4]/factor, dividewithconst[4],"test on division with a const (operator /())-- vector 3rd");
  }
  cerr << "Testing adding  a const" << endl;
  {
    const float factor =5.0;
    Quaternion<float> result = test+factor;
    
    check_if_equal( test[1]+factor, result[1],"test on addition with a const (operator +())-- scalar");
    check_if_equal( test[2]+factor, result[2],"test on addition with a const (operator +())-- vector 1st");
    check_if_equal( test[3]+factor, result[3],"test on addition with a const (operator +())-- vector 2nd");
    check_if_equal( test[4]+factor, result[4],"test on addition with a const (operator +())-- vector 3rd");
  }
    cerr << "Testing subtracing  a const" << endl;
  {
    const float factor =5.0;
    Quaternion<float> result = test-factor;
    
    check_if_equal( test[1]-factor, result[1],"test on subtraction with a const (operator -())-- scalar");
    check_if_equal( test[2]-factor, result[2],"test on subtraction with a const (operator -())-- vector 1st");
    check_if_equal( test[3]-factor, result[3],"test on subtraction with a const (operator -())-- vector 2nd");
    check_if_equal( test[4]-factor, result[4],"test on subtraction with a const (operator -())-- vector 3rd");
  }
  cerr << "Testing multiplication of two quaternions" << endl;
  {  
    Quaternion<float> first_factor(1, 3, -2, 2);
    Quaternion<float> second_factor(2, 0, -6, 3);
    Quaternion<float> result (-16, 12, -19, -11);
    
    first_factor *=second_factor;
    check_if_equal( result[1], first_factor[1],"test on quaternion multiplication (operator*= (Quaternion& q)) -- scalar");
    check_if_equal( result[2], first_factor[2],"test on quaternion multiplication (operator*= (Quaternion& q)) -- vector 1st");
    check_if_equal( result[3], first_factor[3],"test on quaternion multiplication (operator*= (Quaternion& q)) -- vector 2nd");
    check_if_equal( result[4], first_factor[4],"test on quaternion multiplication (operator*= (Quaternion& q))-- vector 3rd");
  }
  cerr << "Testing addition of two quaternions" << endl;
  {  
    const Quaternion<float> first(1, 3, -2, 2);
    const Quaternion<float> second(2, 0, -6, 3);
    const Quaternion<float> result = first + second;
    check_if_equal( result[1], first[1]+second[1],"test on quaternion addition (operator+ (Quaternion& q)) -- scalar");
    check_if_equal( result[2], first[2]+second[2],"test on quaternion addition (operator+ (Quaternion& q)) -- vector 1st");
    check_if_equal( result[3], first[3]+second[3],"test on quaternion addition (operator+ (Quaternion& q)) -- vector 2nd");
    check_if_equal( result[4], first[4]+second[4],"test on quaternion addition (operator+ (Quaternion& q)) -- vector 3rd");
  }
  cerr << "Testing subtraction of two quaternions" << endl;
  {  
    const Quaternion<float> first(1, 3, -2, 2);
    const Quaternion<float> second(2, 0, -6, 3);
    const Quaternion<float> result = first - second;
    check_if_equal( result[1], first[1]-second[1],"test on quaternion subtraction (operator+ (Quaternion& q)) -- scalar");
    check_if_equal( result[2], first[2]-second[2],"test on quaternion subtraction (operator+ (Quaternion& q)) -- vector 1st");
    check_if_equal( result[3], first[3]-second[3],"test on quaternion subtraction (operator+ (Quaternion& q)) -- vector 2nd");
    check_if_equal( result[4], first[4]-second[4],"test on quaternion subtraction (operator+ (Quaternion& q)) -- vector 3rd");
  }
  
  cerr << "Testing neg_quaternion ()" << endl;
  {
    Quaternion<float> neg_test =test;
    neg_test.neg_quaternion();
    
    check_if_equal( neg_test[1], -1*test[1],"test on neg_quaternion () -- scalar");
    check_if_equal( neg_test[2], -1*test[2],"test on neg_quaternion () -- vector 1st");
    check_if_equal( neg_test[3], -1*test[3],"test on neg_quaternion () -- vector 2nd");
    check_if_equal( neg_test[4], -1*test[4],"test on neg_quaternion () -- vector 3rd");
  }
  cerr << "Testing conjugate()" << endl;
  {
    Quaternion<float> conj_test =test;
    conj_test.conjugate();
    
    check_if_equal( conj_test[1], test[1], "test on conjugate() -- scalar");
    check_if_equal( conj_test[2], -1*test[2],"test on conjugate() -- vector 1st");
    check_if_equal( conj_test[3], -1*test[3],"test on conjugate() -- vector 2nd");
    check_if_equal( conj_test[4], -1*test[4],"test on conjugate() -- vector 3rd");
  }
  
  cerr << "Testing normalise()" << endl;
  {
    const float test_squared = sqrt(square(test[1]) + square(test[2]) +square(test[3])+ square(test[4]));   
    Quaternion<float> test_norm = test;
    test_norm /=test_squared;
    Quaternion<float> norm_test =test;
    norm_test.normalise();
    
    check_if_equal( norm_test[1], test_norm[1], "test on normalise() -- scalar");
    check_if_equal( norm_test[2], test_norm[2], "test on normalise() -- vector 1st");
    check_if_equal( norm_test[3], test_norm[3], "test on normalise() -- vector 2nd");
    check_if_equal( norm_test[4], test_norm[4], "test on normalise() -- vector 3rd");
  }
  
  cerr << "Testing inverse()" << endl;
  {
    const Quaternion<float> test_inverse (1, 3, -2, 2);
    float dot_product =(test_inverse[1]*test_inverse[1])+(test_inverse[2]*test_inverse[2])+(test_inverse[3]*test_inverse[3])+(test_inverse[4]*test_inverse[4]);
    Quaternion<float> result_inverse =test_inverse;
    result_inverse[1] /=dot_product;
    result_inverse[2] *=-1;result_inverse[2] /=dot_product;
    result_inverse[3] *=-1;result_inverse[3] /=dot_product;
    result_inverse[4] *=-1;result_inverse[4] /=dot_product;
    
    Quaternion<float> test_inverse_result =test_inverse;
    test_inverse_result.inverse();
    check_if_equal( test_inverse_result[1], result_inverse[1], "test on inverse() -- scalar");
    check_if_equal( test_inverse_result[2], result_inverse[2], "test on inverse() -- vector 1st");
    check_if_equal( test_inverse_result[3], result_inverse[3], "test on inverse() -- vector 2nd");
    check_if_equal( test_inverse_result[4], result_inverse[4], "test on inverse() -- vector 3rd");

    const Quaternion<float> unit = test_inverse * test_inverse_result;
  
    check_if_equal( unit[1], 1.F, "test on inverse(): multiplication -- scalar");
    check_if_equal( unit[2], 0.F, "test on inverse(): multiplication -- vector 1st");
    check_if_equal( unit[2], 0.F, "test on inverse(): multiplication -- vector 2nd");
    check_if_equal( unit[2], 0.F, "test on inverse(): multiplication -- vector 3rd");
  }
  
}

END_NAMESPACE_STIR



USING_NAMESPACE_STIR

int main()
{
  QuaternionTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
