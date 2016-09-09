/*
    Copyright (C) 2005, Hammersmith Imanet Ltd
    Copyright (C) 2013, University College London
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
  \ingroup numerics_test
 
  \brief tests for functions in MatrixFunction.h etc.

  \author Kris Thielemans
*/
#include "stir/numerics/max_eigenvector.h"
#include "stir/numerics/norm.h"
#include "stir/make_array.h"
#include "stir/Array_complex_numbers.h"
#include "stir/more_algorithms.h"
#include "stir/Coordinate2D.h"
#include "stir/Coordinate3D.h"
#include "stir/Succeeded.h"
#include "stir/RunTests.h"
#include "stir/stream.h"
#include <complex>
#include <cmath>
# ifdef BOOST_NO_STDC_NAMESPACE
 namespace std { using ::sqrt; using ::sin; using ::cos}
# endif

START_NAMESPACE_STIR


/*!
  \brief Tests MatrixFunction.h functionality
  \ingroup test_numerics

*/
class MatrixTests : public RunTests
{
public:
  void run_tests();
private:
  void run_tests_1D();
  void run_tests_2D();
  void run_tests_max_eigenvector();
  void run_tests_max_eigenvector_non_diagonal();
};

// local function, copied from the Shape library
// takes Euler angles and make an orthogonal matrix
// note that because of STIR conventions, this matrix has determinant -1.
static 
Array<2,float>
make_orthogonal_matrix(const float alpha, const float beta, const float gamma)
{
  return
    make_array(make_1d_array(
			     std::sin(beta)*std::sin(gamma),
			     std::cos(gamma)*std::sin(alpha) + std::cos(alpha)*std::cos(beta)*std::sin(gamma),
			     std::cos(alpha)*std::cos(gamma) - std::cos(beta)*std::sin(alpha)*std::sin(gamma)),
	       make_1d_array(
			     std::cos(gamma)*std::sin(beta),
			     std::cos(alpha)*std::cos(beta)*std::cos(gamma) - std::sin(alpha)*std::sin(gamma),
			     -(std::cos(beta)*std::cos(gamma)*std::sin(alpha)) - std::cos(alpha)*std::sin(gamma)),
	       make_1d_array(
			     std::cos(beta),
			     -(std::cos(alpha)*std::sin(beta)),
			     std::sin(alpha)*std::sin(beta))
	       );
}

void
MatrixTests::
run_tests()
{
  std::cerr << "Testing numerics/MatrixFunction.h functions\n";
//  run_tests_1D();
//  run_tests_2D();
//  run_tests_max_eigenvector();
  run_tests_max_eigenvector_non_diagonal();
}

void
MatrixTests::
run_tests_1D()
{
  std::cerr << "Testing 1D stuff" << std::endl;

  {
    const Array<1,float> v = make_1d_array(1.F,2.F,3.F,-5.F);
    check_if_equal(norm(v),static_cast<double>(std::sqrt(square(1.F)+square(2)+square(3)+square(5))),
		   "norm of float array");
    check_if_equal(inner_product(v,v),square(1.F)+square(2)+square(3)+square(5),
		   "inner_product of float array with itself");
    const Array<1,float> v2 = make_1d_array(7.F,8.F,3.3F,-5.F);
		   check_if_equal(inner_product(v,v2),1.F*7+2*8+3*3.3F+25,
		   "inner_product of float arrays");
  }
  {
    typedef std::complex<float>  complex_t;
    const Array<1,complex_t > v = make_1d_array(complex_t(1.F,0.F),
						complex_t(2.F,-3.F));
    check_if_equal(norm(v),static_cast<double>(std::sqrt(square(1.F)+square(2)+square(3))),
		   "norm of complex array");
    check_if_equal(inner_product(v,v),
		   complex_t(square(1.F)+square(2)+square(3),0.F),
		   "inner_product of complex array with itself");
    const Array<1,complex_t > v2 = make_1d_array(complex_t(1.F,1.F),
						 complex_t(4.F,5.F));
    check_if_equal(inner_product(v,v2),
		   std::conj(v[0])*v2[0]+std::conj(v[1])*v2[1],
		   "inner_product of complex arrays");
  }
}

void
MatrixTests::
run_tests_2D()
{
  std::cerr << "Testing 2D stuff" << std::endl;
  const Array<2,float> m1 = 
    make_array(make_1d_array(3.F,4.F),
	       make_1d_array(5.F,6.F),
	       make_1d_array(1.5F,-4.6F));
  // matrix*vector
  {
    const Array<1,float> v = make_1d_array(1.F,2.F);
    check_if_equal(matrix_multiply(m1,v),
		   make_1d_array(m1[0][0]*v[0]+m1[0][1]*v[1],
				 m1[1][0]*v[0]+m1[1][1]*v[1],
				 m1[2][0]*v[0]+m1[2][1]*v[1]),
		   "matrix times vector");
  }
  // matrix*matrix
  {
    const Array<2,float> m2 = 
      make_array(make_1d_array(1.F,4.3F,6.F,8.F),
		 make_1d_array(-5.F,6.5F,2.F,5.F));    
    check_if_equal(matrix_multiply(m1,m2),
		   make_array(make_1d_array(m1[0][0]*m2[0][0]+m1[0][1]*m2[1][0],
					    m1[0][0]*m2[0][1]+m1[0][1]*m2[1][1],
					    m1[0][0]*m2[0][2]+m1[0][1]*m2[1][2],
					    m1[0][0]*m2[0][3]+m1[0][1]*m2[1][3]),
			      make_1d_array(m1[1][0]*m2[0][0]+m1[1][1]*m2[1][0],
					    m1[1][0]*m2[0][1]+m1[1][1]*m2[1][1],
					    m1[1][0]*m2[0][2]+m1[1][1]*m2[1][2],
					    m1[1][0]*m2[0][3]+m1[1][1]*m2[1][3]),
			      make_1d_array(m1[2][0]*m2[0][0]+m1[2][1]*m2[1][0],
					    m1[2][0]*m2[0][1]+m1[2][1]*m2[1][1],
					    m1[2][0]*m2[0][2]+m1[2][1]*m2[1][2],
					    m1[2][0]*m2[0][3]+m1[2][1]*m2[1][3])),
		   "matrix times matrix");
  }

  // transpose
  {
    const Array<2,float> m1_trans = 
      make_array(make_1d_array(m1[0][0],m1[1][0],m1[2][0]),
		 make_1d_array(m1[0][1],m1[1][1],m1[2][1]));
    check_if_equal(matrix_transpose(m1), m1_trans,
		   "matrix transposition");
  }
  // diagonal_matrix
  {
    const Array<2,float> d = 
      diagonal_matrix(2, 3.F);
    check_if_equal(d, 
		   make_array(make_1d_array(3.F,0.F),
			      make_1d_array(0.F,3.F)),
		   "diagonal_matrix with all diag-elems equal");
    const Array<2,float> d2 = 
      diagonal_matrix(Coordinate2D<float>(3.F,4.F));
    check_if_equal(d2, 
		   make_array(make_1d_array(3.F,0.F),
			      make_1d_array(0.F,4.F)),
		   "diagonal_matrix with differing diag-elems");
  }

}


void
MatrixTests::
run_tests_max_eigenvector()
{
  std::cerr << "Testing max_eigenvector stuff" << std::endl;
  set_tolerance(.01);
  float max_eigenvalue;
  Array<1,float> max_eigenvector;
  {
    {
      const Array<2,float> d = 
	diagonal_matrix(Coordinate3D<float>(3.F,4.F,-2.F));
      Succeeded success =
	absolute_max_eigenvector_using_power_method(max_eigenvalue,
						    max_eigenvector,
						    d, 
						    make_1d_array(1.F,2.F,3.F),
						    /*tolerance=*/ .001,
						    1000UL);
      check(success == Succeeded::yes, 
	    "abs_max_using_power: succeeded (float diagonal matrix)");
      
      check_if_equal(max_eigenvalue, 4.F, 
		     "abs_max_using_power: eigenvalue (float diagonal matrix)");
      check_if_equal(max_eigenvector, make_1d_array(0.F,1.F,0.F), 
		     "abs_max_using_power: eigenvector (float diagonal matrix)");
      success =
	absolute_max_eigenvector_using_shifted_power_method(max_eigenvalue,
							    max_eigenvector,
							    d, 
							    make_1d_array(1.F,2.F,3.F),
							    .5F, // note: shift should be small enough that it doesn't make the most negative eigenvalue 'larger'
							    /*tolerance=*/ .001,
							    1000UL);
      check(success == Succeeded::yes, 
	    "abs_max_using_shifted_power: succeeded (float diagonal matrix)");
      
      check_if_equal(max_eigenvalue, 4.F, 
		     "abs_max_using_shifted_power: eigenvalue (float diagonal matrix)");
      check_if_equal(max_eigenvector, make_1d_array(0.F,1.F,0.F), 
		     "abs_max_using_shifted_power: eigenvector (float diagonal matrix)");

      success =
	max_eigenvector_using_power_method(max_eigenvalue,
					   max_eigenvector,
					   d, 
					   make_1d_array(1.F,-2.F,-3.F),
					   /*tolerance=*/ .001,
					   1000UL);
      check(success == Succeeded::yes, 
	    "max_using_power: succeeded (float diagonal matrix)");
      
      check_if_equal(max_eigenvalue, 4.F, 
		     "max_using_power: eigenvalue (float diagonal matrix)");
      check_if_equal(max_eigenvector, make_1d_array(0.F,1.F,0.F), 
		     "max_using_power: eigenvector (float diagonal matrix)");

      success =
	max_eigenvector_using_power_method(max_eigenvalue,
					   max_eigenvector,
					   Array<2,float>(d*(-1.F)), 
					   make_1d_array(1.F,2.F,3.F),
					   /*tolerance=*/ .001,
					   1000UL);
      check(success == Succeeded::yes, 
	    "max_using_power: succeeded (float diagonal matrix with large negative value)");
      
      check_if_equal(max_eigenvalue, 2.F, 
		     "max_using_power: eigenvalue (float diagonal matrix with large negative value)");
      check_if_equal(max_eigenvector, make_1d_array(0.F,0.F,1.F), 
		     "max_using_power: eigenvector (float diagonal matrix with large negative value)");
    }
  }

  {
    const float pi2=static_cast<float>(_PI/2);
    const Array<2,float> rotation =
      //      make_orthogonal_matrix(.2F,.4F,-1.F);
      make_orthogonal_matrix(pi2,pi2,pi2);
    std::cerr << rotation;
    check_if_equal(matrix_multiply(rotation, matrix_transpose(rotation)),
		   diagonal_matrix(3,1.F),
		   "construct orthogonal matrix O.O^t");
    check_if_equal(matrix_multiply(matrix_transpose(rotation), rotation),
		   diagonal_matrix(3,1.F),
		   "construct orthogonal matrix O^t.O");

    const Array<2,float> d = 
      diagonal_matrix(Coordinate3D<float>(3.F,4.F,-2.F));

    const Array<2,float> m =
      matrix_multiply(rotation, matrix_multiply(d, matrix_transpose(rotation)));
    Array<1,float> the_max_eigenvector =
      matrix_multiply(rotation,make_1d_array(0.F,1.F,0.F));
    the_max_eigenvector /= (*abs_max_element(the_max_eigenvector.begin(), the_max_eigenvector.end()));

    // now repetition of tests with diagonal matrix
    Succeeded success =
	absolute_max_eigenvector_using_power_method(max_eigenvalue,
						    max_eigenvector,
						    m, 
						    make_1d_array(1.F,2.F,3.F),
						    /*tolerance=*/ .001,
						    1000UL);
      check(success == Succeeded::yes, 
	    "abs_max_using_power: succeeded (float non-diagonal matrix)");
      
      check_if_equal(max_eigenvalue, 4.F, 
		     "abs_max_using_power: eigenvalue (float non-diagonal matrix)");
      check_if_equal(max_eigenvector, the_max_eigenvector, 
		     "abs_max_using_power: eigenvector (float non-diagonal matrix)");
      success =
	absolute_max_eigenvector_using_shifted_power_method(max_eigenvalue,
							    max_eigenvector,
							    m, 
							    make_1d_array(1.F,2.F,3.F),
							    .5F, // note: shift should be small enough that it doesn't make the most negative eigenvalue 'larger'
							    /*tolerance=*/ .001,
							    1000UL);
      check(success == Succeeded::yes, 
	    "abs_max_using_shifted_power: succeeded (float non-diagonal matrix)");
      
      check_if_equal(max_eigenvalue, 4.F, 
		     "abs_max_using_shifted_power: eigenvalue (float non-diagonal matrix)");
      check_if_equal(max_eigenvector, the_max_eigenvector, 
		     "abs_max_using_shifted_power: eigenvector (float non-diagonal matrix)");

      success =
	max_eigenvector_using_power_method(max_eigenvalue,
					   max_eigenvector,
					   m, 
					   make_1d_array(1.F,-2.F,-3.F),
					   /*tolerance=*/ .001,
					   1000UL);
      check(success == Succeeded::yes, 
	    "max_using_power: succeeded (float non-diagonal matrix)");
      
      check_if_equal(max_eigenvalue, 4.F, 
		     "max_using_power: eigenvalue (float non-diagonal matrix)");
      check_if_equal(max_eigenvector, the_max_eigenvector, 
		     "max_using_power: eigenvector (float non-diagonal matrix)");

      success =
	max_eigenvector_using_power_method(max_eigenvalue,
					   max_eigenvector,
					   Array<2,float>(m*(-1.F)), 
					   make_1d_array(1.F,2.F,3.F),
					   /*tolerance=*/ .001,
					   1000UL);
      check(success == Succeeded::yes, 
	    "max_using_power: succeeded (float non-diagonal matrix with large negative value)");
      
      check_if_equal(max_eigenvalue, 2.F, 
		     "max_using_power: eigenvalue (float non-diagonal matrix with large negative value)");
      check_if_equal(max_eigenvector, matrix_multiply(rotation,make_1d_array(0.F,0.F,1.F)), 
		     "max_using_power: eigenvector (float non-diagonal matrix with large negative value)");
  }

  {
    // now test for a case where the power-method fails
      const Array<2,float> d = 
	diagonal_matrix(Coordinate2D<float>(3.F,-3.F));
      Succeeded success =
	absolute_max_eigenvector_using_power_method(max_eigenvalue,
						    max_eigenvector,
						    d, 
						    make_1d_array(1.F,2.F),
						    /*tolerance=*/ .001,
						    100UL);
      check(success == Succeeded::no, 
	    "abs_max_using_power should have failed (float diagonal matrix with opposite max eigenvalues)");
  }
}

void
MatrixTests::
run_tests_max_eigenvector_non_diagonal()
{
    std::cout << "Eigenvector and eigenvalue calculation without the CoV matrix." << std::endl;
    set_tolerance(.01);

    // Initial array
    Array<2, float> v = make_array(make_1d_array(5.F, 0.F, 3.F, 7.F),
                                   make_1d_array(1.F, -5.F, 7.F, 3.F),
                                   make_1d_array(4.F, 9.F, 8.F, 10.F));

    const int m_min_row = v.get_min_index();
    const int m_max_row = v.get_max_index();
    const int m_min_col = v[m_min_row].get_min_index();
    const int m_max_col = v[m_min_row].get_max_index();

    // remove the mean value of each column
    float mean = 0.0f;

    for(int i=m_min_col; i<=m_max_col; ++i)
    {
        for (int k = m_min_row; k<= m_max_row; ++k)
            mean += static_cast<float>(v[k][i]);
        mean /= static_cast<float>(m_max_row+1);

        for (int k = m_min_row; k<= m_max_row; ++k)
            v[k][i] -= mean;
        mean = 0.0f;
    }

    // CoV array of the aforementioned array as calculated by matlab.
    Array<2, float> check_cov = make_array(
                make_1d_array(4.3333F, 8.8333F, -3.0000F, 5.6667F),
                make_1d_array(8.8333F, 50.3333F, 6.5000F, 24.1667F),
                make_1d_array(-3.0000F, 6.5000F, 7.0000F, 1.0000F),
                make_1d_array(5.6667F, 24.1667F, 1.0000F, 12.3333F));

    float chk_max_eigenvalue = 0.f;
    Array<1,float> chk_max_eigenvector;

    Succeeded chk_eigen_success =
            absolute_max_eigenvector_using_power_method(chk_max_eigenvalue,
                                                        chk_max_eigenvector,
                                                        check_cov,
                                                        make_1d_array(1.F,1.F,1.F, 1.F),
                                                        /*tolerance=*/ .001,
                                                        1000UL);

    check(chk_eigen_success == Succeeded::yes,
          "absolute_max_eigenvector_using_power_method: succeeded (float diagonal matrix)");

    float max_eigenvalue = 0.f;
    Array<1,float> max_eigenvector;

    Succeeded eigen_success =
            absolute_max_eigenvector_using_power_method(max_eigenvalue,
                                                        max_eigenvector,
                                                        v,
                                                        make_1d_array(1.F,1.F,1.F, 1.F),
                                                        /*tolerance=*/ .001,
                                                        1000UL,
                                                        false);
    check(eigen_success == Succeeded::yes,
          "absolute_max_eigenvector_using_power_method: succeeded (without float diagonal matrix)");

    check_if_equal(chk_max_eigenvalue, max_eigenvalue, "Eigenvalues do not match");
    check_if_equal(chk_max_eigenvector, max_eigenvector, "Eigenvectors do not match");
}

END_NAMESPACE_STIR


int main()
{
  stir::MatrixTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
