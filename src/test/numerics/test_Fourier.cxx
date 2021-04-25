//
//

/*!
  \file 
  \ingroup tests
  \ingroup DFT
  \brief Tests for function in the DFT group

  \author Kris Thielemans

*/
/*
    Copyright (C) 2018, University College London
    See STIR/LICENSE.txt for details
*/
#include "stir/VectorWithOffset.h"
#include "stir/Array.h"
#include <complex>
#include "stir/Array_complex_numbers.h"
#include "stir/RunTests.h"
#include "stir/stream.h"
#include "stir/round.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange3D.h"
#include "stir/numerics/norm.h"
#include "stir/numerics/fourier.h"
#include <iostream>
#include <algorithm>


using std::cin;
using std::cout;
using std::endl;

START_NAMESPACE_STIR
#define DOARRAY

#ifdef DOARRAY
  typedef Array<1,std::complex<float> > ArrayC1;
  typedef Array<1,float> ArrayF1;
  typedef Array<2,float> ArrayF2;
  typedef Array<2,std::complex<float> > ArrayC2;
  typedef Array<3,float> ArrayF3;
  typedef Array<3,std::complex<float> > ArrayC3;
#else
  typedef VectorWithOffset<std::complex<float> > ArrayC1;
  typedef VectorWithOffset<float> ArrayF1;
  typedef VectorWithOffset<ArrayC1 > ArrayC2;
#endif




inline float rand1() 
{
  return 2*(rand()-RAND_MAX/2.F)/RAND_MAX;
}

/*!
  \ingroup numerics_test
  \brief A simple class to test the erf and erfc functions.
*/
class FourierTests : public RunTests
{
public:
  FourierTests() 
  = default;
  void run_tests() override;
private:
  template <int num_dimensions>
  void test_single_dimension(const IndexRange<num_dimensions>& index_range);
};

template <int num_dimensions>
void FourierTests::test_single_dimension(const IndexRange<num_dimensions>& index_range)
{
  typedef Array<num_dimensions, std::complex<float> > complex_type;
  typedef Array<num_dimensions, float> real_type;
  complex_type complex_array(index_range);
  real_type real_array(index_range);

  // fill
  {
    for (typename real_type::full_iterator iter= real_array.begin_all();
         iter!=real_array.end_all();
         ++iter)
      *iter= rand1();

    std::copy(real_array.begin_all(), real_array.end_all(), complex_array.begin_all());
  }

  const int sign=1;
  
  complex_type pos_frequencies =
    fourier_for_real_data(real_array, sign);
  const complex_type all_frequencies =
    pos_frequencies_to_all(pos_frequencies);

  fourier(complex_array,sign);
  //cout << pos_frequencies;
  //cout << all_frequencies << complex_array;
  //cout << '\n' << complex_array-all_frequencies;
  complex_array -= all_frequencies;
  cout << "\nReal FT Residual norm "  <<
    norm(complex_array.begin_all(), complex_array.end_all())/norm(real_array.begin_all(), real_array.end_all());

  real_type test_inverse_real =
    inverse_fourier_for_real_data(pos_frequencies,sign);
  //cout <<"\nv,test "<< v << test_inverse_real << test_inverse_real/v;
  test_inverse_real -= real_array;
  cout << "\ninverse Real FT Residual norm "  <<
    norm(test_inverse_real.begin_all(), test_inverse_real.end_all())/norm(real_array.begin_all(), real_array.end_all());

  // fill
  {
    for (typename complex_type::full_iterator iter= complex_array.begin_all();
         iter!=complex_array.end_all();
         ++iter)
      *iter = std::complex<float>(rand1(), rand1());
  }
  const complex_type array_copy(complex_array);

  fourier(complex_array,sign);
  inverse_fourier(complex_array,sign);
  complex_array -= array_copy;
  cout << "\ninverse  FT Residual norm "  <<
    norm(complex_array.begin_all(), complex_array.end_all())/norm(array_copy.begin_all(), array_copy.end_all());
}

void FourierTests::run_tests()
{  
  std::cerr << "Testing Fourier Functions..." << std::endl;

  std::cerr << "... Testing 1D\n";
  test_single_dimension(IndexRange<1>(128));
  std::cerr << "... Testing 2D\n";
  test_single_dimension(IndexRange2D(128,256));
  std::cerr << "... Testing 3D\n";
  test_single_dimension(IndexRange3D(128,256,16));
}

END_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc != 1)
  {
    std::cerr << "Usage : " << argv[0] << " \n";
    return EXIT_FAILURE;
  }
  stir::FourierTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
