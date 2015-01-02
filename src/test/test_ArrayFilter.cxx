
/*!
  \file 
  \ingroup test
 
  \brief tests for the stir::ArrayFilter classes

  \author Kris Thielemans


*/
/*
    Copyright (C) 2004- 2009, Hammersmith Imanet Ltd
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

#include "stir/Array.h"
#include "stir/ArrayFilterUsingRealDFTWithPadding.h"
#include "stir/ArrayFilter1DUsingConvolution.h"
#include "stir/ArrayFilter1DUsingConvolutionSymmetricKernel.h"
#include "stir/ArrayFilter2DUsingConvolution.h"
#include "stir/IndexRange2D.h"
#include "stir/ArrayFilter3DUsingConvolution.h"
#include "stir/IndexRange3D.h"
#include "stir/Succeeded.h"
#include "stir/modulo.h"
#include "stir/RunTests.h"

#include "stir/stream.h"//XXX
#include <iostream>
#include <algorithm>

#ifdef DO_TIMINGS
#include "stir/CPUTimer.h"
#endif

START_NAMESPACE_STIR


/*!
  \brief Tests Array functionality
  \ingroup test

*/
class ArrayFilterTests : public RunTests
{
public:
  void run_tests();
private:



template <int num_dimensions>
void 
compare_results_1arg(const ArrayFunctionObject<num_dimensions,float>& filter1,
		     const ArrayFunctionObject<num_dimensions,float>& filter2,
		     const Array<num_dimensions,float>& test)
{
  {
    Array<num_dimensions,float> out1(test);
    Array<num_dimensions,float> out2(out1);
    filter1(out1);
    filter2(out2);
    
    check_if_equal( out1, out2, "test comparing output of filters, equal length");
    //std::cerr << out1 << out2;
  }
  {
    Array<num_dimensions,float> out1(test);
    BasicCoordinate<num_dimensions, int> min_indices, max_indices;
    check(test.get_regular_range(min_indices, max_indices), "test only works for Arrays of regular range");
    const IndexRange<num_dimensions> larger_range(min_indices-2, max_indices+1);
    out1.resize(larger_range);

    Array<num_dimensions,float> out2(out1);
    filter1(out1);
    filter2(out2);
    
    if (!check_if_equal( out1, out2, "test comparing output of filters, larger length"))
      {}//std::cerr << out1 << out2;
  }
}

template <int num_dimensions>
void 
compare_results_2arg(const ArrayFunctionObject<num_dimensions,float>& filter1,
		     const ArrayFunctionObject<num_dimensions,float>& filter2,
		     const Array<num_dimensions,float>& test)
{
  BasicCoordinate<num_dimensions, int> min_indices, max_indices;
  check(test.get_regular_range(min_indices, max_indices), "test only works for Arrays of regular range");
  {
    Array<num_dimensions,float> out1(test.get_index_range());
    Array<num_dimensions,float> out2(out1.get_index_range());
    filter1(out1, test);
    filter2(out2, test);
    
    check_if_equal( out1, out2, "test comparing output of filter2, equal length");
    //std::cerr << out1 << out2;
  }
  {
    const IndexRange<num_dimensions> larger_range(min_indices-2, max_indices+1);
    Array<num_dimensions,float> out1(larger_range);
    Array<num_dimensions,float> out2(larger_range);
    filter1(out1, test);
    filter2(out2, test);
    
    check_if_equal( out1, out2, "test comparing output of filter2, larger length");
    //std::cerr << out1 << out2;
  }
  {
    const IndexRange<num_dimensions> smaller_range(min_indices+2, max_indices-1);
    Array<num_dimensions,float> out1(smaller_range);
    Array<num_dimensions,float> out2(smaller_range);
    filter1(out1, test);
    filter2(out2, test);
    
    check_if_equal( out1, out2, "test comparing output of filters, smaller length");
  }
  if (num_dimensions==1)
  {
    IndexRange<num_dimensions> influenced_range;
    if (filter2.get_influenced_indices(influenced_range, test.get_index_range())==Succeeded::yes)
      {
	BasicCoordinate<num_dimensions, int> min_indices, max_indices;
	check(influenced_range.get_regular_range(min_indices, max_indices), "test only works for Arrays of regular range");
	const IndexRange<num_dimensions> larger_range(min_indices-3, max_indices+4);// WARNING ALIASING +7
	//Array<num_dimensions,float> out1(IndexRange<num_dimensions>(influenced_range.get_min_index()-3, influenced_range.get_max_index()+4));
	Array<num_dimensions,float> out1(larger_range);
	Array<num_dimensions,float> out2(out1.get_index_range());
	filter1(out1, test);
	filter2(out2, test);
    
	check_if_equal( out1, out2, "test comparing output of filters, out range is in range+ kernel + extra");
	check_if_zero( out2[out2.get_min_index()], "test conv 0 beyond kernel length");
	check_if_zero( out2[out2.get_min_index()+1], "test conv 0 beyond kernel length");
	check_if_zero( out2[out2.get_min_index()+2], "test conv 0 beyond kernel length");
	check_if_zero( out2[out2.get_max_index()], "test conv 0 beyond kernel length");
	check_if_zero( out2[out2.get_max_index()-1], "test conv 0 beyond kernel length");
	check_if_zero( out2[out2.get_max_index()-2], "test conv 0 beyond kernel length");
	check_if_zero( out2[out2.get_max_index()-3], "test conv 0 beyond kernel length");

	// really not necessary if above tests were ok,
	// but in case they failed, this gives some extra info
	check_if_zero( out1[out1.get_min_index()], "test DFT 0 beyond kernel length");
	check_if_zero( out1[out1.get_min_index()+1], "test DFT 0 beyond kernel length");
	check_if_zero( out1[out1.get_min_index()+2], "test DFT 0 beyond kernel length");
	check_if_zero( out1[out1.get_max_index()], "test DFT 0 beyond kernel length");
	check_if_zero( out1[out1.get_max_index()-1], "test DFT 0 beyond kernel length");
	check_if_zero( out1[out1.get_max_index()-2], "test DFT 0 beyond kernel length");
	check_if_zero( out1[out1.get_max_index()-3], "test DFT 0 beyond kernel length");
	//std::cerr << out1 << out2;
      }
  }
}

};
void
ArrayFilterTests::run_tests()
{ 
  std::cerr << "\nTesting 1D\n";
  {
    const int size1 = 100;
    Array<1,float> test(IndexRange<1>(100));// warning: not using 'size1' here. gcc 3.3 fails to compile it otherwise
    Array<1,float> test_neg_offset(IndexRange<1>(-10,size1-11));
    Array<1,float> test_pos_offset(IndexRange<1>(10,size1+9));
    // initialise to some arbitrary values
    for (int i=test.get_min_index(); i<=test.get_max_index(); ++i)
      test[i]=i*i*2-i-100.F;
    std::copy(test.begin(), test.end(), test_neg_offset.begin());
    std::copy(test.begin(), test.end(), test_pos_offset.begin());

    {
      const int kernel_half_length=30;
      const int DFT_kernel_size=256;
      // necessary to avoid aliasing in DFT
      assert(DFT_kernel_size>=(kernel_half_length*2+1)*2);
      assert(DFT_kernel_size>=2*size1+3);// note +3 as test grows the array
      Array<1,float> kernel_for_DFT(IndexRange<1>(0,DFT_kernel_size-1));
      Array<1,float> kernel_for_conv(IndexRange<1>(-kernel_half_length,kernel_half_length));
      for (int i=-kernel_half_length; i<kernel_half_length; ++i)
	{
	  kernel_for_conv[i] = i*i-3*i+1.F;
	  kernel_for_DFT[modulo(i,DFT_kernel_size)] =
	    kernel_for_conv[i];
	}
  
  
      ArrayFilterUsingRealDFTWithPadding<1,float> DFT_filter;
      check(DFT_filter.set_kernel(kernel_for_DFT)==Succeeded::yes, "initialisation DFT filter");
      ArrayFilter1DUsingConvolution<float> conv_filter(kernel_for_conv);

      check(!DFT_filter.is_trivial(), "DFT is_trivial");
      check(!conv_filter.is_trivial(), "conv is_trivial");
      set_tolerance(test.find_max()*kernel_for_conv.sum()*1.E-6);
      //std::cerr << get_tolerance();

      std::cerr <<"Comparing DFT and Convolution with input offset 0\n";
      compare_results_2arg(DFT_filter, conv_filter, test);
      compare_results_1arg(DFT_filter, conv_filter, test);
      std::cerr <<"Comparing DFT and Convolution with input negative offset\n";
      compare_results_2arg(DFT_filter, conv_filter, test_neg_offset);
      compare_results_1arg(DFT_filter, conv_filter, test_neg_offset);
      std::cerr <<"Comparing DFT and Convolution with input positive offset\n";
      compare_results_2arg(DFT_filter, conv_filter, test_pos_offset);
      compare_results_1arg(DFT_filter, conv_filter, test_pos_offset);
    }
    {
      const int kernel_half_length=30;
      Array<1,float> kernel_for_symconv(IndexRange<1>(0,kernel_half_length));
      Array<1,float> kernel_for_conv(IndexRange<1>(-kernel_half_length,kernel_half_length));
      for (int i=0; i<kernel_half_length; ++i)
	{
	  kernel_for_symconv[i] =
	    kernel_for_conv[i] = 
	    kernel_for_conv[-i] = i*i-3*i+1.F;
	}

      // symmetric convolution currently requires equal in and out range
      Array<1,float> test(IndexRange<1>(100));
      // initialise to some arbitrary values
      for (int i=test.get_min_index(); i<=test.get_max_index(); ++i)
	test[i]=i*i*2-i-100.F;
    
  
  
      ArrayFilter1DUsingConvolution<float> conv_filter(kernel_for_conv);
      ArrayFilter1DUsingConvolutionSymmetricKernel<float> symconv_filter(kernel_for_symconv);

      check(!symconv_filter.is_trivial(), "symconv is_trivial");
      check(!conv_filter.is_trivial(), "conv is_trivial");
      set_tolerance(test.find_max()*kernel_for_conv.sum()*1.E-6);
      std::cerr <<"Comparing SymmetricConvolution and Convolution\n";
      // note: SymmetricConvolution cannot handle different input and output ranges
      compare_results_1arg(symconv_filter, conv_filter, test);
    }
    std::cerr << "Testing boundary conditions\n";
    {
      Array<1,float> kernel(IndexRange<1>(-1,2));
      kernel[-1] = 1;
      kernel[0] = 2;
      kernel[1] = 3;
      kernel[2] = .5;
      const ArrayFilter1DUsingConvolution<float> conv_filter_zero_BC(kernel);
      const ArrayFilter1DUsingConvolution<float> conv_filter_cst_BC(kernel, BoundaryConditions::constant);
      {      
	Array<1,float> test(IndexRange<1>(101));
	// initialise to some arbitrary values
	for (int i=test.get_min_index(); i<=test.get_max_index(); ++i)
	  test[i]=i*i*2-i-100.F;
	set_tolerance(test.find_max()*kernel.sum()*1.E-6);
      
	Array<1,float> out_zero_BCs = test;
	Array<1,float> out_cst_BCs = test;
	conv_filter_zero_BC(out_zero_BCs);
	conv_filter_cst_BC(out_cst_BCs);
	// test if internal array elements are the same
	{
	  Array<1,float> out_zero_BCs_small=out_zero_BCs;
	  out_zero_BCs_small.resize(out_zero_BCs.get_min_index() + kernel.get_max_index(),
				    out_zero_BCs.get_max_index() + kernel.get_min_index());
	  Array<1,float> out_cst_BCs_small=out_cst_BCs;
	  out_cst_BCs_small.resize(out_cst_BCs.get_min_index() + kernel.get_max_index(),
				   out_cst_BCs.get_max_index() + kernel.get_min_index());
	  check_if_equal(out_cst_BCs_small, out_zero_BCs_small, "comparing 1D with different boundary conditions: internal values");
	}
	// edge
	float left_boundary=test[0]*kernel[2] + test[0]*kernel[1] + test[0]*kernel[0] + test[1]*kernel[-1];
	check_if_equal(out_cst_BCs[0], left_boundary, "1D with cst BC: left edge");
	left_boundary=test[0]*kernel[0] + test[1]*kernel[-1];
	check_if_equal(out_zero_BCs[0], left_boundary, "1D with zero BC: left edge");
	float right_boundary=test[98]*kernel[2] + test[99]*kernel[1] + test[100]*kernel[0] + test[100]*kernel[-1];
	check_if_equal(out_cst_BCs[100], right_boundary, "1D with cst BC: right edge");
	right_boundary=test[98]*kernel[2] + test[99]*kernel[1] + test[100]*kernel[0];
	check_if_equal(out_zero_BCs[100], right_boundary, "1D with zero BC: right edge");
      }
      {
	Array<1,float> test(-2,5), test_out(-5,8);
	test.fill(1.F);
	set_tolerance(test.find_max()*kernel.sum()*1.E-6);
	conv_filter_zero_BC(test_out, test);
	check_if_equal(test_out[-5],0.F,"1D with zero BC: element -5");
	check_if_equal(test_out[-4],0.F,"1D with zero BC: element -4");
	check_if_equal(test_out[-3],kernel[-1],"1D with zero BC: element -3");
	check_if_equal(test_out[-2],kernel[-1]+kernel[0],"1D with zero BC: element -2");
	check_if_equal(test_out[-1],kernel[-1]+kernel[0]+kernel[1],"1D with zero BC: element -1");
	check_if_equal(test_out[0],kernel[2]+kernel[1]+kernel[0]+kernel[-1],"1D with zero BC: element 0");
	check_if_equal(test_out[4],kernel[2]+kernel[1]+kernel[0]+kernel[-1],"1D with zero BC: element 4");
	check_if_equal(test_out[5],kernel[2]+kernel[1]+kernel[0],"1D with zero BC: element 5");
	check_if_equal(test_out[6],kernel[2]+kernel[1],"1D with zero BC: element 6");	
	check_if_equal(test_out[7],kernel[2],"1D with zero BC: element 7");
	check_if_equal(test_out[8],0.F,"1D with zero BC: element 8");
	conv_filter_cst_BC(test_out, test);
	const float sum=kernel.sum();
	check_if_equal(test_out[-5],sum,"1D with cst BC: element -5");
	check_if_equal(test_out[-4],sum,"1D with cst BC: element -4");
	check_if_equal(test_out[-3],sum,"1D with cst BC: element -3");
	check_if_equal(test_out[-2],sum,"1D with cst BC: element -2");
	check_if_equal(test_out[-1],sum,"1D with cst BC: element -1");
	check_if_equal(test_out[0],sum,"1D with cst BC: element 0");
	check_if_equal(test_out[4],sum,"1D with cst BC: element 4");
	check_if_equal(test_out[5],sum,"1D with cst BC: element 5");
	check_if_equal(test_out[6],sum,"1D with cst BC: element 6");
	check_if_equal(test_out[7],sum,"1D with cst BC: element 7");
	check_if_equal(test_out[8],sum,"1D with cst BC: element 8");
      }
    } // boundary conditions


  } // 1D

  std::cerr << "\nTesting 2D\n";
  {
    set_tolerance(.001F);
    const int size1=6;const int size2=20;
    Array<2,float> test(IndexRange2D(size1,size2));
    Array<2,float> test_neg_offset(IndexRange2D(-5,size1-6,-10,size2-11));
    Array<2,float> test_pos_offset(IndexRange2D(1,size1,2,size2+1));
    // initialise to some arbitrary values
    {
      Array<2,float>::full_iterator iter = test.begin_all();
      /*for (int i=-100; iter != test.end_all(); ++i, ++iter)
       *iter = 1;//i*i*2.F-i-100.F;*/
      test[0][0]=1;
      std::copy(test.begin_all(), test.end_all(), test_neg_offset.begin_all());
      std::copy(test.begin_all(), test.end_all(), test_pos_offset.begin_all());
    }
    {
      const int kernel_half_length=14;
      const int DFT_kernel_size=64;
      // necessary to avoid aliasing in DFT
      assert(DFT_kernel_size>=(kernel_half_length*2+1)*2);
      assert(DFT_kernel_size>=2*size2+3);// note +3 as test grows the array
      assert(DFT_kernel_size>=2*size1+3);// note +3 as test grows the array
      const Coordinate2D<int> sizes(DFT_kernel_size/2,DFT_kernel_size);
      Array<2,float> kernel_for_DFT(IndexRange2D(DFT_kernel_size/2,DFT_kernel_size));
      Array<2,float> kernel_for_conv(IndexRange2D(-(kernel_half_length/2),kernel_half_length/2,
						  -kernel_half_length,kernel_half_length));
      for (int i=-(kernel_half_length/2); i<kernel_half_length/2; ++i)
	for (int j=-kernel_half_length; j<kernel_half_length; ++j)
	{
	  const Coordinate2D<int> index(i,j);
	  kernel_for_conv[index] = i*i-3*i+1.F+j*i/20.F;
	  kernel_for_DFT[modulo(index,sizes)] =
	    kernel_for_conv[index];
	}
  
  
      ArrayFilterUsingRealDFTWithPadding<2,float> DFT_filter;
      check(DFT_filter.set_kernel(kernel_for_DFT)==Succeeded::yes, "initialisation DFT filter");
      ArrayFilter2DUsingConvolution<float> conv_filter(kernel_for_conv);

      check(!DFT_filter.is_trivial(), "DFT is_trivial");
      check(!conv_filter.is_trivial(), "conv is_trivial");
      set_tolerance(test.find_max()*kernel_for_conv.sum()*1.E-6);
      //std::cerr << get_tolerance();

      std::cerr <<"Comparing DFT and Convolution with input offset 0\n";
      compare_results_2arg(DFT_filter, conv_filter, test);
      compare_results_1arg(DFT_filter, conv_filter, test);
      std::cerr <<"Comparing DFT and Convolution with input negative offset\n";
      compare_results_2arg(DFT_filter, conv_filter, test_neg_offset);
      compare_results_1arg(DFT_filter, conv_filter, test_neg_offset);
      std::cerr <<"Comparing DFT and Convolution with input positive offset\n";
      compare_results_2arg(DFT_filter, conv_filter, test_pos_offset);
      compare_results_1arg(DFT_filter, conv_filter, test_pos_offset);
    }
  }
  std::cerr << "\nTesting 3D\n";
  {
    set_tolerance(.001F);
    const int size1=5;const int size2=7; const int size3=6;
    Array<3,float> test(IndexRange3D(size1,size2,size3));
    Array<3,float> test_neg_offset(IndexRange3D(-5,size1-6,-10,size2-11,-4,size3-5));
    Array<3,float> test_pos_offset(IndexRange3D(1,size1,2,size2+1,3,size3+4));
    // initialise to some arbitrary values
    {
      Array<3,float>::full_iterator iter = test.begin_all();
      for (int i=-100; iter != test.end_all(); ++i, ++iter)
	*iter = 1;//i*i*2.F-i-100.F;
      std::copy(test.begin_all(), test.end_all(), test_neg_offset.begin_all());
      std::copy(test.begin_all(), test.end_all(), test_pos_offset.begin_all());
    }
    {
      const int kernel_half_length=7;
      const int DFT_kernel_size=32;
      // necessary to avoid aliasing in DFT
      assert(DFT_kernel_size>=(kernel_half_length*2+1)*2);
      assert(DFT_kernel_size/2>=2*size1+3);// note +3 as test grows the array
      assert(DFT_kernel_size>=2*size2+3);// note +3 as test grows the array
      assert(DFT_kernel_size>=2*size3+3);// note +3 as test grows the array
      const Coordinate3D<int> sizes(DFT_kernel_size/2,DFT_kernel_size,DFT_kernel_size);
      Array<3,float> kernel_for_DFT(IndexRange3D(DFT_kernel_size/2,DFT_kernel_size,DFT_kernel_size));
      Array<3,float> kernel_for_conv(IndexRange3D(-(kernel_half_length/2),kernel_half_length/2,
						  -kernel_half_length,kernel_half_length,
						  -kernel_half_length,kernel_half_length));
      for (int i=-(kernel_half_length/2); i<kernel_half_length/2; ++i)
	for (int j=-kernel_half_length; j<kernel_half_length; ++j)
	  for (int k=-kernel_half_length; k<kernel_half_length; ++k)
	{
	  Coordinate3D<int> index(i,j,k);
	  kernel_for_conv[index] = i*i-3*i+1.F+j*i/20.F+k;
	  kernel_for_DFT[modulo(index,sizes)] =
	    kernel_for_conv[index];
	}
  
  
      ArrayFilterUsingRealDFTWithPadding<3,float> DFT_filter;
      check(DFT_filter.set_kernel(kernel_for_DFT)==Succeeded::yes, "initialisation DFT filter");
      ArrayFilter3DUsingConvolution<float> conv_filter(kernel_for_conv);

      check(!DFT_filter.is_trivial(), "DFT is_trivial");
      check(!conv_filter.is_trivial(), "conv is_trivial");
      set_tolerance(test.find_max()*kernel_for_conv.sum()*1.E-6);
      //std::cerr << get_tolerance();

      std::cerr <<"Comparing DFT and Convolution with input offset 0\n";
      compare_results_2arg(DFT_filter, conv_filter, test);
      compare_results_1arg(DFT_filter, conv_filter, test);
      std::cerr <<"Comparing DFT and Convolution with input negative offset\n";
      compare_results_2arg(DFT_filter, conv_filter, test_neg_offset);
      compare_results_1arg(DFT_filter, conv_filter, test_neg_offset);
      std::cerr <<"Comparing DFT and Convolution with input positive offset\n";
      compare_results_2arg(DFT_filter, conv_filter, test_pos_offset);
      compare_results_1arg(DFT_filter, conv_filter, test_pos_offset);
    }
  }

}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  ArrayFilterTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
