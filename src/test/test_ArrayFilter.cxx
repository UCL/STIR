// $Id$

/*!
  \file 
  \ingroup test
 
  \brief tests for the ArrayFilter classes

  \author Kris Thielemans

  $Date$
  $Revision$

*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/Array.h"
#include "stir/ArrayFilterUsingRealDFTWithPadding.h"
#include "stir/ArrayFilter1DUsingConvolution.h"
#include "stir/ArrayFilter1DUsingConvolutionSymmetricKernel.h"
#include "stir/Succeeded.h"
#include "stir/modulo.h"
#include "stir/RunTests.h"

//#include "stir/stream.h"
#include <iostream>

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
  void compare_results_2arg(const ArrayFunctionObject<1,float>& filter1,
			    const ArrayFunctionObject<1,float>& filter2,
			    const Array<1,float>& test);
  void compare_results_1arg(const ArrayFunctionObject<1,float>& filter1,
			    const ArrayFunctionObject<1,float>& filter2,
			    const Array<1,float>& test);
};


void 
ArrayFilterTests::
compare_results_1arg(const ArrayFunctionObject<1,float>& filter1,
		     const ArrayFunctionObject<1,float>& filter2,
		     const Array<1,float>& test)
{
  {
    Array<1,float> out1(test);
    Array<1,float> out2(out1);
    filter1(out1);
    filter2(out2);
    
    check_if_equal( out1, out2, "test comparing output of filters, equal length");
    //std::cerr << out1 << out2;
  }
  {
    Array<1,float> out1(test);
    out1.resize(test.get_min_index()-5, test.get_max_index()+7);
    Array<1,float> out2(out1);
    filter1(out1);
    filter2(out2);
    
    check_if_equal( out1, out2, "test comparing output of filters, larger length");
    //std::cerr << out1 << out2;
  }
}

void 
ArrayFilterTests::
compare_results_2arg(const ArrayFunctionObject<1,float>& filter1,
		     const ArrayFunctionObject<1,float>& filter2,
		     const Array<1,float>& test)
{
  {
    Array<1,float> out1(test.get_index_range());
    Array<1,float> out2(out1.get_index_range());
    filter1(out1, test);
    filter2(out2, test);
    
    check_if_equal( out1, out2, "test comparing output of filter2, equal length");
    //std::cerr << out1 << out2;
  }
  {
    Array<1,float> out1(IndexRange<1>(200));
    Array<1,float> out2(out1.get_index_range());
    filter1(out1, test);
    filter2(out2, test);
    
    check_if_equal( out1, out2, "test comparing output of filter2, larger length");
    //std::cerr << out1 << out2;
  }
  {
    Array<1,float> out1(IndexRange<1>(50));
    Array<1,float> out2(out1.get_index_range());
    filter1(out1, test);
    filter2(out2, test);
    
    check_if_equal( out1, out2, "test comparing output of filters, smaller length");
  }
  {
    IndexRange<1> influenced_range;
    if (filter2.get_influenced_indices(influenced_range, test.get_index_range())==Succeeded::yes)
      {
	Array<1,float> out1(IndexRange<1>(influenced_range.get_min_index()-3, influenced_range.get_max_index()+4));
	Array<1,float> out2(out1.get_index_range());
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

void
ArrayFilterTests::run_tests()
{ 
  Array<1,float> test(IndexRange<1>(100));
  // initialise to some arbitrary values
  for (int i=test.get_min_index(); i<=test.get_max_index(); ++i)
    test[i]=i*i*2-i-100.F;

  {
    const int kernel_half_length=30;
    const int DFT_kernel_size=256;
    // necessary for avoid aliasing in DFT
    assert(DFT_kernel_size>=kernel_half_length*2*2);
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

    cerr <<"Comparing DFT and Convolution\n";
    compare_results_2arg(DFT_filter, conv_filter, test);
    compare_results_1arg(DFT_filter, conv_filter, test);
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
    cerr <<"Comparing SymmetricConvolution and Convolution\n";
    // note: SymmetricConvolution cannot handle different input and output ranges
    compare_results_1arg(symconv_filter, conv_filter, test);
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
