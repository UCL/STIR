
#include "local/stir/SeparableLowPassArrayFilter.h"
#include "stir/ArrayFilter1DUsingConvolution.h"


#include <iostream>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ios;
using std::fstream;
using std::iostream;
using std::cerr;
using std::endl;
#endif
/*
    Copyright (C) 2000- 2002, IRSL
    See STIR/LICENSE.txt for details
*/


START_NAMESPACE_STIR

template <int num_dimensions, typename elemT>
SeparableLowPassArrayFilter<num_dimensions,elemT>::
SeparableLowPassArrayFilter()
{
 for (int i=1;i<=num_dimensions;i++)
  {
    all_1d_array_filters[i-1] = 	 
      new ArrayFilter1DUsingConvolution<float>();
  }
}


template <int num_dimensions, typename elemT> 
SeparableLowPassArrayFilter<num_dimensions,elemT>::
SeparableLowPassArrayFilter(const VectorWithOffset<elemT>& filter_coefficients_v, int z_trivial)
:filter_coefficients(filter_coefficients_v)
{

  cerr << "Printing filter coefficients" << endl;
  for (int i =filter_coefficients_v.get_min_index();i<=filter_coefficients_v.get_max_index();i++)    
    cerr  << i<<"   "<< filter_coefficients_v[i] <<"   " << endl;
   //err << " Z_TRIVIAL "  << z_trivial << endl;
    
 if (!z_trivial) 
 {
 for (int i=1;i<=num_dimensions;i++)
  {
   //cerr << "in the right loop" << endl;
    all_1d_array_filters[i-1] = 	 
      new ArrayFilter1DUsingConvolution<float>(filter_coefficients_v);
  }
 }
 else
 {
   for (int i=2;i<=num_dimensions;i++)
  {
     //cerr << "in the wrong loop" << endl;
    all_1d_array_filters[i-1] = 	 
      new ArrayFilter1DUsingConvolution<float>(filter_coefficients_v);
      //new ArrayFilter1DUsingConvolutionSymmetricKernel<float>(filter_coefficients_v);
  }
   all_1d_array_filters[0] = 	 
    new ArrayFilter1DUsingConvolution<float>();
    
 }
   
}

template SeparableLowPassArrayFilter<3,float>;

END_NAMESPACE_STIR
