//
//
/*!
  \file
  \ingroup test
  \brief 

  \author Kris Thielemans
*/
/*
    Copyright (C) 2000- 2011, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_SeparableLowPassArrayFilter2__H__
#define __stir_SeparableLowPassArrayFilter2__H__

#include "stir/Array.h"
#include "stir/BasicCoordinate.h"
#include "stir/ArrayFunctionObject_2ArgumentImplementation.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#include "stir/VectorWithOffset.h"
#include "stir/ArrayFunction.h"
#include <iostream>

#include "stir/ArrayFilter1DUsingConvolution.h"

START_NAMESPACE_STIR
/*!
  \ingroup buildblock
  \brief This class implements an \c n -dimensional ArrayFunctionObject whose operation
  is separable.

  'Separable' means that its operation consists of \c n 1D operations, one on each
  index of the \c n -dimensional array. 
  \see in_place_apply_array_functions_on_each_index()

  TODO
  
 */
// TODO don't use 2Argument, but do 1arg explicitly as well
template <int num_dimensions, typename elemT>
class SeparableArrayFunctionObject2 : 
   public ArrayFunctionObject_2ArgumentImplementation<num_dimensions,elemT>
{
public:
  SeparableArrayFunctionObject2 ();
  SeparableArrayFunctionObject2 (const VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >&); 
  bool is_trivial() const;
  // TODO reimplement get_*indices
protected:
 
  VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > > all_1d_array_filters;
  virtual void do_it(Array<num_dimensions,elemT>& out_array, const Array<num_dimensions,elemT>& in_array) const;

};

   
template <int num_dim, typename elemT>
SeparableArrayFunctionObject2<num_dim, elemT>::
SeparableArrayFunctionObject2()
: all_1d_array_filters(VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >(num_dim))
{}


template <int num_dim, typename elemT>
SeparableArrayFunctionObject2<num_dim, elemT>::
SeparableArrayFunctionObject2(const VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >& array_filters)
: all_1d_array_filters(array_filters)
{
  assert(all_1d_array_filters.get_length() == num_dim);
}

template <int num_dimensions, typename elemT>
void 
SeparableArrayFunctionObject2<num_dimensions, elemT>::
do_it(Array<num_dimensions,elemT>& out_array, const Array<num_dimensions,elemT>& in_array) const
{
   
  //if (!is_trivial())
   apply_array_functions_on_each_index(out_array, in_array,
                                             all_1d_array_filters.begin(), 
                                             all_1d_array_filters.end());
  //else somehow copy in_array into out_array but keeping index ranges
   //TODO

}

template <int num_dim, typename elemT>
bool 
SeparableArrayFunctionObject2<num_dim, elemT>::
is_trivial() const
{
  for (typename  VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >::const_iterator iter=all_1d_array_filters.begin();
        iter!=all_1d_array_filters.end();++iter)
   {
     // TODO insert condition on is_null_ptr here (see SeparableArrayFunctionObject)
     if (!(*iter)->is_trivial())
       return false;
   }
   return true;
}

//----------------------------------------------

template <int num_dimensions, typename elemT>
class SeparableLowPassArrayFilter2:
      public SeparableArrayFunctionObject2 <num_dimensions,elemT> 
{
public:  

  //! Default constructor
  SeparableLowPassArrayFilter2();  
  
  SeparableLowPassArrayFilter2(const VectorWithOffset<elemT>& filter_coefficients);
  
private:
 VectorWithOffset<float> filter_coefficients;
 
};


template <int num_dimensions, typename elemT>
SeparableLowPassArrayFilter2<num_dimensions,elemT>::
SeparableLowPassArrayFilter2()
{
 for (int i=1;i<=num_dimensions;i++)
  {
    all_1d_array_filters[i-1] = 	 
      new ArrayFilter1DUsingConvolution<float>();
  }
}


template <int num_dimensions, typename elemT> 
SeparableLowPassArrayFilter2<num_dimensions,elemT>::
SeparableLowPassArrayFilter2(const VectorWithOffset<elemT>& filter_coefficients_v)
:filter_coefficients(filter_coefficients_v)
{
  assert(num_dimensions==3);

  std::cerr << "Printing filter coefficients" << endl;
  for (int i =filter_coefficients_v.get_min_index();i<=filter_coefficients_v.get_max_index();i++)    
    std::cerr  << i<<"   "<< filter_coefficients_v[i] <<"   " << endl;


   all_1d_array_filters[2] = 	 
      new ArrayFilter1DUsingConvolution<float>(filter_coefficients_v);

   all_1d_array_filters[0] = 	 
       new ArrayFilter1DUsingConvolution<float>();
   all_1d_array_filters[1] = 	 
       new ArrayFilter1DUsingConvolution<float>(filter_coefficients_v);
  
    
}
END_NAMESPACE_STIR

#endif
