//
// $Id$
//
/*!
  \file
  \ingroup test
  \brief 

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
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
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif
#include "local/stir/ArrayFilter1DUsingConvolution.h"

START_NAMESPACE_STIR

/*!
  \warning Both in_array and out_array have to have regular ranges. Moreover, they have to have matching ranges except for the outermost level. 
  */
template <int num_dim, typename elemT, typename FunctionObject> 
inline void
apply_array_function_on_1st_index(Array<num_dim, elemT>& out_array, const Array<num_dim, elemT>& in_array, FunctionObject f)
{
  assert(in_array.is_regular());
  assert(out_array.is_regular());
  const int in_min_index = in_array.get_min_index();
  const int in_max_index = in_array.get_max_index();
  const int out_min_index = out_array.get_min_index();
  const int out_max_index = out_array.get_max_index();

  // construct a vector with a full_iterator for every in_array[i]
  VectorWithOffset< Array<num_dim-1, elemT>::const_full_iterator > 
    in_full_iterators (in_min_index, in_max_index);  
  for (int i=in_min_index; i<=in_max_index; ++i)
    in_full_iterators[i] = in_array[i].begin_all();
  // same for out_array[i]
  VectorWithOffset< Array<num_dim-1, elemT>::full_iterator > 
    out_full_iterators (out_min_index, out_max_index);  
  for (int i=out_min_index; i<=out_max_index; ++i)
    out_full_iterators[i] = out_array[i].begin_all();
  
  // allocate 1d array
  Array<1, elemT> in_array1d (in_min_index, in_max_index);
  Array<1, elemT> out_array1d (out_min_index, out_max_index);

  while (in_full_iterators[in_min_index] != in_array[in_min_index].end_all())
  {
    assert(out_full_iterators[out_min_index] != out_array[out_min_index].end_all());
    // copy elements into 1d array
    for (int i=in_min_index; i<=in_max_index; ++i)    
      in_array1d[i] = *in_full_iterators[i];
    
    // apply function
    (*f)(out_array1d, in_array1d);
    
    // put results back
    for (int i=out_min_index; i<=out_max_index; ++i)    
      *out_full_iterators[i] = out_array1d[i];
    
    // now increment full_iterators to do next index
    for (int i=in_min_index; i<=in_max_index; ++i)
      ++in_full_iterators[i];
    for (int i=out_min_index; i<=out_max_index; ++i)
      ++out_full_iterators[i];
  }
  assert(out_full_iterators[out_min_index] == out_array[out_min_index].end_all());
    
}

// TODO do not rely on vector<shared_ptr <etc>>
#define ActualFunctionObjectIter shared_ptr<ArrayFunctionObject<1,elemT> > const* 
//VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >::const_iterator 

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <int num_dim, typename elemT, typename FunctionObjectIter> 
inline void 
apply_array_functions_on_each_index(Array<num_dim, elemT>& out_array, 
                                    const Array<num_dim, elemT>& in_array, 
                                    FunctionObjectIter start, 
                                    FunctionObjectIter stop)
{
  assert(start+num_dim == stop);
  assert(num_dim > 1);
    
  assert(out_array.is_regular());
  BasicCoordinate<num_dim, int> tmp_out_min_indices, tmp_out_max_indices;
  out_array.get_regular_range(tmp_out_min_indices, tmp_out_max_indices);
  tmp_out_min_indices[1] = in_array.get_min_index();
  tmp_out_max_indices[1] = in_array.get_max_index();
  Array<num_dim, elemT> tmp_out_array(IndexRange<num_dim>(tmp_out_min_indices, tmp_out_max_indices));
  
  for (int i=in_array.get_min_index(); i<=in_array.get_max_index(); ++i)
    apply_array_functions_on_each_index(tmp_out_array[i], in_array[i], start+1, stop);

  apply_array_function_on_1st_index(out_array, tmp_out_array, *start);

}
#endif

// specialisation that uses ArrayFunctionObject::is_trivial etc
template <int num_dim, typename elemT> 
inline void 
apply_array_functions_on_each_index(Array<num_dim, elemT>& out_array, 
                                    const Array<num_dim, elemT>& in_array, 
                                    ActualFunctionObjectIter start, 
                                    ActualFunctionObjectIter stop)
{
  assert(start+num_dim == stop);
  assert(num_dim > 1);
    

  //cerr << "apply_array_functions_on_each_index dim " << num_dim << std::endl;
  if ((**start).is_trivial())
    {
      for (int i=max(out_array.get_min_index(), in_array.get_min_index());
	   i<=min(out_array.get_max_index(),in_array.get_max_index());
	   ++i)
	apply_array_functions_on_each_index(out_array[i], in_array[i], start+1, stop);
    }
  else
    {
      assert(out_array.is_regular());
 
      IndexRange<1> influencing_indices;
      if ((**start).get_influencing_indices(influencing_indices, 
                                            IndexRange<1>(out_array.get_min_index(),
                                                          out_array.get_max_index()))
        == Succeeded::no)
        influencing_indices = IndexRange<1>(influencing_indices.get_min_index(),
                                            in_array.get_max_index());
      else
      {
        influencing_indices =
          IndexRange<1>(max(influencing_indices.get_min_index(), 
                            in_array.get_min_index()),
                        min(influencing_indices.get_max_index(), 
                            in_array.get_max_index()));
      }
      BasicCoordinate<num_dim, int> tmp_out_min_indices, tmp_out_max_indices;
      out_array.get_regular_range(tmp_out_min_indices, tmp_out_max_indices);
      tmp_out_min_indices[1] = influencing_indices.get_min_index();
      tmp_out_max_indices[1] = influencing_indices.get_max_index();
      Array<num_dim, elemT> tmp_out_array(IndexRange<num_dim>(tmp_out_min_indices, tmp_out_max_indices));
      
      for (int i=influencing_indices.get_min_index(); i<=influencing_indices.get_max_index(); ++i)
	apply_array_functions_on_each_index(tmp_out_array[i], in_array[i], start+1, stop);
      
      apply_array_function_on_1st_index(out_array, tmp_out_array, *start);
    }

}

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

// has to be here to get general 1D specialisation to compile
template <typename elemT>
inline void 
apply_array_functions_on_each_index(Array<1, elemT>& out_array, 
                                    const Array<1, elemT>& in_array, 
                                    ActualFunctionObjectIter start, 
                                    ActualFunctionObjectIter stop)
{
  assert(start+1 == stop);
  (**start)(out_array, in_array);
}

#endif


//! 1d specialisation. 
/*!
  \warning 
  Currently uses (**start)(out_array, in_array), while preferably it would use (*start)(out_array, in_array). However, this doesn't compile when the function objects are part of a 
 std::vector< shared_ptr<ArrayFunctionObject<1,elemT> > >.
 (TODO)
*/
#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
// silly business for deficient compilers (including VC 6.0)
#define elemT float
#define FunctionObjectIter ActualFunctionObjectIter
#else
template <typename elemT, typename FunctionObjectIter> 
#endif
inline void 
apply_array_functions_on_each_index(Array<1, elemT>& out_array, const Array<1, elemT>& in_array, FunctionObjectIter start, FunctionObjectIter stop)
{
  assert(start+1 == stop);
  (**start)(out_array, in_array);
}

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef elemT
#undef FunctionObjectIter
#endif

#undef ActualFunctionObjectIter

//-----------------------------------------------------
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
   
  if (!is_trivial())
   apply_array_functions_on_each_index(out_array, in_array,
                                             all_1d_array_filters.begin(), 
                                             all_1d_array_filters.end());

}

template <int num_dim, typename elemT>
bool 
SeparableArrayFunctionObject2<num_dim, elemT>::
is_trivial() const
{
  for ( VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >::const_iterator iter=all_1d_array_filters.begin();
        iter!=all_1d_array_filters.end();++iter)
   {
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

  cerr << "Printing filter coefficients" << endl;
  for (int i =filter_coefficients_v.get_min_index();i<=filter_coefficients_v.get_max_index();i++)    
    cerr  << i<<"   "<< filter_coefficients_v[i] <<"   " << endl;


   all_1d_array_filters[2] = 	 
      new ArrayFilter1DUsingConvolution<float>(filter_coefficients_v);

   all_1d_array_filters[0] = 	 
       new ArrayFilter1DUsingConvolution<float>();
   all_1d_array_filters[1] = 	 
       new ArrayFilter1DUsingConvolution<float>(filter_coefficients_v);
  
    
}
END_NAMESPACE_STIR

#endif