// $Id$
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \ingroup Array

  \brief Implementations for ArrayFunction.h

  \author Kris Thielemans (some functions based on some earlier work by Darren Hague)
  \author PARAPET project

  $Date$
  $Revision$
  
  \warning Compilers without partial specialisation of templates are
   catered for by explicit instantiations. If you need it for any other
   types, you'd have to add them by hand.
 */
#include <cmath>
#include <complex>
# ifdef BOOST_NO_STDC_NAMESPACE
    namespace std { using ::log; using ::exp; }
# endif


START_NAMESPACE_STIR

//----------------------------------------------------------------------
// element wise and in place numeric functions
//----------------------------------------------------------------------

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class elemT>
inline Array<1,elemT>&
in_place_log(Array<1,elemT>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = std::log(v[i]); 
  return v; 
}

#else
inline Array<1,float>& 
in_place_log(Array<1,float>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = std::log(v[i]); 
  return v; 
}
#endif


template <int num_dimensions, class elemT>
inline Array<num_dimensions, elemT>& 
in_place_log(Array<num_dimensions, elemT>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    in_place_log(v[i]); 
  return v; 
}

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class elemT>
inline Array<1,elemT>& 
in_place_exp(Array<1,elemT>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = std::exp(v[i]); 
  return v; 
}
#else
inline Array<1,float>& 
in_place_exp(Array<1,float>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = std::exp(v[i]); 
  return v; 
}
#endif

template <int num_dimensions, class elemT>
inline Array<num_dimensions, elemT>& 
in_place_exp(Array<num_dimensions, elemT>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    in_place_exp(v[i]); 
  return v; 
}

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class elemT>
inline Array<1,elemT>& 
in_place_abs(Array<1,elemT>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    if (v[i] < 0)
      v[i] = -v[i];
  return v; 
}
#else
inline Array<1,float>& 
in_place_abs(Array<1,float>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    if (v[i] < 0)
      v[i] = -v[i];
  return v; 
}
#endif


template <int num_dimensions, class elemT>
inline Array<num_dimensions, elemT>& 
in_place_abs(Array<num_dimensions, elemT>& v)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    in_place_abs(v[i]); 
  return v; 
}


// this generic function does not seem to work of f is an overloaded function
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <class elemT, class FUNCTION>
inline Array<1,elemT>& 
in_place_apply_function(Array<1,elemT>& v, FUNCTION f)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = f(v[i]); 
  return v; 
}
#else
inline Array<1,float>& 
in_place_apply_function(Array<1,float>& v, float (*f)(float))  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    v[i] = f(v[i]); 
  return v; 
}
#endif


template <int num_dimensions, class elemT, class FUNCTION>
inline Array<num_dimensions, elemT>& 
in_place_apply_function(Array<num_dimensions, elemT>& v, FUNCTION f)  
{	
  for(int i=v.get_min_index(); i<=v.get_max_index(); i++)
    in_place_apply_function(v[i], f); 
  return v; 
}


template <int num_dim, typename elemT, typename FunctionObjectPtr> 
inline void
in_place_apply_array_function_on_1st_index(Array<num_dim, elemT>& array, FunctionObjectPtr f)
{
  assert(array.is_regular());
  const int outer_min_index = array.get_min_index();
  const int outer_max_index = array.get_max_index();

  // construct a vector with a full_iterator for every array[i]
  VectorWithOffset<
#ifndef _MSC_VER
    typename 
#endif
      Array<num_dim-1, elemT>::full_iterator > 
    full_iterators (outer_min_index, outer_max_index);  
  for (int i=outer_min_index; i<=outer_max_index; ++i)
    full_iterators[i] = array[i].begin_all();
  
  // allocate 1d array
  Array<1, elemT> array1d (outer_min_index, outer_max_index);

  while (full_iterators[outer_min_index] != array[outer_min_index].end_all())
  {
    // copy elements into 1d array
    for (int i=outer_min_index; i<=outer_max_index; ++i)    
      array1d[i] = *full_iterators[i];
    
    // apply function
    (*f)(array1d);
    
    // put results back
    // and increment full_iterators to do next index
    for (int i=outer_min_index; i<=outer_max_index; ++i)
      *full_iterators[i]++ = array1d[i];
  }
    
}

template <int num_dim, typename elemT, typename FunctionObjectPtr> 
inline void
apply_array_function_on_1st_index(Array<num_dim, elemT>& out_array, 
                                  const Array<num_dim, elemT>& in_array, 
                                  FunctionObjectPtr f)
{
  assert(in_array.is_regular());
  assert(out_array.is_regular());
  const int in_min_index = in_array.get_min_index();
  const int in_max_index = in_array.get_max_index();
  const int out_min_index = out_array.get_min_index();
  const int out_max_index = out_array.get_max_index();

  // construct a vector with a full_iterator for every in_array[i]
  VectorWithOffset< typename Array<num_dim-1, elemT>::const_full_iterator > 
    in_full_iterators (in_min_index, in_max_index);  
  for (int i=in_min_index; i<=in_max_index; ++i)
    in_full_iterators[i] = in_array[i].begin_all();
  // same for out_array[i]
  VectorWithOffset<typename  Array<num_dim-1, elemT>::full_iterator > 
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
    // increment in_full_iterators for next index
    for (int i=in_min_index; i<=in_max_index; ++i)    
      in_array1d[i] = *(in_full_iterators[i]++);
    
    // apply function
    (*f)(out_array1d, in_array1d);
    assert(out_array1d.get_min_index() == out_min_index);
    assert(out_array1d.get_max_index() == out_max_index);
    
    // put results back
    // increment out_full_iterators for next index
    for (int i=out_min_index; i<=out_max_index; ++i)    
      *(out_full_iterators[i]++) = out_array1d[i];    
  }
  assert(out_full_iterators[out_min_index] == out_array[out_min_index].end_all());
    
}

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
// silly business for deficient compilers (including VC 6.0)
#define elemT float
#define FunctionObjectPtrIter ActualFunctionObjectPtrIter
#endif

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <int num_dim>
#else
template <int num_dim, typename elemT, typename FunctionObjectPtrIter> 
#endif
inline void 
in_place_apply_array_functions_on_each_index(Array<num_dim, elemT>& array, 
                                             FunctionObjectPtrIter start, 
                                             FunctionObjectPtrIter stop)
{
  assert(start+num_dim == stop);
  assert(num_dim > 1);
  in_place_apply_array_function_on_1st_index(array, *start);

  ++start;
  for (typename Array<num_dim, elemT>::iterator restiter = array.begin(); restiter != array.end(); ++restiter)
    in_place_apply_array_functions_on_each_index(*restiter, start, stop);
}

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename elemT, typename FunctionObjectPtrIter> 
#endif
inline void 
in_place_apply_array_functions_on_each_index(Array<1, elemT>& array, FunctionObjectPtrIter start, FunctionObjectPtrIter stop)
{
  assert(start+1 == stop);
  (**start)(array);
}


#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <int num_dim, typename elemT, typename FunctionObjectPtrIter> 
inline void 
apply_array_functions_on_each_index(Array<num_dim, elemT>& out_array, 
                                    const Array<num_dim, elemT>& in_array, 
                                    FunctionObjectPtrIter start, 
                                    FunctionObjectPtrIter stop)
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
#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <int num_dim>
#else
template <int num_dim, typename elemT> 
#endif
inline void 
apply_array_functions_on_each_index(Array<num_dim, elemT>& out_array, 
                                    const Array<num_dim, elemT>& in_array, 
                                    ActualFunctionObjectPtrIter start, 
                                    ActualFunctionObjectPtrIter stop)
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
#endif
inline void 
apply_array_functions_on_each_index(Array<1, elemT>& out_array, 
                                    const Array<1, elemT>& in_array, 
                                    ActualFunctionObjectPtrIter start, 
                                    ActualFunctionObjectPtrIter stop)
{
  assert(start+1 == stop);
  (**start)(out_array, in_array);
}


#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename elemT, typename FunctionObjectPtrIter> 
inline void 
apply_array_functions_on_each_index(Array<1, elemT>& out_array, 
                                    const Array<1, elemT>& in_array, 
                                    FunctionObjectPtrIter start, FunctionObjectPtrIter stop)
{
  assert(start+1 == stop);
  (**start)(out_array, in_array);
}
#endif

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#undef elemT
#undef FunctionObjectPtrIter
#endif



END_NAMESPACE_STIR

