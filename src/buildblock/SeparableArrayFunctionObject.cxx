//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Implementations for SeparableArrayFunctionObject

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/SeparableArrayFunctionObject.h"
#include "stir/ArrayFunction.h"

START_NAMESPACE_STIR

template <int num_dim, typename elemT>
SeparableArrayFunctionObject<num_dim, elemT>::
SeparableArrayFunctionObject()
: all_1d_array_filters(VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >(num_dim))
{}

template <int num_dim, typename elemT>
SeparableArrayFunctionObject<num_dim, elemT>::
SeparableArrayFunctionObject(const VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >& array_filters)
: all_1d_array_filters(array_filters)
{
  assert(all_1d_array_filters.get_length() == num_dim);
}


template <int num_dim, typename elemT>
void 
SeparableArrayFunctionObject<num_dim, elemT>::
do_it(Array<num_dim,elemT>& array) const
{
  if (!is_trivial())
    {
#ifndef NDEBUG
      // currently in_place_apply_array_functions_on_each_index doesn't handle 0 
      // pointers gracefully, so we check here that there aren't any
      for ( VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >::const_iterator
	      iter=all_1d_array_filters.begin();
	    iter!=all_1d_array_filters.end();
	    ++iter)
	assert(iter->use_count()>0);
#endif
       in_place_apply_array_functions_on_each_index(array, 
						    all_1d_array_filters.begin(), 
						    all_1d_array_filters.end());
    }
}

template <int num_dim, typename elemT>
bool 
SeparableArrayFunctionObject<num_dim, elemT>::
is_trivial() const
{
  for ( VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >::const_iterator 
	  iter=all_1d_array_filters.begin();
        iter!=all_1d_array_filters.end();
	++iter)
   {
     if (iter->use_count()>0 && !(*iter)->is_trivial())
       return false;
   }
   return true;
}


// instantiation
template SeparableArrayFunctionObject<3, float>;

END_NAMESPACE_STIR



