//
// $Id$
//

#ifndef __stir_array_index_functions_h_
#define __stir_array_index_functions_h_

/*!
  \file 
  \ingroup buildblock
 
  \brief a variety of useful functions for indexing Array objects

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "stir/Array.h"
#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

template <int num_dimensions, int num_dimensions2, typename elemT>
inline
const Array<num_dimensions-num_dimensions2,elemT>&
get(const Array<num_dimensions,elemT>& a, const BasicCoordinate<num_dimensions2,int> &c) 
{
  return get(a[c[1]],cut_first_dimension(c)); 
}

template <int num_dimensions, typename elemT>
inline
const elemT&
get(const Array<num_dimensions,elemT>& a, const BasicCoordinate<num_dimensions,int> &c) 
{
  return a[c];
}			
template <int num_dimensions, typename elemT>
inline
const Array<num_dimensions-1,elemT>&
get(const Array<num_dimensions,elemT>& a, const BasicCoordinate<1,int> &c) 
{
  return a[c[1]]; 
}			



template <int num_dimensions, typename T>
inline
BasicCoordinate<num_dimensions, int>
get_first_index(const Array<num_dimensions, T>& a)
{
  return join(a.get_min_index(), get_first_index(*a.begin()));
}

template <typename T>
inline
BasicCoordinate<1, int>
get_first_index(const Array<1, T>& a)
{
  BasicCoordinate<1, int> result;
  result[1] = a.get_min_index();
  return result;
}

template <int num_dimensions2, typename T>
inline
bool 
next(BasicCoordinate<1, int>& index, 
     const Array<num_dimensions2, T>& a)
{
  index[1]++;
  return index[1]<=a.get_max_index();
}
template <int num_dimensions, int num_dimensions2, typename T>
inline
bool 
next(BasicCoordinate<num_dimensions, int>& index, 
     const Array<num_dimensions2, T>& a)
{
  index[num_dimensions]++;
  BasicCoordinate<num_dimensions-1, int> upper_index= cut_last_dimension(index);
  if (index[num_dimensions]<=get(a,cut_last_dimension(index)).get_max_index())
    return true;
  if (!next(upper_index, a))
    return false;
  index=join(upper_index, get(a,cut_last_dimension(index)).get_min_index());
  return true;
}

END_NAMESPACE_STIR
#endif
