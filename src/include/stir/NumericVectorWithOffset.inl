//
// $Id$
//
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
  \brief inline implementations for NumericVectorWithOffset

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
 */

// include for min,max definitions
#include <algorithm>

START_NAMESPACE_STIR

template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>::NumericVectorWithOffset()
  : base_type()
{}

template <class T, class NUMBER>
inline 
NumericVectorWithOffset<T, NUMBER>::NumericVectorWithOffset(const int hsz)
  : base_type(hsz)
{}

template <class T, class NUMBER>
inline 
NumericVectorWithOffset<T, NUMBER>::NumericVectorWithOffset(const int min_index, const int max_index)   
  : base_type(min_index, max_index)
{}

template <class T, class NUMBER>
NumericVectorWithOffset<T, NUMBER>::
NumericVectorWithOffset(const base_type& t)
  : base_type(t)
{}

// addition
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>
NumericVectorWithOffset<T, NUMBER>::operator+ (const NumericVectorWithOffset &v) const 
{
  this->check_state();
  NumericVectorWithOffset retval(*this);
  return retval += v; 
}

// subtraction
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator- (const NumericVectorWithOffset &v) const 
{
  this->check_state();
  NumericVectorWithOffset retval(*this);
  return retval -= v; 
}

// elem by elem multiplication
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator* (const NumericVectorWithOffset &v) const
{
  this->check_state();
  NumericVectorWithOffset retval(*this);
  return retval *= v; 
}

// elem by elem division
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator/ (const NumericVectorWithOffset &v) const
{
  this->check_state();
  NumericVectorWithOffset retval(*this);
  return retval /= v;
}

// Add a constant to every element
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator+ (const NUMBER &v) const 
{
  this->check_state();
  NumericVectorWithOffset retval(*this);
  return retval += v;
}

// Subtract a constant from every element
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator- (const NUMBER &v) const 
{
  this->check_state();
  NumericVectorWithOffset retval(*this);
  return retval -= v; 
}

// Multiply every element by a constant
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator* (const NUMBER &v) const 
{
  this->check_state();
  NumericVectorWithOffset retval(*this);
  return retval *= v;
}

// Divide every element by a constant
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator/ (const NUMBER &v) const 
{
  this->check_state();
  NumericVectorWithOffset retval(*this);
  return retval /= v;
}


/*! This will grow the vector automatically if the 2nd argument has
    smaller min_index and/or larger max_index.
    New elements are first initialised with T() before adding.*/
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator+= (const NumericVectorWithOffset &v) 
{
  this->check_state();
  // first check if *this is empty
  if (this->get_length() == 0)
  {
    return *this = v;
  }
#ifndef STIR_NO_NAMESPACES
  grow (std::min(this->get_min_index(),v.get_min_index()), std::max(this->get_max_index(),v.get_max_index()));
#else
  grow (min(this->get_min_index(),v.get_min_index()), max(this->get_max_index(),v.get_max_index()));
#endif
  for (int i=v.get_min_index(); i<=v.get_max_index(); i++)
    this->num[i] += v.num[i];
  this->check_state();
  return *this; 
}

/*! See operator+= (const NumericVectorWithOffset&) for growing behaviour */ 
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator-= (const NumericVectorWithOffset &v)
{
  this->check_state();
  // first check if *this is empty
  if (this->get_length() == 0)
  {
    *this = v;
    return *this *= -1;
  }
#ifndef STIR_NO_NAMESPACES
  grow (std::min(this->get_min_index(),v.get_min_index()), std::max(this->get_max_index(),v.get_max_index()));
#else
  grow (min(this->get_min_index(),v.get_min_index()), max(this->get_max_index(),v.get_max_index()));
#endif
  for (int i=v.get_min_index(); i<=v.get_max_index(); i++)
    this->num[i] -= v.num[i];
  this->check_state();
  return *this; 
}

/*! See operator+= (const NumericVectorWithOffset&) for growing behaviour */ 
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator*= (const NumericVectorWithOffset &v)
{
  this->check_state();
  // first check if *this is empty
  if (this->get_length() == 0)
  {
    // we have to return an object of the same dimensions as v, but filled with 0. 
    *this =v;
    return *this *= 0;
  }
#ifndef STIR_NO_NAMESPACES
  grow (std::min(this->get_min_index(),v.get_min_index()), std::max(this->get_max_index(),v.get_max_index()));
#else
  grow (min(this->get_min_index(),v.get_min_index()), max(this->get_max_index(),v.get_max_index()));
#endif
  for (int i=v.get_min_index(); i<=v.get_max_index(); i++)
    this->num[i] *= v.num[i];
  this->check_state();
  return *this; 
}

/*! See operator+= (const NumericVectorWithOffset&) for growing behaviour */ 
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator/= (const NumericVectorWithOffset &v)
{
  this->check_state();
  // first check if *this is empty
  if (this->get_length() == 0)
  {
    // we have to return an object of the same dimensions as v, but filled with 0. 
    *this =v;
    return *this *= 0;
  }
#ifndef STIR_NO_NAMESPACES
  grow (std::min(this->get_min_index(),v.get_min_index()), std::max(this->get_max_index(),v.get_max_index()));
#else
  grow (min(this->get_min_index(),v.get_min_index()), max(this->get_max_index(),v.get_max_index()));
#endif
  for (int i=v.get_min_index(); i<=v.get_max_index(); i++)
    this->num[i] /= v.num[i];
  this->check_state();
  return *this; 
}

template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator+= (const NUMBER &v) 
{
  this->check_state();
  for (int i=this->get_min_index(); i<=this->get_max_index(); i++)
    this->num[i] += v;
  this->check_state();
  return *this; 
}

template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator-= (const NUMBER &v) 
{
  this->check_state();
  for (int i=this->get_min_index(); i<=this->get_max_index(); i++)
    this->num[i] -= v;
  this->check_state();
  return *this;
}

template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator*= (const NUMBER &v) 
{
  this->check_state();
  for (int i=this->get_min_index(); i<=this->get_max_index(); i++)
    this->num[i] *= v;
  this->check_state();
  return *this; 
}

template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator/= (const NUMBER &v) 
{
  this->check_state();
  for (int i=this->get_min_index(); i<=this->get_max_index(); i++)
    this->num[i] /= v;
  this->check_state();
  return *this;
}

END_NAMESPACE_STIR
