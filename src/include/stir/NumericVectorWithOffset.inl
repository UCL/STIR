//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2005-06-03, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2013, 2020 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*! 
  \file 
  \ingroup Array
  \brief inline implementations for NumericVectorWithOffset

  \author Kris Thielemans
  \author PARAPET project

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
NumericVectorWithOffset(const VectorWithOffset<T>& t)
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
  this->grow (std::min(this->get_min_index(),v.get_min_index()), std::max(this->get_max_index(),v.get_max_index()));
#else
  this->grow (min(this->get_min_index(),v.get_min_index()), max(this->get_max_index(),v.get_max_index()));
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
  this->grow (std::min(this->get_min_index(),v.get_min_index()), std::max(this->get_max_index(),v.get_max_index()));
#else
  this->grow (min(this->get_min_index(),v.get_min_index()), max(this->get_max_index(),v.get_max_index()));
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
  this->grow (std::min(this->get_min_index(),v.get_min_index()), std::max(this->get_max_index(),v.get_max_index()));
#else
  this->grow (min(this->get_min_index(),v.get_min_index()), max(this->get_max_index(),v.get_max_index()));
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
  this->grow (std::min(this->get_min_index(),v.get_min_index()), std::max(this->get_max_index(),v.get_max_index()));
#else
  this->grow (min(this->get_min_index(),v.get_min_index()), max(this->get_max_index(),v.get_max_index()));
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

template <class T, class NUMBER>
template <class NUMBER2>
inline void
NumericVectorWithOffset<T, NUMBER>::
axpby(const NUMBER2 a, const NumericVectorWithOffset& x,
      const NUMBER2 b, const NumericVectorWithOffset& y)
{  
  NumericVectorWithOffset<T, NUMBER>::xapyb(x,a,y,b);
}

template <class T, class NUMBER>
inline void
NumericVectorWithOffset<T, NUMBER>::
xapyb(const NumericVectorWithOffset& x, const NUMBER a,
      const NumericVectorWithOffset& y, const NUMBER b)
{  
  this->check_state();
  if ((this->get_min_index() != x.get_min_index())
      || (this->get_min_index() != y.get_min_index())
      || (this->get_max_index() != x.get_max_index())
      || (this->get_max_index() != y.get_max_index()))
       error("NumericVectorWithOffset::xapyb: index ranges don't match");

  typename NumericVectorWithOffset::iterator this_iter = this->begin();
  typename NumericVectorWithOffset::const_iterator x_iter = x.begin();
  typename NumericVectorWithOffset::const_iterator y_iter = y.begin();
  while (this_iter != this->end())
    {
      *this_iter++ = (*x_iter++) * a + (*y_iter++) * b;
    }
}

template <class T, class NUMBER>
inline void
NumericVectorWithOffset<T, NUMBER>::
xapyb(const NumericVectorWithOffset& x, const NumericVectorWithOffset& a,
      const NumericVectorWithOffset& y, const NumericVectorWithOffset& b)
{  
  this->check_state();
  if ((this->get_min_index() != x.get_min_index())
      || (this->get_min_index() != y.get_min_index())
      || (this->get_min_index() != a.get_min_index())
      || (this->get_min_index() != b.get_min_index())            
      || (this->get_max_index() != x.get_max_index())
      || (this->get_max_index() != y.get_max_index())
      || (this->get_max_index() != a.get_max_index())
      || (this->get_max_index() != b.get_max_index()))
       error("NumericVectorWithOffset::xapyb: index ranges don't match");

  typename NumericVectorWithOffset::iterator this_iter = this->begin();
  typename NumericVectorWithOffset::const_iterator x_iter = x.begin();
  typename NumericVectorWithOffset::const_iterator y_iter = y.begin();
  typename NumericVectorWithOffset::const_iterator a_iter = a.begin();
  typename NumericVectorWithOffset::const_iterator b_iter = b.begin();

  while (this_iter != this->end())
    {
      *this_iter++ = (*x_iter++) * (*a_iter++) + (*y_iter++) * (*b_iter++);
    }
}

template <class T, class NUMBER>
template <class T2>
inline void
NumericVectorWithOffset<T, NUMBER>::
sapyb(const T2& a,
      const NumericVectorWithOffset& y, const T2& b)
{  
  this->xapyb(*this,a,y,b);
}

END_NAMESPACE_STIR
