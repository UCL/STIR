//
// $Id$: $Date$
//

/*! 
  \file 
  \ingroup buildblock
  \brief inline implementations for NumericVectorWithOffset

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$

  \version $Revision$
 */


// include for min,max definitions
#include <algorithm>

START_NAMESPACE_TOMO

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


// addition
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>
NumericVectorWithOffset<T, NUMBER>::operator+ (const NumericVectorWithOffset &v) const 
{
  check_state();
  NumericVectorWithOffset retval(*this);
  return retval += v; 
}

// subtraction
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator- (const NumericVectorWithOffset &v) const 
{
  check_state();
  NumericVectorWithOffset retval(*this);
  return retval -= v; 
}

// elem by elem multiplication
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator* (const NumericVectorWithOffset &v) const
{
  check_state();
  NumericVectorWithOffset retval(*this);
  return retval *= v; 
}

// elem by elem division
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator/ (const NumericVectorWithOffset &v) const
{
  check_state();
  NumericVectorWithOffset retval(*this);
  return retval /= v;
}

// Add a constant to every element
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator+ (const NUMBER &v) const 
{
  check_state();
  NumericVectorWithOffset retval(*this);
  return retval += v;
}

// Subtract a constant from every element
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator- (const NUMBER &v) const 
{
  check_state();
  NumericVectorWithOffset retval(*this);
  return retval -= v; 
}

// Multiply every element by a constant
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator* (const NUMBER &v) const 
{
  check_state();
  NumericVectorWithOffset retval(*this);
  return retval *= v;
}

// Divide every element by a constant
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER> 
NumericVectorWithOffset<T, NUMBER>::operator/ (const NUMBER &v) const 
{
  check_state();
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
  check_state();
  // first check if *this is empty
  if (get_length() == 0)
  {
    return *this = v;
  }
#ifndef TOMO_NO_NAMESPACES
  grow (std::min(get_min_index(),v.get_min_index()), std::max(get_max_index(),v.get_max_index()));
#else
  grow (min(get_min_index(),v.get_min_index()), max(get_max_index(),v.get_max_index()));
#endif
  for (int i=v.get_min_index(); i<=v.get_max_index(); i++)
    num[i] += v.num[i];
  check_state();
  return *this; 
}

/*! See operator+= (const NumericVectorWithOffset&) for growing behaviour */ 
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator-= (const NumericVectorWithOffset &v)
{
  check_state();
  // first check if *this is empty
  if (get_length() == 0)
  {
    *this = v;
    return *this *= -1;
  }
#ifndef TOMO_NO_NAMESPACES
  grow (std::min(get_min_index(),v.get_min_index()), std::max(get_max_index(),v.get_max_index()));
#else
  grow (min(get_min_index(),v.get_min_index()), max(get_max_index(),v.get_max_index()));
#endif
  for (int i=v.get_min_index(); i<=v.get_max_index(); i++)
    num[i] -= v.num[i];
  check_state();
  return *this; 
}

/*! See operator+= (const NumericVectorWithOffset&) for growing behaviour */ 
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator*= (const NumericVectorWithOffset &v)
{
  check_state();
  // first check if *this is empty
  if (get_length() == 0)
  {
    // we have to return an object of the same dimensions as v, but filled with 0. 
    *this =v;
    return *this *= 0;
  }
#ifndef TOMO_NO_NAMESPACES
  grow (std::min(get_min_index(),v.get_min_index()), std::max(get_max_index(),v.get_max_index()));
#else
  grow (min(get_min_index(),v.get_min_index()), max(get_max_index(),v.get_max_index()));
#endif
  for (int i=v.get_min_index(); i<=v.get_max_index(); i++)
    num[i] *= v.num[i];
  check_state();
  return *this; 
}

/*! See operator+= (const NumericVectorWithOffset&) for growing behaviour */ 
template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator/= (const NumericVectorWithOffset &v)
{
  check_state();
  // first check if *this is empty
  if (get_length() == 0)
  {
    // we have to return an object of the same dimensions as v, but filled with 0. 
    *this =v;
    return *this *= 0;
  }
#ifndef TOMO_NO_NAMESPACES
  grow (std::min(get_min_index(),v.get_min_index()), std::max(get_max_index(),v.get_max_index()));
#else
  grow (min(get_min_index(),v.get_min_index()), max(get_max_index(),v.get_max_index()));
#endif
  for (int i=v.get_min_index(); i<=v.get_max_index(); i++)
    num[i] /= v.num[i];
  check_state();
  return *this; 
}

template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator+= (const NUMBER &v) 
{
  check_state();
  for (int i=get_min_index(); i<=get_max_index(); i++)
    num[i] += v;
  check_state();
  return *this; 
}

template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator-= (const NUMBER &v) 
{
  check_state();
  for (int i=get_min_index(); i<=get_max_index(); i++)
    num[i] -= v;
  check_state();
  return *this;
}

template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator*= (const NUMBER &v) 
{
  check_state();
  for (int i=get_min_index(); i<=get_max_index(); i++)
    num[i] *= v;
  check_state();
  return *this; 
}

template <class T, class NUMBER>
inline NumericVectorWithOffset<T, NUMBER>& 
NumericVectorWithOffset<T, NUMBER>::operator/= (const NUMBER &v) 
{
  check_state();
  for (int i=get_min_index(); i<=get_max_index(); i++)
    num[i] /= v;
  check_state();
  return *this;
}

END_NAMESPACE_TOMO
