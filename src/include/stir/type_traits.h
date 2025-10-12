
/*
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_typetraits_H__
#define __stir_typetraits_H__

/*!
  \file
  \ingroup buildblock
  \brief defines various type traits, checking for iterators etc

  \author Kris Thielemans

*/
#include <type_traits>
#include "stir/common.h"

START_NAMESPACE_STIR

/*!
 \defgroup typetraits type traits for template metaprogramming that help with STIR
 \ingroup buildblock

 Can be used as follows
 \code
  // void function that is enabled only if the class has an `iterator` typedef
  template <typename T>
  std::enable_if_t<has_iterator_v<T>>
  func(T& obj)
  {
    for (auto iter: obj)
      // do something
  }
  \endcode
*/
//@{

#if 0
// Helper template to check if a method exists
template <typename T>
struct has_method_foo<T, typename = void> : std::false_type {};
struct has_method_foo<T, std::void_t<decltype(std::declval<T>().foo())>> : std::true_type {}; // Specialization: method exists

#endif

//! Helper to check if a type has an iterator typedef
template <typename T, typename = void>
struct has_iterator : std::false_type
{
};

template <typename T>
struct has_iterator<T, std::void_t<typename T::iterator>> : std::true_type
{
};

//! Bool set to has_iterator<T>::value
template <typename T>
constexpr bool has_iterator_v = has_iterator<T>::value;

//! Helper to check if a type has a full_iterator typedef (e.g. Array<2,int>)
template <typename T, typename = void>
struct has_full_iterator : std::false_type
{
};

template <typename T>
struct has_full_iterator<T, std::void_t<typename T::full_iterator>> : std::true_type
{
};

//! Bool set to has_full_iterator<T>::value
template <typename T>
constexpr bool has_full_iterator_v = has_full_iterator<T>::value;

//! Helper to check if the type has an iterator but no full_iterator typedef (e.g. std::vector<int>)
template <typename T>
struct has_iterator_and_no_full_iterator : std::conjunction<has_iterator<T>, std::negation<has_full_iterator<T>>>
{
};

//! Bool set to has_iterator_and_no_full_iterator<T>::value
template <typename T>
constexpr bool has_iterator_and_no_full_iterator_v = has_iterator_and_no_full_iterator<T>::value;

//@}

END_NAMESPACE_STIR

#endif
