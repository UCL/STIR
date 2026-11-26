//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2010-07-01, Hammersmith Imanet Ltd
    Copyright (C) 2012-06-01 - 2012, Kris Thielemans
    Copyright (C) 2023 - 2026, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Array
  \brief inline implementations of stir::VectorWithOffset

  \author Kris Thielemans
  \author PARAPET project


*/

#include "stir/IndexRange.h"
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include "thresholding.h"
#include "stir/error.h"

START_NAMESPACE_STIR

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::init()
{
  length = 0;    // i.e. an empty row of zero length,
  start = 0;     // no offsets
  num = nullptr; // and no data.
  begin_allocated_memory = nullptr;
  end_allocated_memory = nullptr;
  allocated_memory_sptr = nullptr;
}

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::init_with_copy(const indexT min_index, const indexT max_index, T const* const data_ptr)
{
  this->pointer_access = false;
  this->resize(min_index, max_index);
  if (this->length > 0)
    std::copy(data_ptr, data_ptr + this->length, this->begin());
}

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::init(const indexT min_index, const indexT max_index, T* const data_ptr, bool copy_data)
{
  if (copy_data)
    {
      this->init_with_copy(min_index, max_index, data_ptr);
    }
  else
    {
      this->pointer_access = false;
      this->length = max_index >= min_index ? static_cast<size_type>(max_index - min_index) + 1 : 0U;
      this->start = min_index;
      this->begin_allocated_memory = data_ptr;
      this->end_allocated_memory = data_ptr + this->length;
      this->num = this->begin_allocated_memory - this->start;
      this->check_state();
    }
}

template <class T, class indexT>
bool
VectorWithOffset<T, indexT>::owns_memory_for_data() const
{
  return this->allocated_memory_sptr ? true : false;
}

/*!
This function (only non-empty when debugging)
is used before and after any modification of the object
*/
template <class T, class indexT>
void
VectorWithOffset<T, indexT>::check_state() const
{
  // disable for normal debugging
#if _DEBUG > 1
  assert(((length > 0) || (length == 0 && start == 0 && num == begin_allocated_memory)));

#endif
  assert(begin_allocated_memory <= num + start);
  assert(end_allocated_memory >= begin_allocated_memory);
  assert(static_cast<size_type>(end_allocated_memory - begin_allocated_memory) >= length);
  assert(!allocated_memory_sptr || (allocated_memory_sptr.get() == begin_allocated_memory));
}

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::_destruct_and_deallocate()
{
  // check if data is being accessed via a pointer (see get_data_ptr())
  assert(pointer_access == false);
  // TODO when reserve() no longer initialises new elements,
  // we'll have to be careful to delete only initialised elements
  // and just de-allocate the rest

  this->allocated_memory_sptr = nullptr;
}

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::recycle()
{
  this->check_state();
  this->_destruct_and_deallocate();
  this->init();
}

template <class T, class indexT>
indexT
VectorWithOffset<T, indexT>::get_min_index() const
{
  return start;
}

template <class T, class indexT>
indexT
VectorWithOffset<T, indexT>::get_max_index() const
{
  assert(std::is_signed_v<indexT> || (length > 0));
  return start + length - 1;
}

/*! Out of range errors are detected using assert() */
template <class T, class indexT>
T&
VectorWithOffset<T, indexT>::operator[](indexT i)
{
  this->check_state();
  assert(i >= this->get_min_index());
  assert(i <= this->get_max_index());

  return num[i];
}

/*! Out of range errors are detected using assert() */
template <class T, class indexT>
const T&
VectorWithOffset<T, indexT>::operator[](indexT i) const
{
  this->check_state();
  assert(i >= this->get_min_index());
  assert(i <= this->get_max_index());

  return num[i];
}

template <class T, class indexT>
T&
VectorWithOffset<T, indexT>::at(indexT i)
{
  if (length == 0 || i < this->get_min_index() || i > this->get_max_index())
    throw std::out_of_range("index out of range");
  this->check_state();
  return num[i];
}

template <class T, class indexT>
const T&
VectorWithOffset<T, indexT>::at(indexT i) const
{
  if (length == 0 || i < this->get_min_index() || i > this->get_max_index())
    throw std::out_of_range("index out of range");
  this->check_state();

  return num[i];
}

template <class T, class indexT>
bool
VectorWithOffset<T, indexT>::empty() const
{
  return length == 0;
}

template <class T, class indexT>
typename VectorWithOffset<T, indexT>::iterator
VectorWithOffset<T, indexT>::begin()
{
  this->check_state();
  return typename VectorWithOffset<T, indexT>::iterator(num + this->get_min_index());
}

template <class T, class indexT>
typename VectorWithOffset<T, indexT>::const_iterator
VectorWithOffset<T, indexT>::begin() const
{
  this->check_state();
  return typename VectorWithOffset<T, indexT>::const_iterator(num + this->get_min_index());
}

template <class T, class indexT>
typename VectorWithOffset<T, indexT>::iterator
VectorWithOffset<T, indexT>::end()
{
  return this->begin() + this->length;
}

template <class T, class indexT>
typename VectorWithOffset<T, indexT>::const_iterator
VectorWithOffset<T, indexT>::end() const
{
  return this->begin() + this->length;
}

template <class T, class indexT>
typename VectorWithOffset<T, indexT>::reverse_iterator
VectorWithOffset<T, indexT>::rbegin()
{
  this->check_state();
  return std::make_reverse_iterator(end());
}

template <class T, class indexT>
typename VectorWithOffset<T, indexT>::const_reverse_iterator
VectorWithOffset<T, indexT>::rbegin() const
{
  this->check_state();
  return std::make_reverse_iterator(end());
}

template <class T, class indexT>
typename VectorWithOffset<T, indexT>::reverse_iterator
VectorWithOffset<T, indexT>::rend()
{
  this->check_state();
  return std::make_reverse_iterator(begin());
}

template <class T, class indexT>
typename VectorWithOffset<T, indexT>::const_reverse_iterator
VectorWithOffset<T, indexT>::rend() const
{
  this->check_state();
  return std::make_reverse_iterator(begin());
}

template <class T, class indexT>
VectorWithOffset<T, indexT>::VectorWithOffset()
    : pointer_access(false)
{
  this->init();
}

template <class T, class indexT>
VectorWithOffset<T, indexT>::VectorWithOffset(const indexT hsz)
    : VectorWithOffset(0, hsz > 0 ? hsz - 1 : 0)
{
  // note: somewhat awkward implementation to avoid problems when indexT is unsigned
  if (hsz <= 0)
    this->recycle();
}

template <class T, class indexT>
VectorWithOffset<T, indexT>::VectorWithOffset(const indexT min_index, const indexT max_index)
    : length(max_index >= min_index ? static_cast<size_type>(max_index - min_index + 1) : 0),
      start(min_index),
      pointer_access(false)
{
  if (max_index >= min_index)
    {
      allocated_memory_sptr = shared_ptr<T[]>(new T[length]);
      begin_allocated_memory = allocated_memory_sptr.get();
      end_allocated_memory = begin_allocated_memory + length;
      num = begin_allocated_memory - min_index;
    }
  else
    this->init();
  this->check_state();
}

#if STIR_VERSION < 070000
template <class T, class indexT>
VectorWithOffset<T, indexT>::VectorWithOffset(const indexT min_index,
                                              const indexT max_index,
                                              T* const data_ptr,
                                              T* const end_of_data_ptr)
    : length(static_cast<size_type>(max_index - min_index) + 1),
      start(min_index),
      allocated_memory_sptr(nullptr), // we don't own the data
      pointer_access(false)
{
  this->begin_allocated_memory = data_ptr;
  this->end_allocated_memory = end_of_data_ptr;
  this->num = this->begin_allocated_memory - this->start;
  this->check_state();
}

template <class T, class indexT>
VectorWithOffset<T, indexT>::VectorWithOffset(const indexT sz, T* const data_ptr, T* const end_of_data_ptr)
    : VectorWithOffset((std::is_signed_v<indexT> || sz > 0) ? 0 : 1,
                       (std::is_signed_v<indexT> || sz > 0) ? sz - 1 : 0,
                       data_ptr,
                       end_of_data_ptr)
{}
#endif // STIR_VERSION < 070000

template <class T, class indexT>
VectorWithOffset<T, indexT>::VectorWithOffset(const indexT min_index, const indexT max_index, const T* const data_ptr)
{
  // first set empty, such that resize() will work ok
  this->init();
  this->init_with_copy(min_index, max_index, data_ptr);
}

template <class T, class indexT>
VectorWithOffset<T, indexT>::VectorWithOffset(const indexT sz, const T* const data_ptr)
    : VectorWithOffset((std::is_signed_v<indexT> || sz > 0) ? 0 : 1, (std::is_signed_v<indexT> || sz > 0) ? sz - 1 : 0, data_ptr)
{}

template <class T, class indexT>
VectorWithOffset<T, indexT>::VectorWithOffset(const indexT min_index, const indexT max_index, shared_ptr<T[]> data_sptr)
{
  this->allocated_memory_sptr = data_sptr;
  this->init(min_index, max_index, data_sptr.get(), /* copy_data = */ false);
}

template <class T, class indexT>
VectorWithOffset<T, indexT>::VectorWithOffset(VectorWithOffset<T, indexT>&& other) noexcept
    : VectorWithOffset()
{
  swap(*this, other);
}

template <class T, class indexT>
VectorWithOffset<T, indexT>::~VectorWithOffset()
{
  // check if data is being accessed via a pointer (see get_data_ptr())
  assert(pointer_access == false);
  _destruct_and_deallocate();
}

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::set_offset(const indexT min_index)
{
  this->check_state();
  //  only do something when non-zero length
  if (length == 0)
    return;
  // note: num += (start - min_index), but split up in 2 steps in case indexT is unsigned
  num += start;
  num -= min_index;
  start = min_index;
}

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::set_min_index(const indexT min_index)
{
  this->set_offset(min_index);
}

template <class T, class indexT>
size_t
VectorWithOffset<T, indexT>::capacity() const
{
  return size_t(end_allocated_memory - begin_allocated_memory);
}

template <class T, class indexT>
indexT
VectorWithOffset<T, indexT>::get_capacity_min_index() const
{
  // the behaviour for length==0 depends on num==begin_allocated_memory
  assert(length > 0 || num == begin_allocated_memory);
  return static_cast<indexT>(begin_allocated_memory - num);
}

template <class T, class indexT>
indexT
VectorWithOffset<T, indexT>::get_capacity_max_index() const
{
  // the behaviour for length==0 depends on num==begin_allocated_memory
  assert(length > 0 || num == begin_allocated_memory);
  return static_cast<indexT>(end_allocated_memory - num - 1);
}

// the new members will be initialised with the default constructor for T
//  but this should change in the future
template <class T, class indexT>
void
VectorWithOffset<T, indexT>::reserve(const indexT new_capacity_min_index, const indexT new_capacity_max_index)
{
  this->check_state();
  const indexT actual_capacity_min_index
      = length == 0 ? new_capacity_min_index : std::min(this->get_capacity_min_index(), new_capacity_min_index);
  const indexT actual_capacity_max_index
      = length == 0 ? new_capacity_max_index : std::max(this->get_capacity_max_index(), new_capacity_max_index);
  if (actual_capacity_min_index > actual_capacity_max_index)
    return;

  const size_type new_capacity = static_cast<size_type>(actual_capacity_max_index - actual_capacity_min_index + 1);
  if (new_capacity <= this->capacity())
    return;

  // check if data is being accessed via a pointer (see get_data_ptr())
  assert(pointer_access == false);
  // TODO use allocator here instead of new
  shared_ptr<T[]> new_allocated_memory_sptr(new T[new_capacity]);

  const size_type extra_at_the_left
      = length == 0 ? 0U : std::max(indexT(0), indexT(this->get_min_index() - actual_capacity_min_index));
  std::copy(this->begin(), this->end(), new_allocated_memory_sptr.get() + extra_at_the_left);
  this->_destruct_and_deallocate();
  allocated_memory_sptr = std::move(new_allocated_memory_sptr);
  begin_allocated_memory = allocated_memory_sptr.get();
  end_allocated_memory = begin_allocated_memory + new_capacity;
  num = begin_allocated_memory + extra_at_the_left - (length > 0 ? start : 0);
  this->check_state();
}

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::reserve(const size_type new_size)
{
  // note: for 0 new_size, we avoid a wrap-around
  // otherwise we would be reserving quite a lot of memory!
  if (new_size != 0)
    reserve(0, static_cast<indexT>(new_size - 1));
}

// the new members will be initialised with the default constructor for T
template <class T, class indexT>
void
VectorWithOffset<T, indexT>::resize(const indexT min_index, const indexT max_index)
{
  this->check_state();
  if (min_index > max_index)
    {
      length = 0;
      start = 0;
      num = begin_allocated_memory;
      return;
    }
  const size_type old_length = length;
  if (old_length > 0)
    {
      if (min_index == this->get_min_index() && max_index == this->get_max_index())
        return;
      // determine overlapping range to avoid copying too much data when calling reserve()
      const indexT overlap_min_index = std::max(this->get_min_index(), min_index);
      const indexT overlap_max_index = std::min(this->get_max_index(), max_index);
      // TODO when using non-initialised memory, call delete here on elements that go out of range
      length = overlap_max_index - overlap_min_index < 0 ? 0 : static_cast<size_type>(overlap_max_index - overlap_min_index) + 1;
      if (length == 0)
        {
          start = 0;
          num = begin_allocated_memory;
        }
      else
        {
          // do not change num as num[0] should remain the same
          start = overlap_min_index;
        }
    } // end if (length>0)
  const size_type overlapping_length = length;
  this->reserve(min_index, max_index);
  // TODO when using allocator, call default constructor for new elements here
  // (and delete the ones that go out of range!)
  length = static_cast<size_type>(max_index - min_index) + 1;
  start = min_index;
  if (overlapping_length > 0)
    {
      // do not change num as num[0] should remain the same
    }
  else
    {
      // we have reallocated the whole array, so set num correctly
      num = begin_allocated_memory - min_index;
    }
  this->check_state();
}

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::resize(const size_type new_size)
{
  if (new_size == 0)
    {
      length = 0;
      start = 0;
      num = begin_allocated_memory;
    }
  else
    this->resize(0, static_cast<indexT>(new_size - 1));
}

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::grow(const indexT min_index, const indexT max_index)
{
  this->resize(min_index, max_index);
}

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::grow(const size_type new_size)
{
  this->resize(new_size);
}

template <class T, class indexT>
VectorWithOffset<T, indexT>&
VectorWithOffset<T, indexT>::operator=(const VectorWithOffset& il)
{
  this->check_state();
  if (this == &il)
    return *this; // in case of x=x
  {
    if (this->capacity() < il.size())
      {
        // first truncate current and then reserve space
        length = 0;
        start = 0;
        num = begin_allocated_memory;
        this->reserve(il.get_min_index(), il.get_max_index());
      }
    length = il.length;
    this->set_offset(il.get_min_index());
    std::copy(il.begin(), il.end(), this->begin());
  }

  this->check_state();
  return *this;
}

template <class T, class indexT>
VectorWithOffset<T, indexT>::VectorWithOffset(const VectorWithOffset& il)
    : pointer_access(false)
{
  this->init();
  *this = il; // Uses assignment operator (above)
}

template <class T, class indexT>
indexT
VectorWithOffset<T, indexT>::get_length() const
{
  this->check_state();
  return static_cast<indexT>(length);
}

template <class T, class indexT>
size_t
VectorWithOffset<T, indexT>::size() const
{
  this->check_state();
  return size_t(length);
}

template <class T, class indexT>
bool
VectorWithOffset<T, indexT>::operator==(const VectorWithOffset& iv) const
{
  this->check_state();
  if (length != iv.length || start != iv.start)
    return false;
  return std::equal(this->begin(), this->end(), iv.begin());
}

template <class T, class indexT>
bool
VectorWithOffset<T, indexT>::operator!=(const VectorWithOffset& iv) const
{
  return !(*this == iv);
}

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::fill(const T& n)
{
  this->check_state();
  std::fill(this->begin(), this->end(), n);
  this->check_state();
}

template <class T, class indexT>
inline void
VectorWithOffset<T, indexT>::apply_lower_threshold(const T& lower)
{
  this->check_state();
  threshold_lower(this->begin(), this->end(), lower);
  this->check_state();
}

template <class T, class indexT>
inline void
VectorWithOffset<T, indexT>::apply_upper_threshold(const T& upper)
{
  this->check_state();
  threshold_upper(this->begin(), this->end(), upper);
  this->check_state();
}

/*!
  This returns a \c T* to the first element of a,
  members are guaranteed to be stored contiguously in memory.

  Use only in emergency cases...

  To prevent invalidating the safety checks (and making
  reimplementation more difficult), NO manipulation with
  the vector is allowed between the pairs
      get_data_ptr() and release_data_ptr()
  and
      get_const_data_ptr() and release_data_ptr().
  (This is checked with assert() in DEBUG mode.)
*/
template <class T, class indexT>
T*
VectorWithOffset<T, indexT>::get_data_ptr()
{
  assert(!pointer_access);

  pointer_access = true;
  return (num + start);

  // if implementation changes, this would need to keep track
  // if which pointer it returns.
};

/*!
  This returns a \c const \c T* to the first element of a,
  members are guaranteed to be stored contiguously in memory.

  Use get_const_data_ptr() when you are not going to modify
  the data.

  \see get_data_ptr()
*/
template <class T, class indexT>
const T*
VectorWithOffset<T, indexT>::get_const_data_ptr() const
{
  assert(!pointer_access);

  pointer_access = true;
  return (num + start);

  // if implementation changes, this would need to keep track
  // if which pointer it returns.
};

/*!
  This has to be used when access to the T* returned by get_data_ptr() is
  finished. It updates
  the vector with any changes you made, and allows access to
  the other member functions again.

  \see get_data_ptr()
*/
template <class T, class indexT>
void
VectorWithOffset<T, indexT>::release_data_ptr()
{
  assert(pointer_access);

  pointer_access = false;
}

/*!
  This has to be used when access to the const T* returned by get_const_data_ptr() is
  finished. It allows access to
  the other member functions again.

  \see get_const_data_ptr()
*/

template <class T, class indexT>
void
VectorWithOffset<T, indexT>::release_const_data_ptr() const
{
  assert(pointer_access);

  pointer_access = false;
}

/********************** arithmetic operators ****************/
template <class T, class indexT>
inline VectorWithOffset<T, indexT>&
VectorWithOffset<T, indexT>::operator+=(const VectorWithOffset& v)
{
  this->check_state();
#if 1
  if (this->get_min_index() != v.get_min_index() && this->get_max_index() != v.get_max_index())
    error("VectorWithOffset::+= with non-matching range");
#else
  // first check if *this is empty
  if (this->get_length() == 0)
    {
      return *this = v;
    }
  this->grow(std::min(get_min_index(), v.get_min_index()), std::max(get_max_index(), v.get_max_index()));
#endif
  for (auto i = v.get_min_index(); i <= v.get_max_index(); i++)
    num[i] += v.num[i];
  this->check_state();
  return *this;
}

template <class T, class indexT>
inline VectorWithOffset<T, indexT>&
VectorWithOffset<T, indexT>::operator-=(const VectorWithOffset& v)
{
  this->check_state();
#if 1
  if (this->get_min_index() != v.get_min_index() && this->get_max_index() != v.get_max_index())
    error("VectorWithOffset::-= with non-matching range");
#else
  // first check if *this is empty
  if (get_length() == 0)
    {
      *this = v;
      return *this *= -1;
    }
  grow(std::min(get_min_index(), v.get_min_index()), std::max(get_max_index(), v.get_max_index()));
#endif
  for (auto i = v.get_min_index(); i <= v.get_max_index(); i++)
    num[i] -= v.num[i];
  this->check_state();
  return *this;
}

template <class T, class indexT>
inline VectorWithOffset<T, indexT>&
VectorWithOffset<T, indexT>::operator*=(const VectorWithOffset& v)
{
  this->check_state();
#if 1
  if (this->get_min_index() != v.get_min_index() && this->get_max_index() != v.get_max_index())
    error("VectorWithOffset::*= with non-matching range");
#else
  // first check if *this is empty
  if (get_length() == 0)
    {
      // we have to return an object of the same dimensions as v, but filled with 0.
      *this = v;
      return *this *= 0;
    }
  grow(std::min(get_min_index(), v.get_min_index()), std::max(get_max_index(), v.get_max_index()));
#endif
  for (auto i = v.get_min_index(); i <= v.get_max_index(); i++)
    num[i] *= v.num[i];
  this->check_state();
  return *this;
}

template <class T, class indexT>
inline VectorWithOffset<T, indexT>&
VectorWithOffset<T, indexT>::operator/=(const VectorWithOffset& v)
{
  this->check_state();
#if 1
  if (this->get_min_index() != v.get_min_index() && this->get_max_index() != v.get_max_index())
    error("VectorWithOffset::/= with non-matching range");
#else
  // first check if *this is empty
  if (get_length() == 0)
    {
      // we have to return an object of the same dimensions as v, but filled with 0.
      *this = v;
      return *this *= 0;
    }
  grow(std::min(get_min_index(), v.get_min_index()), std::max(get_max_index(), v.get_max_index()));
#endif
  for (auto i = v.get_min_index(); i <= v.get_max_index(); i++)
    num[i] /= v.num[i];
  this->check_state();
  return *this;
}

/**** operator+=(T) etc *****/
#if 0
// disabled for now
// warning: not tested
template <class T, class indexT>
inline VectorWithOffset<T, indexT>& 
VectorWithOffset<T, indexT>::operator+= (const T &t)
{
  typename iterator iter = this->begin();
  const typename iterator end_iter = this->end();
  while (iter != end_iter)
    *iter++ += t;
}

template <class T, class indexT>
inline VectorWithOffset<T, indexT>& 
VectorWithOffset<T, indexT>::operator-= (const T &t)
{
  typename iterator iter = this->begin();
  const typename iterator end_iter = this->end();
  while (iter != end_iter)
    *iter++ -= t;
}
template <class T, class indexT>
inline VectorWithOffset<T, indexT>& 
VectorWithOffset<T, indexT>::operator*= (const T &t)
{
  typename iterator iter = this->begin();
  const typename iterator end_iter = this->end();
  while (iter != end_iter)
    *iter++ *= t;
}
template <class T, class indexT>
inline VectorWithOffset<T, indexT>& 
VectorWithOffset<T, indexT>::operator/= (const T &t)
{
  typename iterator iter = this->begin();
  const typename iterator end_iter = this->end();
  while (iter != end_iter)
    *iter++ /= t;
}

#endif

/**** operator* etc ********/

// addition
template <class T, class indexT>
inline VectorWithOffset<T, indexT>
VectorWithOffset<T, indexT>::operator+(const VectorWithOffset& v) const
{
  this->check_state();
  VectorWithOffset retval(*this);
  return retval += v;
}

// subtraction
template <class T, class indexT>
inline VectorWithOffset<T, indexT>
VectorWithOffset<T, indexT>::operator-(const VectorWithOffset& v) const
{
  this->check_state();
  VectorWithOffset retval(*this);
  return retval -= v;
}

// elem by elem multiplication
template <class T, class indexT>
inline VectorWithOffset<T, indexT>
VectorWithOffset<T, indexT>::operator*(const VectorWithOffset& v) const
{
  this->check_state();
  VectorWithOffset retval(*this);
  return retval *= v;
}

// elem by elem division
template <class T, class indexT>
inline VectorWithOffset<T, indexT>
VectorWithOffset<T, indexT>::operator/(const VectorWithOffset& v) const
{
  this->check_state();
  VectorWithOffset retval(*this);
  return retval /= v;
}

END_NAMESPACE_STIR
