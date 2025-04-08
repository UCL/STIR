//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2011-01-11, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2023 - 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Array
  \brief inline implementations for the stir::Array class

  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project


*/
// include for min,max definitions
#include <algorithm>
#include "stir/assign.h"
#include "stir/HigherPrecision.h"
#include "stir/error.h"
//#include "stir/info.h"
//#include <string>

START_NAMESPACE_STIR

/**********************************************
 inlines for Array<num_dimensions, elemT>
 **********************************************/
template <int num_dimensions, typename elemT>
bool
Array<num_dimensions, elemT>::is_contiguous() const
{
  auto mem = &(*this->begin_all());
  for (auto i = this->get_min_index(); i <= this->get_max_index(); ++i)
    {
      if (!(*this)[i].is_contiguous())
        return false;
      if (i == this->get_max_index())
        return true;
      mem += (*this)[i].size_all();
      if (mem != &(*(*this)[i + 1].begin_all()))
        return false;
    }
  return true;
}

template <int num_dimensions, typename elemT>
void
Array<num_dimensions, elemT>::resize(const IndexRange<num_dimensions>& range)
{
  base_type::resize(range.get_min_index(), range.get_max_index());
  typename base_type::iterator iter = this->begin();
  typename IndexRange<num_dimensions>::const_iterator range_iter = range.begin();
  for (; iter != this->end(); ++iter, ++range_iter)
    (*iter).resize(*range_iter);
}

template <int num_dimensions, typename elemT>
void
Array<num_dimensions, elemT>::init(const IndexRange<num_dimensions>& range, elemT* const data_ptr, bool copy_data)
{
  base_type::resize(range.get_min_index(), range.get_max_index());
  auto iter = this->begin();
  auto range_iter = range.begin();
  auto ptr = data_ptr;
  for (; iter != this->end(); ++iter, ++range_iter)
    {
      (*iter).init(*range_iter, ptr, copy_data);
      ptr += range_iter->size_all();
    }
}

template <int num_dimensions, typename elemT>
void
Array<num_dimensions, elemT>::grow(const IndexRange<num_dimensions>& range)
{
  resize(range);
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::Array()
    : base_type(),
      _allocated_full_data_ptr(nullptr)
{}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::Array(const IndexRange<num_dimensions>& range)
    : base_type(),
      _allocated_full_data_ptr(new elemT[range.size_all()])
{
  // info("Array constructor range " + std::to_string(reinterpret_cast<std::size_t>(this->_allocated_full_data_ptr)) + " of size "
  // + std::to_string(range.size_all())); set elements to zero
  std::for_each(this->_allocated_full_data_ptr.get(), this->_allocated_full_data_ptr.get() + range.size_all(), [](elemT& e) {
    assign(e, 0);
  });
  this->init(range, this->_allocated_full_data_ptr.get(), false);
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::Array(const IndexRange<num_dimensions>& range, shared_ptr<elemT[]> data_sptr)
{
  this->_allocated_full_data_ptr = data_sptr;
  this->init(range, this->_allocated_full_data_ptr.get(), false);
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::Array(const self& t)
    : base_type(t),
      _allocated_full_data_ptr(nullptr)
{
  // info("constructor " + std::to_string(num_dimensions) + "copy of size " + std::to_string(this->size_all()));
}

#ifndef SWIG
// swig cannot parse this ATM, but we don't need it anyway in the wrappers
template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::Array(const base_type& t)
    : base_type(t),
      _allocated_full_data_ptr(nullptr)
{
  // info("constructor basetype " + std::to_string(num_dimensions) + " of size " + std::to_string(this->size_all()));
}
#endif

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::~Array()
{
  if (this->_allocated_full_data_ptr)
    {
      // info("Array destructor full_data_ptr " + std::to_string(reinterpret_cast<std::size_t>(this->_allocated_full_data_ptr)) +
      // " of size " + std::to_string(this->size_all())); delete [] this->_allocated_full_data_ptr;
    }
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::Array(Array<num_dimensions, elemT>&& other) noexcept
    : Array()
{
  swap(*this, other);
  // info("move constructor " + std::to_string(num_dimensions) + "copy of size " + std::to_string(this->size_all()));
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>&
Array<num_dimensions, elemT>::operator=(Array<num_dimensions, elemT> other)
{
  swap(*this, other);
  // info("Array= " + std::to_string(num_dimensions) + "copy of size " + std::to_string(this->size_all()));
  return *this;
}

template <int num_dimensions, typename elemT>
typename Array<num_dimensions, elemT>::full_iterator
Array<num_dimensions, elemT>::end_all()
{
  // note this value is fixed by the current convention in full_iterator::operator++()
  return full_iterator(this->end(),
                       this->end(),
                       typename Array<num_dimensions - 1, elemT>::full_iterator(0),
                       typename Array<num_dimensions - 1, elemT>::full_iterator(0));
}

template <int num_dimensions, typename elemT>
typename Array<num_dimensions, elemT>::const_full_iterator
Array<num_dimensions, elemT>::end_all_const() const
{
  return const_full_iterator(this->end(),
                             this->end(),
                             typename Array<num_dimensions - 1, elemT>::const_full_iterator(0),
                             typename Array<num_dimensions - 1, elemT>::const_full_iterator(0));
}

template <int num_dimensions, typename elemT>
typename Array<num_dimensions, elemT>::const_full_iterator
Array<num_dimensions, elemT>::end_all() const
{
  return this->end_all_const();
}

template <int num_dimensions, typename elemT>
typename Array<num_dimensions, elemT>::full_iterator
Array<num_dimensions, elemT>::begin_all()
{
  if (this->begin() == this->end())
    {
      // empty array
      return end_all();
    }
  else
    return full_iterator(this->begin(), this->end(), this->begin()->begin_all(), this->begin()->end_all());
}

template <int num_dimensions, typename elemT>
typename Array<num_dimensions, elemT>::const_full_iterator
Array<num_dimensions, elemT>::begin_all_const() const
{
  if (this->begin() == this->end())
    {
      // empty array
      return end_all();
    }
  else
    return const_full_iterator(this->begin(), this->end(), this->begin()->begin_all_const(), this->begin()->end_all_const());
}

template <int num_dimensions, typename elemT>
typename Array<num_dimensions, elemT>::const_full_iterator
Array<num_dimensions, elemT>::begin_all() const
{
  return begin_all_const();
}

template <int num_dimensions, class elemT>
IndexRange<num_dimensions>
Array<num_dimensions, elemT>::get_index_range() const
{
  VectorWithOffset<IndexRange<num_dimensions - 1>> range(this->get_min_index(), this->get_max_index());

  typename VectorWithOffset<IndexRange<num_dimensions - 1>>::iterator range_iter = range.begin();
  const_iterator array_iter = this->begin();

  for (; range_iter != range.end(); range_iter++, array_iter++)
    {
      *range_iter = (*array_iter).get_index_range();
    }
  return IndexRange<num_dimensions>(range);
}

template <int num_dimensions, typename elemT>
size_t
Array<num_dimensions, elemT>::size_all() const
{
  this->check_state();
  size_t acc = 0;
#ifdef STIR_OPENMP
#  if _OPENMP >= 201107
#    pragma omp parallel for reduction(+ : acc)
#  endif
#endif
  for (int i = this->get_min_index(); i <= this->get_max_index(); i++)
    acc += this->num[i].size_all();
  return acc;
}

/*!
  If is_contiguous() is \c false, calls error(). Otherwise, return a
  \c elemT* to the first element of the array.

  Use only in emergency cases...

  To prevent invalidating the safety checks (and making
  reimplementation more difficult), NO manipulation with
  the array is allowed between the pairs
      get_full_data_ptr() and release_full_data_ptr()
  and
      get_const_full_data_ptr() and release_const_full_data_ptr().
  (This is checked with assert() in DEBUG mode.)
*/
template <int num_dimensions, typename elemT>
elemT*
Array<num_dimensions, elemT>::get_full_data_ptr()
{
  this->_full_pointer_access = true;
  if (!this->is_contiguous())
    error("Array::get_full_data_ptr() called for non-contiguous array.");
  return &(*this->begin_all());
};

/*!
  If is_contiguous() is \c false, calls error(). Otherwise, return a
  \c const \c elemT* to the first element of the array.

  Use get_const_full_data_ptr() when you are not going to modify
  the data.

  \see get_full_data_ptr()
*/
template <int num_dimensions, typename elemT>
const elemT*
Array<num_dimensions, elemT>::get_const_full_data_ptr() const
{
  this->_full_pointer_access = true;
  if (!this->is_contiguous())
    error("Array::get_const_full_data_ptr() called for non-contiguous array.");
  return &(*this->begin_all_const());
};

/*!
  This has to be used when access to the elemT* returned by get_full_data_ptr() is
  finished. It updates
  the Array with any changes you made, and allows access to
  the other member functions again.

  \see get_full_data_ptr()
*/
template <int num_dimensions, typename elemT>
void
Array<num_dimensions, elemT>::release_full_data_ptr()
{
  assert(this->_full_pointer_access);

  this->_full_pointer_access = false;
}

/*!
  This has to be used when access to the const elemT* returned by get_const_full_data_ptr() is
  finished. It allows access to the other member functions again.

  \see get_const_full_data_ptr()
*/

template <int num_dimensions, typename elemT>
void
Array<num_dimensions, elemT>::release_const_full_data_ptr() const
{
  assert(this->_full_pointer_access);
  this->_full_pointer_access = false;
}

template <int num_dimensions, typename elemT>
elemT
Array<num_dimensions, elemT>::sum() const
{
  this->check_state();
  typename HigherPrecision<elemT>::type acc;
  assign(acc, 0);
#ifdef STIR_OPENMP
#  if _OPENMP >= 201107
#    pragma omp parallel for reduction(+ : acc)
#  endif
#endif
  for (int i = this->get_min_index(); i <= this->get_max_index(); i++)
    acc += this->num[i].sum();
  return static_cast<elemT>(acc);
}

template <int num_dimensions, typename elemT>
elemT
Array<num_dimensions, elemT>::sum_positive() const
{
  this->check_state();
  typename HigherPrecision<elemT>::type acc;
  assign(acc, 0);
#ifdef STIR_OPENMP
#  if _OPENMP >= 201107
#    pragma omp parallel for reduction(+ : acc)
#  endif
#endif
  for (int i = this->get_min_index(); i <= this->get_max_index(); i++)
    acc += this->num[i].sum_positive();
  return static_cast<elemT>(acc);
}

template <int num_dimensions, typename elemT>
elemT
Array<num_dimensions, elemT>::find_max() const
{
  this->check_state();
  if (this->size() > 0)
    {
      elemT maxval = this->num[this->get_min_index()].find_max();
#ifdef STIR_OPENMP
#  if _OPENMP >= 201107
#    pragma omp parallel for reduction(max : maxval)
#  endif
#endif
      for (int i = this->get_min_index() + 1; i <= this->get_max_index(); i++)
        {
          maxval = std::max(this->num[i].find_max(), maxval);
        }
      return maxval;
    }
  else
    {
      // TODO we should return elemT::minimum or something
      return 0;
    }
}

template <int num_dimensions, typename elemT>
elemT
Array<num_dimensions, elemT>::find_min() const
{
  this->check_state();
  if (this->size() > 0)
    {
      elemT minval = this->num[this->get_min_index()].find_min();
#ifdef STIR_OPENMP
#  if _OPENMP >= 201107
#    pragma omp parallel for reduction(min : minval)
#  endif
#endif
      for (int i = this->get_min_index() + 1; i <= this->get_max_index(); i++)
        {
          minval = std::min(this->num[i].find_min(), minval);
        }
      return minval;
    }
  else
    {
      // TODO we should return elemT::maximum or something
      return 0;
    }
}

template <int num_dimensions, typename elemT>
void
Array<num_dimensions, elemT>::fill(const elemT& n)
{
  this->check_state();
  for (int i = this->get_min_index(); i <= this->get_max_index(); i++)
    this->num[i].fill(n);
  this->check_state();
}

template <int num_dimensions, typename elemT>
void
Array<num_dimensions, elemT>::apply_lower_threshold(const elemT& l)
{
  this->check_state();
  for (int i = this->get_min_index(); i <= this->get_max_index(); i++)
    this->num[i].apply_lower_threshold(l);
  this->check_state();
}

template <int num_dimensions, typename elemT>
void
Array<num_dimensions, elemT>::apply_upper_threshold(const elemT& u)
{
  this->check_state();
  for (int i = this->get_min_index(); i <= this->get_max_index(); i++)
    this->num[i].apply_upper_threshold(u);
  this->check_state();
}

template <int num_dimensions, typename elemT>
bool
Array<num_dimensions, elemT>::is_regular() const
{
  return get_index_range().is_regular();
}

// TODO terribly inefficient at the moment
template <int num_dimensions, typename elemT>
bool
Array<num_dimensions, elemT>::get_regular_range(BasicCoordinate<num_dimensions, int>& min,
                                                BasicCoordinate<num_dimensions, int>& max) const
{
  const IndexRange<num_dimensions> range = get_index_range();
  return range.get_regular_range(min, max);
}

template <int num_dimension, typename elemT>
Array<num_dimension - 1, elemT>&
Array<num_dimension, elemT>::operator[](int i)
{
  return base_type::operator[](i);
}

template <int num_dimension, typename elemT>
const Array<num_dimension - 1, elemT>&
Array<num_dimension, elemT>::operator[](int i) const
{
  return base_type::operator[](i);
}
template <int num_dimensions, typename elemT>
elemT&
Array<num_dimensions, elemT>::operator[](const BasicCoordinate<num_dimensions, int>& c)
{
  return (*this)[c[1]][cut_first_dimension(c)];
}
template <int num_dimensions, typename elemT>
const elemT&
Array<num_dimensions, elemT>::operator[](const BasicCoordinate<num_dimensions, int>& c) const
{
  return (*this)[c[1]][cut_first_dimension(c)];
}

template <int num_dimension, typename elemT>
Array<num_dimension - 1, elemT>&
Array<num_dimension, elemT>::at(int i)
{
  return base_type::at(i);
}

template <int num_dimension, typename elemT>
const Array<num_dimension - 1, elemT>&
Array<num_dimension, elemT>::at(int i) const
{
  return base_type::at(i);
}
template <int num_dimensions, typename elemT>
elemT&
Array<num_dimensions, elemT>::at(const BasicCoordinate<num_dimensions, int>& c)
{
  return (*this).at(c[1]).at(cut_first_dimension(c));
}
template <int num_dimensions, typename elemT>
const elemT&
Array<num_dimensions, elemT>::at(const BasicCoordinate<num_dimensions, int>& c) const
{
  return (*this).at(c[1]).at(cut_first_dimension(c));
}

template <int num_dimensions, typename elemT>
template <typename elemT2>
void
Array<num_dimensions, elemT>::axpby(const elemT2 a, const Array& x, const elemT2 b, const Array& y)
{
  Array<num_dimensions, elemT>::xapyb(x, a, y, b);
}

template <int num_dimensions, typename elemT>
void
Array<num_dimensions, elemT>::xapyb(const Array& x, const elemT a, const Array& y, const elemT b)
{
  this->check_state();
  if ((this->get_index_range() != x.get_index_range()) || (this->get_index_range() != y.get_index_range()))
    error("Array::xapyb: index ranges don't match");

  typename Array::full_iterator this_iter = this->begin_all();
  typename Array::const_full_iterator x_iter = x.begin_all();
  typename Array::const_full_iterator y_iter = y.begin_all();
  while (this_iter != this->end_all())
    {
      *this_iter++ = (*x_iter++) * a + (*y_iter++) * b;
    }
}

template <int num_dimensions, typename elemT>
void
Array<num_dimensions, elemT>::xapyb(const Array& x, const Array& a, const Array& y, const Array& b)
{
  this->check_state();
  if ((this->get_index_range() != x.get_index_range()) || (this->get_index_range() != y.get_index_range())
      || (this->get_index_range() != a.get_index_range()) || (this->get_index_range() != b.get_index_range()))
    error("Array::xapyb: index ranges don't match");

  typename Array::full_iterator this_iter = this->begin_all();
  typename Array::const_full_iterator x_iter = x.begin_all();
  typename Array::const_full_iterator y_iter = y.begin_all();
  typename Array::const_full_iterator a_iter = a.begin_all();
  typename Array::const_full_iterator b_iter = b.begin_all();

  while (this_iter != this->end_all())
    {
      *this_iter++ = (*x_iter++) * (*a_iter++) + (*y_iter++) * (*b_iter++);
    }
}

template <int num_dimensions, typename elemT>
template <class T>
void
Array<num_dimensions, elemT>::sapyb(const T& a, const Array& y, const T& b)
{
  this->xapyb(*this, a, y, b);
}

/**********************************************
 inlines for Array<1, elemT>
 **********************************************/
template <class elemT>
void
Array<1, elemT>::init(const IndexRange<1>& range, elemT* const data_ptr, bool copy_data)
{
  base_type::init(range.get_min_index(), range.get_max_index(), data_ptr, copy_data);
}

template <class elemT>
void
Array<1, elemT>::resize(const int min_index, const int max_index, bool initialise_with_0)
{
  this->check_state();
  const int oldstart = this->get_min_index();
  const size_type oldlength = this->size();

  base_type::resize(min_index, max_index);

  if (!initialise_with_0)
    {
      this->check_state();
      return;
    }

  if (oldlength == 0)
    {
      for (int i = this->get_min_index(); i <= this->get_max_index(); i++)
        assign(this->num[i], 0);
    }
  else
    {
      for (int i = this->get_min_index(); i < oldstart && i <= this->get_max_index(); ++i)
        assign(this->num[i], 0);
      for (int i = std::max(static_cast<int>(oldstart + oldlength), this->get_min_index()); i <= this->get_max_index(); ++i)
        assign(this->num[i], 0);
    }
  this->check_state();
}

template <class elemT>
void
Array<1, elemT>::resize(const int min_index, const int max_index)
{
  resize(min_index, max_index, true);
}

template <class elemT>
void
Array<1, elemT>::resize(const IndexRange<1>& range)
{
  resize(range.get_min_index(), range.get_max_index());
}

template <class elemT>
void
Array<1, elemT>::grow(const int min_index, const int max_index)
{
  resize(min_index, max_index);
}

template <class elemT>
void
Array<1, elemT>::grow(const IndexRange<1>& range)
{
  grow(range.get_min_index(), range.get_max_index());
}

template <class elemT>
Array<1, elemT>::Array()
    : base_type()
{}

template <class elemT>
Array<1, elemT>::Array(const IndexRange<1>& range)
    : base_type()
{
  grow(range);
}

template <class elemT>
Array<1, elemT>::Array(const int min_index, const int max_index)
    : base_type()
{
  grow(min_index, max_index);
}

template <class elemT>
Array<1, elemT>::Array(const IndexRange<1>& range, shared_ptr<elemT[]> data_sptr)
    : base_type(range.get_min_index(), range.get_max_index(), data_sptr)
{}

template <class elemT>
Array<1, elemT>::Array(const IndexRange<1>& range, const elemT* const data_ptr)
    : base_type(range.get_min_index(), range.get_max_index(), data_ptr)
{}

template <class elemT>
Array<1, elemT>::Array(const base_type& il)
    : base_type(il)
{}

template <typename elemT>
Array<1, elemT>::Array(const Array<1, elemT>& other)
    : base_type(other)
{}

template <typename elemT>
Array<1, elemT>::~Array()
{}

template <typename elemT>
Array<1, elemT>::Array(Array<1, elemT>&& other) noexcept
    : Array()
{
  swap(*this, other);
}

template <typename elemT>
Array<1, elemT>&
Array<1, elemT>::operator=(const Array<1, elemT>& other)
{
  // use the base_type assignment, as this tries to avoid reallocating memory
  base_type::operator=(other);
  return *this;
}

template <typename elemT>
typename Array<1, elemT>::full_iterator
Array<1, elemT>::begin_all()
{
  return this->begin();
}

template <typename elemT>
typename Array<1, elemT>::const_full_iterator
Array<1, elemT>::begin_all() const
{
  return this->begin();
}

template <typename elemT>
typename Array<1, elemT>::full_iterator
Array<1, elemT>::end_all()
{
  return this->end();
}

template <typename elemT>
typename Array<1, elemT>::const_full_iterator
Array<1, elemT>::end_all() const
{
  return this->end();
}

template <typename elemT>
typename Array<1, elemT>::const_full_iterator
Array<1, elemT>::begin_all_const() const
{
  return this->begin();
}

template <typename elemT>
typename Array<1, elemT>::const_full_iterator
Array<1, elemT>::end_all_const() const
{
  return this->end();
}

template <typename elemT>
IndexRange<1>
Array<1, elemT>::get_index_range() const
{
  return IndexRange<1>(this->get_min_index(), this->get_max_index());
}

template <typename elemT>
size_t
Array<1, elemT>::size_all() const
{
  return this->size();
}

template <class elemT>
elemT
Array<1, elemT>::sum() const
{
  this->check_state();
  typename HigherPrecision<elemT>::type acc;
  assign(acc, 0);
#ifdef STIR_OPENMP
#  if _OPENMP >= 201107
#    pragma omp parallel for reduction(+ : acc)
#  endif
#endif
  for (int i = this->get_min_index(); i <= this->get_max_index(); ++i)
    acc += this->num[i];
  return static_cast<elemT>(acc);
};

template <class elemT>
elemT
Array<1, elemT>::sum_positive() const
{
  this->check_state();
  typename HigherPrecision<elemT>::type acc;
  assign(acc, 0);
#ifdef STIR_OPENMP
#  if _OPENMP >= 201107
#    pragma omp parallel for reduction(+ : acc)
#  endif
#endif
  for (int i = this->get_min_index(); i <= this->get_max_index(); i++)
    {
      if (this->num[i] > 0)
        acc += this->num[i];
    }
  return static_cast<elemT>(acc);
};

template <class elemT>
elemT
Array<1, elemT>::find_max() const
{
  this->check_state();
  if (this->size() > 0)
    {
      return *std::max_element(this->begin(), this->end());
    }
  else
    {
      // TODO return elemT::minimum or so
      return 0;
    }
  this->check_state();
};

template <class elemT>
elemT
Array<1, elemT>::find_min() const
{
  this->check_state();
  if (this->size() > 0)
    {
      return *std::min_element(this->begin(), this->end());
    }
  else
    {
      // TODO return elemT::maximum or so
      return 0;
    }
  this->check_state();
};

template <typename elemT>
bool
Array<1, elemT>::is_regular() const
{
  return true;
}

template <typename elemT>
bool
Array<1, elemT>::get_regular_range(BasicCoordinate<1, int>& min, BasicCoordinate<1, int>& max) const
{
  const IndexRange<1> range = get_index_range();
  return range.get_regular_range(min, max);
}

#ifndef STIR_USE_BOOST

/* KT 31/01/2000 I had to add these functions here, although they are
in NumericVectorWithOffset already.
Reason: we allow addition (and similar operations) of tensors of
different sizes. This implies that operator+= can call a 'grow'
on retval. For this to work, retval should be a Array, not
its base_type (which happens if these function are not repeated
in this class).
Complicated...
*/
template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator+(const base_type& iv) const
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return retval += iv;
};

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator-(const base_type& iv) const
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return retval -= iv;
}
template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator*(const base_type& iv) const
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return retval *= iv;
}

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator/(const base_type& iv) const
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return retval /= iv;
}

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator+(const elemT a) const
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return (retval += a);
};

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator-(const elemT a) const
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return (retval -= a);
};

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator*(const elemT a) const
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return (retval *= a);
};

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator/(const elemT a) const
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return (retval /= a);
};

#endif // boost

template <typename elemT>
const elemT&
Array<1, elemT>::operator[](int i) const
{
  return base_type::operator[](i);
};

template <typename elemT>
elemT&
Array<1, elemT>::operator[](int i)
{
  return base_type::operator[](i);
};

template <typename elemT>
const elemT&
Array<1, elemT>::operator[](const BasicCoordinate<1, int>& c) const
{
  return (*this)[c[1]];
};

template <typename elemT>
elemT&
Array<1, elemT>::operator[](const BasicCoordinate<1, int>& c)
{
  return (*this)[c[1]];
};

template <typename elemT>
const elemT&
Array<1, elemT>::at(int i) const
{
  return base_type::at(i);
};

template <typename elemT>
elemT&
Array<1, elemT>::at(int i)
{
  return base_type::at(i);
};

template <typename elemT>
const elemT&
Array<1, elemT>::at(const BasicCoordinate<1, int>& c) const
{
  return (*this).at(c[1]);
};

template <typename elemT>
elemT&
Array<1, elemT>::at(const BasicCoordinate<1, int>& c)
{
  return (*this).at(c[1]);
};

END_NAMESPACE_STIR
