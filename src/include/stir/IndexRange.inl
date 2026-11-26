//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2005, Hammersmith Imanet Ltd
    Copyright (C) 2025-2026, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup Array
  \brief inline definitions for the stir::IndexRange class

  \author Kris Thielemans
  \author PARAPET project



*/
#include <algorithm>
#include <type_traits>

START_NAMESPACE_STIR

/***************************************
 n-D version
 ***************************************/

template <int num_dimensions, typename indexT>
IndexRange<num_dimensions, indexT>::IndexRange()
    : base_type(),
      is_regular_range(regular_true)
{}

template <int num_dimensions, typename indexT>
IndexRange<num_dimensions, indexT>::IndexRange(const IndexRange<num_dimensions, indexT>& range)
    : base_type(range),
      is_regular_range(range.is_regular_range)
{}

template <int num_dimensions, typename indexT>
IndexRange<num_dimensions, indexT>::IndexRange(const base_type& range)
    : base_type(range),
      is_regular_range(regular_to_do)
{}

template <int num_dimensions, typename indexT>
IndexRange<num_dimensions, indexT>::IndexRange(const BasicCoordinate<num_dimensions, indexT>& min_v,
                                               const BasicCoordinate<num_dimensions, indexT>& max_v)
    : base_type(min_v[1], max_v[1]),
      is_regular_range(regular_true)
{
  const IndexRange<num_dimensions - 1, indexT> lower_dims(cut_first_dimension(min_v), cut_first_dimension(max_v));
  this->fill(lower_dims);
}

template <int num_dimensions, typename indexT>
IndexRange<num_dimensions, indexT>::IndexRange(const BasicCoordinate<num_dimensions, indexT>& sizes)
    : base_type(sizes[1]),
      is_regular_range(regular_true)
{
  const IndexRange<num_dimensions - 1, indexT> lower_dims(cut_first_dimension(sizes));
  this->fill(lower_dims);
}

template <int num_dimensions, typename indexT>
bool
IndexRange<num_dimensions, indexT>::empty() const
{
  this->check_state();
  if (base_type::empty())
    return true;
  if (this->is_regular_range == regular_true)
    return this->begin()->empty();
  // else
  for (auto i : *this)
    if (i.empty())
      return true;
  return false;
}
template <int num_dimensions, typename indexT>
std::size_t
IndexRange<num_dimensions, indexT>::size_all() const
{
  this->check_state();
  if (this->empty())
    return std::size_t(0);
  if (this->is_regular_range == regular_true)
    return this->get_length() * this->begin()->size_all();
  // else
  size_t acc = 0;
  for (indexT i = this->get_min_index(); i <= this->get_max_index(); i++)
    acc += this->num[i].size_all();
  return acc;
}

template <int num_dimensions, typename indexT>
bool
IndexRange<num_dimensions, indexT>::operator==(const IndexRange<num_dimensions, indexT>& range2) const
{
  return this->get_min_index() == range2.get_min_index() && this->get_length() == range2.get_length()
         && std::equal(this->begin(), this->end(), range2.begin());
}

template <int num_dimensions, typename indexT>
bool
IndexRange<num_dimensions, indexT>::operator!=(const IndexRange<num_dimensions, indexT>& range2) const
{
  return !(*this == range2);
}

template <int num_dimensions, typename indexT>
bool
IndexRange<num_dimensions, indexT>::get_regular_range(BasicCoordinate<num_dimensions, indexT>& min,
                                                      BasicCoordinate<num_dimensions, indexT>& max) const
{
  if (base_type::empty())
    {
      // use empty range (suitable for unsigned indexT)
      std::fill(min.begin(), min.end(), 1);
      std::fill(max.begin(), max.end(), 0);
      return true;
    }

  // if not a regular range, exit
  if (is_regular_range == regular_false)
    return false;

  typename base_type::const_iterator iter = base_type::begin();

  BasicCoordinate<num_dimensions - 1, indexT> lower_dim_min;
  BasicCoordinate<num_dimensions - 1, indexT> lower_dim_max;
  if (!iter->get_regular_range(lower_dim_min, lower_dim_max))
    return false;

  if (is_regular_range == regular_to_do)
    {
      // check if all lower dimensional ranges have same regular range
      BasicCoordinate<num_dimensions - 1, indexT> lower_dim_min_try;
      BasicCoordinate<num_dimensions - 1, indexT> lower_dim_max_try;

      for (++iter; iter != base_type::end(); ++iter)
        {
          if (!iter->get_regular_range(lower_dim_min_try, lower_dim_max_try))
            {
              is_regular_range = regular_false;
              return false;
            }
          if (!std::equal(lower_dim_min.begin(), lower_dim_min.end(), lower_dim_min_try.begin())
              || !std::equal(lower_dim_max.begin(), lower_dim_max.end(), lower_dim_max_try.begin()))
            {
              is_regular_range = regular_false;
              return false;
            }
        }
      // yes, they do
      is_regular_range = regular_true;
    }

  min = join(base_type::get_min_index(), lower_dim_min);
  max = join(base_type::get_max_index(), lower_dim_max);

  return true;
}

template <int num_dimensions, typename indexT>
bool
IndexRange<num_dimensions, indexT>::is_regular() const
{
  switch (is_regular_range)
    {
    case regular_true:
      return true;
    case regular_false:
      return false;
      case regular_to_do: {
        BasicCoordinate<num_dimensions, indexT> min;
        BasicCoordinate<num_dimensions, indexT> max;
        return get_regular_range(min, max);
      }
    }
  // although we never get here, VC insists on a return value...
  // we check anyway
  assert(false);
  return true;
}

/***************************************
 1D version
 ***************************************/

template <typename indexT>
bool
IndexRange<1, indexT>::empty() const
{
  return max < min;
}

template <typename indexT>
void
IndexRange<1, indexT>::recycle()
{
  // note: max - min + 1 needs to be 0 (as we don't do a check in size())
  // note: use (1,0) as opposed to (0,-1) to avoid problem with unsigned indexT
  this->min = indexT(1);
  this->max = indexT(0);
}

template <typename indexT>
IndexRange<1, indexT>::IndexRange()
{
  this->recycle();
}

template <typename indexT>
IndexRange<1, indexT>::IndexRange(const indexT min_v, const indexT max_v)
    : min(min_v),
      max(max_v)
{
  if (max_v < min_v)
    this->recycle();
}

template <typename indexT>
IndexRange<1, indexT>::IndexRange(const BasicCoordinate<1, indexT>& min_v, const BasicCoordinate<1, indexT>& max_v)
    : IndexRange<1, indexT>(min_v[1], max_v[1])
{}

template <typename indexT>
IndexRange<1, indexT>::IndexRange(const size_type sz)
    : min((std::is_signed_v<indexT> || sz > 0) ? indexT(0) : indexT(1)),
      max((std::is_signed_v<indexT> || sz > 0) ? indexT(sz - 1) : indexT(0))
{}

template <typename indexT>
IndexRange<1, indexT>::IndexRange(const BasicCoordinate<1, indexT>& size)
    : IndexRange<1, indexT>(size[1])
{}

template <typename indexT>
indexT
IndexRange<1, indexT>::get_min_index() const
{
  return min;
}

template <typename indexT>
indexT
IndexRange<1, indexT>::get_max_index() const
{
  return max;
}

template <typename indexT>
typename IndexRange<1, indexT>::size_type
IndexRange<1, indexT>::get_length() const
{
  return static_cast<size_type>((max + 1) - min); // note: this order of calculation avoids problems with unsigned indexT
}

template <typename indexT>
std::size_t
IndexRange<1, indexT>::size_all() const
{
  return std::size_t(this->get_length());
}

template <typename indexT>
bool
IndexRange<1, indexT>::operator==(const IndexRange<1, indexT>& range2) const
{
  return get_min_index() == range2.get_min_index() && get_length() == range2.get_length();
}

template <typename indexT>
bool
IndexRange<1, indexT>::is_regular() const
{
  // 1D case: always true
  return true;
}

template <typename indexT>
bool
IndexRange<1, indexT>::get_regular_range(BasicCoordinate<1, indexT>& min_v, BasicCoordinate<1, indexT>& max_v) const
{
  // somewhat complicated as we can't assign ints to BasicCoordinate<1,indexT>
  BasicCoordinate<1, indexT> tmp;
  tmp[1] = min;
  min_v = tmp;
  tmp[1] = max;
  max_v = tmp;
  return true;
}

template <typename indexT>
void
IndexRange<1, indexT>::resize(const indexT min_index, const indexT max_index)
{
  if (max_index < min_index)
    {
      this->recycle();
      return;
    }
  min = min_index;
  max = max_index;
}
END_NAMESPACE_STIR
