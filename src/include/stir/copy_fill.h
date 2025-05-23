/*
    Copyright (C) 2020 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_copy_fill__H__
#define __stir_copy_fill__H__

/*!
  \file
  \ingroup buildblock
  \brief Declaration of stir::copy_to and stir::fill_from templates
  \author Kris Thielemans
*/

#include "stir/ProjDataInMemory.h"

START_NAMESPACE_STIR
/*!
 \defgroup copy_fill Generic fill/copy functionality for STIR object from/to iterators
 \ingroup buildblock

 \todo current mechanism requires overloading for specific classes. This could be resolved using SFINAE.
 @{
*/


//! Helper class for stir::copy_to and stir::fill_from
/*! Default implementation that uses STIR iterators \c stir_object.begin_all().
 */
template < typename T>
struct CopyFill
{
  template <typename iterT>
  static
    iterT copy_to(const T& stir_object, iterT iter)
    {
      return std::copy(stir_object.begin_all(), stir_object.end_all(), iter);
    }
  template <typename iterT>
  static
  void fill_from(T& stir_object, iterT iter, iterT iter_end)
  {
    std::copy(iter, iter_end, stir_object.begin_all());
  }
};

//! Helper class for stir::copy_to and stir::fill_from
/*! Specialisation that uses ProjData::copy_to etc, unless it's a ProjDataInMemory
 */
template<>
struct CopyFill<ProjData>
{
  template < typename iterT>
    static
    iterT copy_to(const ProjData& stir_object, iterT iter)
    {
#if 1
      if (auto pdm_ptr = dynamic_cast<ProjDataInMemory const *>(&stir_object))
        {
          // std::cerr<<"Using stir::copy_to\n";
          return CopyFill<ProjDataInMemory>::copy_to(*pdm_ptr, iter);
        }
      else
#endif
        {
          // std::cerr<<"Using member copy_to\n";
          return stir_object.copy_to(iter);
        }
    }

  template < typename iterT>
    static
    void fill_from(ProjData& stir_object, iterT iter, iterT iter_end)
    {
      if (auto pdm_ptr = dynamic_cast<ProjDataInMemory *>(&stir_object))
        CopyFill<ProjDataInMemory>::fill_from(*pdm_ptr, iter, iter_end);
      else
        stir_object.fill_from(iter);
    }
};

//! Copy all bins to a range specified by a iterator
/*! 
  \return \a iter advanced over the range (as std::copy)
  
  \warning there is no range-check on \a iter
*/
template <typename T, typename iterT>
  inline iterT copy_to(const T& stir_object, iterT iter)
{
  return CopyFill<T>::copy_to(stir_object, iter);
}

//! set all elements of \a stir_object from an iterator
/*!  
   \warning there is no size/range-check on \a iter
*/
template <typename T, typename iterT>
  inline void fill_from(T& stir_object, iterT iter, iterT iter_end)
{
  //return
  CopyFill<T>::fill_from(stir_object, iter, iter_end);
}

//@}

END_NAMESPACE_STIR

#endif
