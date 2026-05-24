//
//
/*
    Copyright (C) 2005- 2005, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_make_array_H__
#define __stir_make_array_H__
/*!
  \file
  \ingroup Array

  \brief Declaration of functions for constructing arrays stir::make_1d_array etc

  \author Kris Thielemans

*/
#include "stir/Array.h"

START_NAMESPACE_STIR

template <class T>
inline VectorWithOffset<T> make_vector(const T& a0);

template <class T>
inline VectorWithOffset<T> make_vector(const T& a0, const T& a1);

template <class T>
inline VectorWithOffset<T> make_vector(const T& a0, const T& a1, const T& a2);

template <class T>
inline VectorWithOffset<T> make_vector(const T& a0, const T& a1, const T& a2, const T& a3);

template <class T>
inline VectorWithOffset<T> make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4);

template <class T>
inline VectorWithOffset<T> make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5);

template <class T>
inline VectorWithOffset<T> make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6);

template <class T>
inline VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6, const T& a7);

template <class T>
inline VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6, const T& a7, const T& a8);

template <class T>
inline VectorWithOffset<T> make_vector(const T& a0,
                                       const T& a1,
                                       const T& a2,
                                       const T& a3,
                                       const T& a4,
                                       const T& a5,
                                       const T& a6,
                                       const T& a7,
                                       const T& a8,
                                       const T& a9);

template <class T>
inline Array<1, T> make_1d_array(const T& a0);

template <class T>
inline Array<1, T> make_1d_array(const T& a0, const T& a1);

template <class T>
inline Array<1, T> make_1d_array(const T& a0, const T& a1, const T& a2);

template <class T>
inline Array<1, T> make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3);

template <class T>
inline Array<1, T> make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4);

template <class T>
inline Array<1, T> make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5);

template <class T>
inline Array<1, T> make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6);

template <class T>
inline Array<1, T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6, const T& a7);

template <class T>
inline Array<1, T> make_1d_array(
    const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6, const T& a7, const T& a8);

template <class T>
inline Array<1, T> make_1d_array(const T& a0,
                                 const T& a1,
                                 const T& a2,
                                 const T& a3,
                                 const T& a4,
                                 const T& a5,
                                 const T& a6,
                                 const T& a7,
                                 const T& a8,
                                 const T& a9);

template <int num_dimensions, typename elemT, typename indexT>
inline Array<num_dimensions + 1, elemT, indexT> make_array(const Array<num_dimensions, elemT, indexT>& a0);

template <int num_dimensions, typename elemT, typename indexT>
inline Array<num_dimensions + 1, elemT, indexT> make_array(const Array<num_dimensions, elemT, indexT>& a0,
                                                           const Array<num_dimensions, elemT, indexT>& a1);

template <int num_dimensions, typename elemT, typename indexT>
inline Array<num_dimensions + 1, elemT, indexT> make_array(const Array<num_dimensions, elemT, indexT>& a0,
                                                           const Array<num_dimensions, elemT, indexT>& a1,
                                                           const Array<num_dimensions, elemT, indexT>& a2);

template <int num_dimensions, typename elemT, typename indexT>
inline Array<num_dimensions + 1, elemT, indexT> make_array(const Array<num_dimensions, elemT, indexT>& a0,
                                                           const Array<num_dimensions, elemT, indexT>& a1,
                                                           const Array<num_dimensions, elemT, indexT>& a2,
                                                           const Array<num_dimensions, elemT, indexT>& a3);

template <int num_dimensions, typename elemT, typename indexT>
inline Array<num_dimensions + 1, elemT, indexT> make_array(const Array<num_dimensions, elemT, indexT>& a0,
                                                           const Array<num_dimensions, elemT, indexT>& a1,
                                                           const Array<num_dimensions, elemT, indexT>& a2,
                                                           const Array<num_dimensions, elemT, indexT>& a3,
                                                           const Array<num_dimensions, elemT, indexT>& a4);

template <int num_dimensions, typename elemT, typename indexT>
inline Array<num_dimensions + 1, elemT, indexT> make_array(const Array<num_dimensions, elemT, indexT>& a0,
                                                           const Array<num_dimensions, elemT, indexT>& a1,
                                                           const Array<num_dimensions, elemT, indexT>& a2,
                                                           const Array<num_dimensions, elemT, indexT>& a3,
                                                           const Array<num_dimensions, elemT, indexT>& a4,
                                                           Array<num_dimensions, elemT, indexT>& a5);

template <int num_dimensions, typename elemT, typename indexT>
inline Array<num_dimensions + 1, elemT, indexT> make_array(const Array<num_dimensions, elemT, indexT>& a0,
                                                           const Array<num_dimensions, elemT, indexT>& a1,
                                                           const Array<num_dimensions, elemT, indexT>& a2,
                                                           const Array<num_dimensions, elemT, indexT>& a3,
                                                           const Array<num_dimensions, elemT, indexT>& a4,
                                                           Array<num_dimensions, elemT, indexT>& a5,
                                                           const Array<num_dimensions, elemT, indexT>& a6);

template <int num_dimensions, typename elemT, typename indexT>
inline Array<num_dimensions + 1, elemT, indexT> make_array(const Array<num_dimensions, elemT, indexT>& a0,
                                                           const Array<num_dimensions, elemT, indexT>& a1,
                                                           const Array<num_dimensions, elemT, indexT>& a2,
                                                           const Array<num_dimensions, elemT, indexT>& a3,
                                                           const Array<num_dimensions, elemT, indexT>& a4,
                                                           Array<num_dimensions, elemT, indexT>& a5,
                                                           const Array<num_dimensions, elemT, indexT>& a6,
                                                           const Array<num_dimensions, elemT, indexT>& a7);

template <int num_dimensions, typename elemT, typename indexT>
inline Array<num_dimensions + 1, elemT, indexT> make_array(const Array<num_dimensions, elemT, indexT>& a0,
                                                           const Array<num_dimensions, elemT, indexT>& a1,
                                                           const Array<num_dimensions, elemT, indexT>& a2,
                                                           const Array<num_dimensions, elemT, indexT>& a3,
                                                           const Array<num_dimensions, elemT, indexT>& a4,
                                                           Array<num_dimensions, elemT, indexT>& a5,
                                                           const Array<num_dimensions, elemT, indexT>& a6,
                                                           const Array<num_dimensions, elemT, indexT>& a7,
                                                           const Array<num_dimensions, elemT, indexT>& a8);

template <int num_dimensions, typename elemT, typename indexT>
inline Array<num_dimensions + 1, elemT, indexT> make_array(const Array<num_dimensions - 1, elemT, indexT>& a0,
                                                           const Array<num_dimensions - 1, elemT, indexT>& a1,
                                                           const Array<num_dimensions - 1, elemT, indexT>& a2,
                                                           const Array<num_dimensions - 1, elemT, indexT>& a3,
                                                           const Array<num_dimensions - 1, elemT, indexT>& a4,
                                                           Array<num_dimensions - 1, elemT, indexT>& a5,
                                                           const Array<num_dimensions - 1, elemT, indexT>& a6,
                                                           const Array<num_dimensions - 1, elemT, indexT>& a7,
                                                           const Array<num_dimensions - 1, elemT, indexT>& a8,
                                                           const Array<num_dimensions - 1, elemT, indexT>& a9);

END_NAMESPACE_STIR

#include "stir/make_array.inl"

#endif
