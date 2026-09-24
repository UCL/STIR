//
//
/*
    Copyright (C) 2005- 2005, Hammersmith Imanet Ltd
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Array

  \brief Implementation of functions for constructing arrays stir::make_1d_array etc

  \author Kris Thielemans

*/

START_NAMESPACE_STIR

template <typename T, typename indexT = int>
VectorWithOffset<T, indexT>
make_vector_with_index_type(const T& a0)
{
  VectorWithOffset<T, indexT> a(1);
  a[0] = a0;
  return a;
}

template <typename T, typename indexT = int>
VectorWithOffset<T, indexT>
make_vector_with_index_type(const T& a0, const T& a1)
{
  VectorWithOffset<T, indexT> a(2);
  a[0] = a0;
  a[1] = a1;
  return a;
}

template <typename T, typename indexT = int>
VectorWithOffset<T, indexT>
make_vector_with_index_type(const T& a0, const T& a1, const T& a2)
{
  VectorWithOffset<T, indexT> a(3);
  a[0] = a0;
  a[1] = a1;
  a[2] = a2;
  return a;
}

template <typename T, typename indexT = int>
VectorWithOffset<T, indexT>
make_vector_with_index_type(const T& a0, const T& a1, const T& a2, const T& a3)
{
  VectorWithOffset<T, indexT> a(4);
  a[0] = a0;
  a[1] = a1;
  a[2] = a2;
  a[3] = a3;
  return a;
}

template <typename T, typename indexT = int>
VectorWithOffset<T, indexT>
make_vector_with_index_type(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4)
{
  VectorWithOffset<T, indexT> a(5);
  a[0] = a0;
  a[1] = a1;
  a[2] = a2;
  a[3] = a3;
  a[4] = a4;
  return a;
}

template <typename T, typename indexT = int>
VectorWithOffset<T, indexT>
make_vector_with_index_type(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5)
{
  VectorWithOffset<T, indexT> a(6);
  a[0] = a0;
  a[1] = a1;
  a[2] = a2;
  a[3] = a3;
  a[4] = a4;
  a[5] = a5;
  return a;
}

template <typename T, typename indexT = int>
VectorWithOffset<T, indexT>
make_vector_with_index_type(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6)
{
  VectorWithOffset<T, indexT> a(7);
  a[0] = a0;
  a[1] = a1;
  a[2] = a2;
  a[3] = a3;
  a[4] = a4;
  a[5] = a5;
  a[6] = a6;
  return a;
}

template <typename T, typename indexT = int>
VectorWithOffset<T, indexT>
make_vector_with_index_type(
    const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6, const T& a7)
{
  VectorWithOffset<T, indexT> a(8);
  a[0] = a0;
  a[1] = a1;
  a[2] = a2;
  a[3] = a3;
  a[4] = a4;
  a[5] = a5;
  a[6] = a6;
  a[7] = a7;
  return a;
}

template <typename T, typename indexT = int>
VectorWithOffset<T, indexT>
make_vector_with_index_type(
    const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6, const T& a7, const T& a8)
{
  VectorWithOffset<T, indexT> a(9);
  a[0] = a0;
  a[1] = a1;
  a[2] = a2;
  a[3] = a3;
  a[4] = a4;
  a[5] = a5;
  a[6] = a6;
  a[7] = a7;
  a[8] = a8;
  return a;
}

template <typename T, typename indexT = int>
VectorWithOffset<T, indexT>
make_vector_with_index_type(const T& a0,
                            const T& a1,
                            const T& a2,
                            const T& a3,
                            const T& a4,
                            const T& a5,
                            const T& a6,
                            const T& a7,
                            const T& a8,
                            const T& a9)
{
  VectorWithOffset<T, indexT> a(10);
  a[0] = a0;
  a[1] = a1;
  a[2] = a2;
  a[3] = a3;
  a[4] = a4;
  a[5] = a5;
  a[6] = a6;
  a[7] = a7;
  a[8] = a8;
  a[9] = a9;
  return a;
}

template <typename T>
VectorWithOffset<T>
make_vector(const T& a0)
{
  return make_vector_with_index_type<T>(a0);
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1)
{
  return make_vector_with_index_type<T>(a0, a1);
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2)
{
  return make_vector_with_index_type<T>(a0, a1, a2);
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3)
{
  return make_vector_with_index_type<T>(a0, a1, a2, a3);
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4)
{
  return make_vector_with_index_type<T>(a0, a1, a2, a3, a4);
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5)
{
  return make_vector_with_index_type<T>(a0, a1, a2, a3, a4, a5);
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6)
{
  return make_vector_with_index_type<T>(a0, a1, a2, a3, a4, a5, a6);
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6, const T& a7)
{
  return make_vector_with_index_type<T>(a0, a1, a2, a3, a4, a5, a6, a7);
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6, const T& a7, const T& a8)
{
  return make_vector_with_index_type<T>(a0, a1, a2, a3, a4, a5, a6, a7, a8);
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0,
            const T& a1,
            const T& a2,
            const T& a3,
            const T& a4,
            const T& a5,
            const T& a6,
            const T& a7,
            const T& a8,
            const T& a9)
{
  return make_vector_with_index_type<T>(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
}

template <typename T, typename indexT = int>
Array<1, T, indexT>
make_1d_array_with_index_type(const T& a0)
{
  const Array<1, T, indexT> a = NumericVectorWithOffset<T, T, indexT>(make_vector_with_index_type<T, indexT>(a0));
  return a;
}

template <typename T, typename indexT = int>
Array<1, T, indexT>
make_1d_array_with_index_type(const T& a0, const T& a1)
{
  const Array<1, T, indexT> a = NumericVectorWithOffset<T, T, indexT>(make_vector_with_index_type<T, indexT>(a0, a1));
  return a;
}

template <typename T, typename indexT = int>
Array<1, T, indexT>
make_1d_array_with_index_type(const T& a0, const T& a1, const T& a2)
{
  const Array<1, T, indexT> a = NumericVectorWithOffset<T, T, indexT>(make_vector_with_index_type<T, indexT>(a0, a1, a2));
  return a;
}

template <typename T, typename indexT = int>
Array<1, T, indexT>
make_1d_array_with_index_type(const T& a0, const T& a1, const T& a2, const T& a3)
{
  const Array<1, T, indexT> a = NumericVectorWithOffset<T, T, indexT>(make_vector_with_index_type<T, indexT>(a0, a1, a2, a3));
  return a;
}

template <typename T, typename indexT = int>
Array<1, T, indexT>
make_1d_array_with_index_type(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4)
{
  const Array<1, T, indexT> a = NumericVectorWithOffset<T, T, indexT>(make_vector_with_index_type<T, indexT>(a0, a1, a2, a3, a4));
  return a;
}

template <typename T, typename indexT = int>
Array<1, T, indexT>
make_1d_array_with_index_type(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5)
{
  const Array<1, T, indexT> a
      = NumericVectorWithOffset<T, T, indexT>(make_vector_with_index_type<T, indexT>(a0, a1, a2, a3, a4, a5));
  return a;
}

template <typename T, typename indexT = int>
Array<1, T, indexT>
make_1d_array_with_index_type(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6)
{
  const Array<1, T, indexT> a
      = NumericVectorWithOffset<T, T, indexT>(make_vector_with_index_type<T, indexT>(a0, a1, a2, a3, a4, a5, a6));
  return a;
}

template <typename T, typename indexT = int>
Array<1, T, indexT>
make_1d_array_with_index_type(
    const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6, const T& a7)
{
  const Array<1, T, indexT> a
      = NumericVectorWithOffset<T, T, indexT>(make_vector_with_index_type<T, indexT>(a0, a1, a2, a3, a4, a5, a6, a7));
  return a;
}

template <typename T, typename indexT = int>
Array<1, T, indexT>
make_1d_array_with_index_type(
    const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6, const T& a7, const T& a8)
{
  const Array<1, T, indexT> a
      = NumericVectorWithOffset<T, T, indexT>(make_vector_with_index_type<T, indexT>(a0, a1, a2, a3, a4, a5, a6, a7, a8));
  return a;
}

template <typename T, typename indexT = int>
Array<1, T, indexT>
make_1d_array_with_index_type(const T& a0,
                              const T& a1,
                              const T& a2,
                              const T& a3,
                              const T& a4,
                              const T& a5,
                              const T& a6,
                              const T& a7,
                              const T& a8,
                              const T& a9)
{
  const Array<1, T, indexT> a
      = NumericVectorWithOffset<T, T, indexT>(make_vector_with_index_type<T, indexT>(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9));
  return a;
}

template <typename T>
Array<1, T>
make_1d_array(const T& a0)
{
  return make_1d_array_with_index_type<T>(a0);
}

template <class T>
Array<1, T>
make_1d_array(const T& a0, const T& a1)
{
  return make_1d_array_with_index_type<T>(a0, a1);
}

template <class T>
Array<1, T>
make_1d_array(const T& a0, const T& a1, const T& a2)
{
  return make_1d_array_with_index_type<T>(a0, a1, a2);
}

template <class T>
Array<1, T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3)
{
  return make_1d_array_with_index_type<T>(a0, a1, a2, a3);
}

template <class T>
Array<1, T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4)
{
  return make_1d_array_with_index_type<T>(a0, a1, a2, a3, a4);
}

template <class T>
Array<1, T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5)
{
  return make_1d_array_with_index_type<T>(a0, a1, a2, a3, a4, a5);
}

template <class T>
Array<1, T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6)
{
  return make_1d_array_with_index_type<T>(a0, a1, a2, a3, a4, a5, a6);
}

template <class T>
Array<1, T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6, const T& a7)
{
  return make_1d_array_with_index_type<T>(a0, a1, a2, a3, a4, a5, a6, a7);
}

template <class T>
Array<1, T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4, const T& a5, const T& a6, const T& a7, const T& a8)
{
  return make_1d_array_with_index_type<T>(a0, a1, a2, a3, a4, a5, a6, a7, a8);
}

template <class T>
Array<1, T>
make_1d_array(const T& a0,
              const T& a1,
              const T& a2,
              const T& a3,
              const T& a4,
              const T& a5,
              const T& a6,
              const T& a7,
              const T& a8,
              const T& a9)
{
  return make_1d_array_with_index_type<T>(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
}

template <int num_dimensions, typename elemT, typename indexT>
Array<num_dimensions + 1, elemT, indexT>
make_array(const Array<num_dimensions, elemT, indexT>& a0)
{
  const Array<1, elemT, indexT> a = NumericVectorWithOffset<elemT, elemT, indexT>(make_vector_with_index_type<elemT, indexT>(a0));
  return a;
}

template <int num_dimensions, typename elemT, typename indexT>
Array<num_dimensions + 1, elemT, indexT>
make_array(const Array<num_dimensions, elemT, indexT>& a0, const Array<num_dimensions, elemT, indexT>& a1)
{
  const Array<num_dimensions + 1, elemT, indexT> a = NumericVectorWithOffset<Array<num_dimensions, elemT, indexT>, elemT, indexT>(
      make_vector_with_index_type<Array<num_dimensions, elemT, indexT>, indexT>(a0, a1));
  return a;
}

template <int num_dimensions, typename elemT, typename indexT>
Array<num_dimensions + 1, elemT, indexT>
make_array(const Array<num_dimensions, elemT, indexT>& a0,
           const Array<num_dimensions, elemT, indexT>& a1,
           const Array<num_dimensions, elemT, indexT>& a2)
{
  const Array<num_dimensions + 1, elemT, indexT> a = NumericVectorWithOffset<Array<num_dimensions, elemT, indexT>, elemT, indexT>(
      make_vector_with_index_type<Array<num_dimensions, elemT, indexT>, indexT>(a0, a1, a2));
  return a;
}

template <int num_dimensions, typename elemT, typename indexT>
Array<num_dimensions + 1, elemT, indexT>
make_array(const Array<num_dimensions, elemT, indexT>& a0,
           const Array<num_dimensions, elemT, indexT>& a1,
           const Array<num_dimensions, elemT, indexT>& a2,
           const Array<num_dimensions, elemT, indexT>& a3)
{
  const Array<num_dimensions + 1, elemT, indexT> a = NumericVectorWithOffset<Array<num_dimensions, elemT, indexT>, elemT, indexT>(
      make_vector_with_index_type<Array<num_dimensions, elemT, indexT>, indexT>(a0, a1, a2, a3));
  return a;
}

template <int num_dimensions, typename elemT, typename indexT>
Array<num_dimensions + 1, elemT, indexT>
make_array(const Array<num_dimensions, elemT, indexT>& a0,
           const Array<num_dimensions, elemT, indexT>& a1,
           const Array<num_dimensions, elemT, indexT>& a2,
           const Array<num_dimensions, elemT, indexT>& a3,
           const Array<num_dimensions, elemT, indexT>& a4)
{
  const Array<num_dimensions + 1, elemT, indexT> a = NumericVectorWithOffset<Array<num_dimensions, elemT, indexT>, elemT, indexT>(
      make_vector_with_index_type<Array<num_dimensions, elemT, indexT>, indexT>(a0, a1, a2, a3, a4));
  return a;
}

template <int num_dimensions, typename elemT, typename indexT>
Array<num_dimensions + 1, elemT, indexT>
make_array(const Array<num_dimensions, elemT, indexT>& a0,
           const Array<num_dimensions, elemT, indexT>& a1,
           const Array<num_dimensions, elemT, indexT>& a2,
           const Array<num_dimensions, elemT, indexT>& a3,
           const Array<num_dimensions, elemT, indexT>& a4,
           Array<num_dimensions, elemT, indexT>& a5)
{
  const Array<num_dimensions + 1, elemT, indexT> a = NumericVectorWithOffset<Array<num_dimensions, elemT, indexT>, elemT, indexT>(
      make_vector_with_index_type<Array<num_dimensions, elemT, indexT>, indexT>(a0, a1, a2, a3, a4, a5));
  return a;
}

template <int num_dimensions, typename elemT, typename indexT>
Array<num_dimensions + 1, elemT, indexT>
make_array(const Array<num_dimensions, elemT, indexT>& a0,
           const Array<num_dimensions, elemT, indexT>& a1,
           const Array<num_dimensions, elemT, indexT>& a2,
           const Array<num_dimensions, elemT, indexT>& a3,
           const Array<num_dimensions, elemT, indexT>& a4,
           Array<num_dimensions, elemT, indexT>& a5,
           const Array<num_dimensions, elemT, indexT>& a6)
{
  const Array<num_dimensions + 1, elemT, indexT> a = NumericVectorWithOffset<Array<num_dimensions, elemT, indexT>, elemT, indexT>(
      make_vector_with_index_type<Array<num_dimensions, elemT, indexT>, indexT>(a0, a1, a2, a3, a4, a5, a6));
  return a;
}

template <int num_dimensions, typename elemT, typename indexT>
Array<num_dimensions + 1, elemT, indexT>
make_array(const Array<num_dimensions, elemT, indexT>& a0,
           const Array<num_dimensions, elemT, indexT>& a1,
           const Array<num_dimensions, elemT, indexT>& a2,
           const Array<num_dimensions, elemT, indexT>& a3,
           const Array<num_dimensions, elemT, indexT>& a4,
           Array<num_dimensions, elemT, indexT>& a5,
           const Array<num_dimensions, elemT, indexT>& a6,
           const Array<num_dimensions, elemT, indexT>& a7)
{
  const Array<num_dimensions + 1, elemT, indexT> a = NumericVectorWithOffset<Array<num_dimensions, elemT, indexT>, elemT, indexT>(
      make_vector_with_index_type<Array<num_dimensions, elemT, indexT>, indexT>(a0, a1, a2, a3, a4, a5, a6, a7));
  return a;
}

template <int num_dimensions, typename elemT, typename indexT>
Array<num_dimensions + 1, elemT, indexT>
make_array(const Array<num_dimensions, elemT, indexT>& a0,
           const Array<num_dimensions, elemT, indexT>& a1,
           const Array<num_dimensions, elemT, indexT>& a2,
           const Array<num_dimensions, elemT, indexT>& a3,
           const Array<num_dimensions, elemT, indexT>& a4,
           Array<num_dimensions, elemT, indexT>& a5,
           const Array<num_dimensions, elemT, indexT>& a6,
           const Array<num_dimensions, elemT, indexT>& a7,
           const Array<num_dimensions, elemT, indexT>& a8)
{
  const Array<num_dimensions + 1, elemT, indexT> a = NumericVectorWithOffset<Array<num_dimensions, elemT, indexT>, elemT, indexT>(
      make_vector_with_index_type<Array<num_dimensions, elemT, indexT>, indexT>(a0, a1, a2, a3, a4, a5, a6, a7, a8));
  return a;
}

template <int num_dimensions, typename elemT, typename indexT>
Array<num_dimensions + 1, elemT, indexT>
make_array(const Array<num_dimensions, elemT, indexT>& a0,
           const Array<num_dimensions, elemT, indexT>& a1,
           const Array<num_dimensions, elemT, indexT>& a2,
           const Array<num_dimensions, elemT, indexT>& a3,
           const Array<num_dimensions, elemT, indexT>& a4,
           Array<num_dimensions, elemT, indexT>& a5,
           const Array<num_dimensions, elemT, indexT>& a6,
           const Array<num_dimensions, elemT, indexT>& a7,
           const Array<num_dimensions, elemT, indexT>& a8,
           const Array<num_dimensions, elemT, indexT>& a9)
{
  const Array<num_dimensions + 1, elemT, indexT> a = NumericVectorWithOffset<Array<num_dimensions, elemT, indexT>, elemT, indexT>(
      make_vector_with_index_type<Array<num_dimensions, elemT, indexT>, indexT>(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9));
  return a;
}

END_NAMESPACE_STIR
