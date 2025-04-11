/*
    Copyright (C) 2004- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Array_IO
  \brief Implementation of stir::read_data() functions for reading stir::Array's from file

  \author Kris Thielemans

*/
#include "stir/Array.h"
#include "stir/TensorWrapper.h"
#include "stir/convert_array.h"
#include "stir/NumericType.h"
#include "stir/NumericInfo.h"
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include "stir/detail/test_if_1d.h"
#include "stir/IO/read_data_1d.h"
#include "stir/warning.h"
#include <typeinfo>

START_NAMESPACE_STIR

namespace detail
{
/* Generic implementation of read_data(). See test_if_1d.h for info why we do this.*/
template <int num_dimensions, class IStreamT, class elemT>
inline Succeeded
read_data_help(is_not_1d, IStreamT& s, Array<num_dimensions, elemT>& data, const ByteOrder byte_order)
{
  if (data.is_contiguous())
    return read_data_1d(s, data, byte_order);

  // otherwise, recurse
  for (typename Array<num_dimensions, elemT>::iterator iter = data.begin(); iter != data.end(); ++iter)
    {
      if (read_data(s, *iter, byte_order) == Succeeded::no)
        return Succeeded::no;
    }
  return Succeeded::yes;
}

/* 1D implementation of read_data(). See test_if_1d.h for info why we do this.*/
// specialisation for 1D case
template <class IStreamT, class elemT>
inline Succeeded
read_data_help(is_1d, IStreamT& s, Array<1, elemT>& data, const ByteOrder byte_order)
{
  return read_data_1d(s, data, byte_order);
}

} // end of namespace detail

template <int num_dimensions, class IStreamT, class elemT>
inline Succeeded
read_data(IStreamT& s, Array<num_dimensions, elemT>& data, const ByteOrder byte_order)
{
  return detail::read_data_help(detail::test_if_1d<num_dimensions>(), s, data, byte_order);
}

#ifdef STIR_WITH_TORCH
template <int num_dimensions, class IStreamT, class elemT>
inline Succeeded
read_data(IStreamT& s, TensorWrapper<num_dimensions, elemT>& data, const ByteOrder byte_order)
{
  if (!data.is_contiguous()) {
      throw std::runtime_error("Tensor must be contiguous to read data.");
    }

  // Ensure the tensor has the correct data type
  if (data.getTensor().scalar_type() != torch::CppTypeToScalarType<elemT>()) {
      throw std::runtime_error("Tensor data type does not match the specified element type.");
    }

  // Get the total number of elements in the tensor
  int64_t num_elements = data.size_all();

  // Calculate the total number of bytes to read
  std::size_t num_bytes = num_elements * sizeof(elemT);

  // Read binary data into the tensor's memory
  s.read(reinterpret_cast<char*>(data.begin_all()), num_bytes);

  if (!s) {
      return Succeeded::no;
    }

  // if (byte_order != ByteOrder::native) {
  //     elemT* data_ptr = data.begin_all();
  //     for (int64_t i = 0; i < num_elements; ++i) {
  //         swap_byte_order(data_ptr[i]);
  //       }
  //   }

  return Succeeded::yes;
}

template <int num_dimensions, class IStreamT, class elemT, class InputType, class ScaleT>
inline Succeeded
read_data(IStreamT& s,
          TensorWrapper<num_dimensions, elemT>& data,
          NumericInfo<InputType> input_type,
          ScaleT& scale_factor,
          const ByteOrder byte_order)
{
  if (typeid(InputType) == typeid(elemT))
    {
      // TODO? you might want to use the scale even in this case,
      // but at the moment we don't
      scale_factor = ScaleT(1);
      return read_data(s, data, byte_order);
    }
  else
    {
      //TODO NE Another time.
      return Succeeded::no;
      // Array<num_dimensions, InputType> in_data(data.get_index_range());
      // Succeeded success = read_data(s, in_data, byte_order);
      // if (success == Succeeded::no)
      //   return Succeeded::no;
      // convert_array(data, scale_factor, in_data);
      // return Succeeded::yes;
    }
}

template <int num_dimensions, class IStreamT, class elemT, class ScaleT>
inline Succeeded
read_data(IStreamT& s, TensorWrapper<num_dimensions, elemT>& data, NumericType type, ScaleT& scale, const ByteOrder byte_order)
{
  switch (type.id)
  {
    // define macro what to do with a specific NumericType
#define CASE(NUMERICTYPE)                                                                                                        \
  case NUMERICTYPE:                                                                                                              \
        return read_data(s, data, NumericInfo<typename TypeForNumericType<NUMERICTYPE>::type>(), scale, byte_order)

               // now list cases that we want
        CASE(NumericType::SCHAR);
    CASE(NumericType::UCHAR);
    CASE(NumericType::SHORT);
    CASE(NumericType::USHORT);
    CASE(NumericType::INT);
    CASE(NumericType::UINT);
    CASE(NumericType::LONG);
    CASE(NumericType::ULONG);
    CASE(NumericType::FLOAT);
    CASE(NumericType::DOUBLE);
#undef CASE
  default:
    warning("read_data : type not yet supported\n, at line %d in file %s", __LINE__, __FILE__);
    return Succeeded::no;
  }
}
#endif

template <int num_dimensions, class IStreamT, class elemT, class InputType, class ScaleT>
inline Succeeded
read_data(IStreamT& s,
          Array<num_dimensions, elemT>& data,
          NumericInfo<InputType> input_type,
          ScaleT& scale_factor,
          const ByteOrder byte_order)
{
  if (typeid(InputType) == typeid(elemT))
    {
      // TODO? you might want to use the scale even in this case,
      // but at the moment we don't
      scale_factor = ScaleT(1);
      return read_data(s, data, byte_order);
    }
  else
    {
      Array<num_dimensions, InputType> in_data(data.get_index_range());
      Succeeded success = read_data(s, in_data, byte_order);
      if (success == Succeeded::no)
        return Succeeded::no;
      convert_array(data, scale_factor, in_data);
      return Succeeded::yes;
    }
}

template <int num_dimensions, class IStreamT, class elemT, class ScaleT>
inline Succeeded
read_data(IStreamT& s, Array<num_dimensions, elemT>& data, NumericType type, ScaleT& scale, const ByteOrder byte_order)
{
  switch (type.id)
    {
      // define macro what to do with a specific NumericType
#define CASE(NUMERICTYPE)                                                                                                        \
  case NUMERICTYPE:                                                                                                              \
    return read_data(s, data, NumericInfo<typename TypeForNumericType<NUMERICTYPE>::type>(), scale, byte_order)

      // now list cases that we want
      CASE(NumericType::INT);
      CASE(NumericType::FLOAT);
      CASE(NumericType::DOUBLE);
#undef CASE
    default:
      warning("read_data : type not yet supported\n, at line %d in file %s", __LINE__, __FILE__);
      return Succeeded::no;
    }
}

#ifdef STIR_WITH_TORCH
#endif

END_NAMESPACE_STIR
