/*
  Copyright (C) 2004 - 2008, Hammersmith Imanet Ltd
  This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Array_IO
  \brief Implementation of stir::write_data() functions for writing stir::Array's to file

  \author Kris Thielemans

*/
#include "stir/Array.h"
#include "stir/convert_array.h"
#include "stir/NumericType.h"
#include "stir/NumericInfo.h"
#include "stir/ByteOrder.h"
#include "stir/Succeeded.h"
#include "stir/detail/test_if_1d.h"
#include "stir/IO/write_data_1d.h"
#include "stir/warning.h"
#include <typeinfo>

START_NAMESPACE_STIR

namespace detail
{
/* Generic implementation of write_data_with_fixed_scale_factor().
   See test_if_1d.h for info why we do this this way.
*/
template <int num_dimensions, class OStreamT, class elemT, class OutputType, class ScaleT>
inline Succeeded
write_data_with_fixed_scale_factor_help(is_not_1d,
                                        OStreamT& s,
                                        const ArrayType<num_dimensions, elemT>& data,
                                        NumericInfo<OutputType> output_type,
                                        const ScaleT scale_factor,
                                        const ByteOrder byte_order,
                                        const bool can_corrupt_data)
{
  for (auto iter = data.begin(); iter != data.end(); ++iter)
    {
      if (write_data_with_fixed_scale_factor(s, *iter, output_type, scale_factor, byte_order, can_corrupt_data) == Succeeded::no)
        return Succeeded::no;
    }
  return Succeeded::yes;
}

// specialisation for 1D case
template <class OStreamT, class elemT, class OutputType, class ScaleT>
inline Succeeded
write_data_with_fixed_scale_factor_help(is_1d,
                                        OStreamT& s,
                                        const ArrayType<1, elemT>& data,
                                        NumericInfo<OutputType>,
                                        const ScaleT scale_factor,
                                        const ByteOrder byte_order,
                                        const bool can_corrupt_data)
{
  if (typeid(OutputType) != typeid(elemT) || scale_factor != 1)
    {
      ScaleT new_scale_factor = scale_factor;
      auto data_tmp = convert_array(new_scale_factor, data, NumericInfo<OutputType>());
      if (std::fabs(new_scale_factor - scale_factor) > scale_factor * .001)
        return Succeeded::no;
      return write_data_1d(s, data_tmp, byte_order, /*can_corrupt_data*/ true);
    }
  else
    {
      return write_data_1d(s, data, byte_order, can_corrupt_data);
    }
}

} // end of namespace detail

template <int num_dimensions, class OStreamT, class elemT, class OutputType, class ScaleT>
Succeeded
write_data_with_fixed_scale_factor(OStreamT& s,
                                   const ArrayType<num_dimensions, elemT>& data,
                                   NumericInfo<OutputType> output_type,
                                   const ScaleT scale_factor,
                                   const ByteOrder byte_order,
                                   const bool can_corrupt_data)
{
  return detail::write_data_with_fixed_scale_factor_help(
      detail::test_if_1d<num_dimensions>(), s, data, output_type, scale_factor, byte_order, can_corrupt_data);
}

template <int num_dimensions, class OStreamT, class elemT, class OutputType, class ScaleT>
Succeeded
write_data(OStreamT& s,
           const ArrayType<num_dimensions, elemT>& data,
           NumericInfo<OutputType> output_type,
           ScaleT& scale_factor,
           const ByteOrder byte_order,
           const bool can_corrupt_data)
{
  find_scale_factor(scale_factor, data, NumericInfo<OutputType>());
  return write_data_with_fixed_scale_factor(s, data, output_type, scale_factor, byte_order, can_corrupt_data);
}

template <int num_dimensions, class OStreamT, class elemT>
inline Succeeded
write_data(OStreamT& s, const ArrayType<num_dimensions, elemT>& data, const ByteOrder byte_order, const bool can_corrupt_data)
{
  return write_data_with_fixed_scale_factor(s, data, NumericInfo<elemT>(), 1.F, byte_order, can_corrupt_data);
}

template <int num_dimensions, class OStreamT, class elemT, class ScaleT>
Succeeded
write_data(OStreamT& s,
           const ArrayType<num_dimensions, elemT>& data,
           NumericType type,
           ScaleT& scale,
           const ByteOrder byte_order,
           const bool can_corrupt_data)
{
  if (NumericInfo<elemT>().type_id() == type)
    {
      // you might want to use the scale even in this case,
      // but at the moment we don't
      scale = ScaleT(1);
      return write_data(s, data, byte_order, can_corrupt_data);
    }
  switch (type.id)
    {
      // define macro what to do with a specific NumericType
#define CASE(NUMERICTYPE)                                                                                                        \
  case NUMERICTYPE:                                                                                                              \
    return write_data(s, data, NumericInfo<typename TypeForNumericType<NUMERICTYPE>::type>(), scale, byte_order, can_corrupt_data)

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
      warning("write_data : type not yet supported\n, at line %d in file %s", __LINE__, __FILE__);
      return Succeeded::no;
    }
}

END_NAMESPACE_STIR
