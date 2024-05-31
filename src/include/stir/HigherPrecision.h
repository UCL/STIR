//
//
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock
  \brief class HigherPrecision

  \author Kris Thielemans
*/
#ifndef __stir_HigherPrecision__H__
#  define _stir_HigherPrecision__H__

#  include "stir/common.h"
#  include <complex>

START_NAMESPACE_STIR

//! Helper class to get a type with higher precision
/*! Specialisations convert float to double, and double to long double
 */
template <class T>
struct HigherPrecision
{
  typedef T type;
};

template <>
struct HigherPrecision<float>
{
  typedef double type;
};

template <>
struct HigherPrecision<double>
{
  typedef long double type;
};

template <class T>
struct HigherPrecision<std::complex<T>>
{
  typedef std::complex<typename HigherPrecision<T>::type> type;
};

END_NAMESPACE_STIR

#endif
