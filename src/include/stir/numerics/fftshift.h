#ifndef __stir_numerics_fftshift_H__
#define __stir_numerics_fftshift_H__
/*
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
 
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup numerics
  \brief Functions to rearrange Fourier data so that DC (zero frequency) is centered.

  \author Dimitra Kyriakopoulou
  
  \details
  These are the familiar "fftshift" operations used around FFTs.
  1D: swap the lower and upper halves of a vector.
  2D: swap left/right halves, then top/bottom halves (quadrant swap).

  \note Assumes an even size (as in our FFT lengths).
*/

#include "stir/Array.h"
#include <complex>

START_NAMESPACE_STIR

//! \brief In-place 1D fftshift: swap halves [0..N/2-1] <-> [N/2..N-1]
template <typename T>
inline void fftshift(Array<1, T>& a, int size)
{
  T temp = 0;
  for (int i = 0; i < size/2; ++i) {
    temp = a[i];
    a[i] = a[size/2 + i];
    a[size/2 + i] = temp;
  }
}

//! \brief In-place 2D fftshift: quadrant swap (left-right, then top-bottom). Accepts complex arrays.
template <typename T>
inline void fftshift(Array<2, std::complex<T> >& a, int size)
{
  std::complex<T> temp;
  // swap left/right halves across all rows
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size/2; ++j) {
      temp = a[i][j];
      a[i][j] = a[i][size/2 + j];
      a[i][size/2 + j] = temp;
    }
  }
  // swap top/bottom halves across all columns
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size/2; ++j) {
      temp = a[j][i];
      a[j][i] = a[size/2 + j][i];
      a[size/2 + j][i] = temp;
    }
  }
}

END_NAMESPACE_STIR
#endif
