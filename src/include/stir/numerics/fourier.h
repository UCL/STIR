//
//

/*!
  \file 
  \ingroup DFT
  \brief Functions for computing FFTs

  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_numerics_stir_fourier_h__
#define  __stir_numerics_stir_fourier_h__
#include "stir/VectorWithOffset.h"
#include "stir/Array_complex_numbers.h"
START_NAMESPACE_STIR



/*! \ingroup DFT
  \brief Compute multi-dimensional discrete fourier transform.

  \param[in,out] c The type of \a c should normally be <code>Array\<n,std::complex\<T\> \></code>.
  The function will then compute the \a n- dimensional fourier transform of the data,
  and store the result in \a c.

  \param[in] sign This can be used to implement a different convention for the DFT.

  \see fourier_1d for conventions and restrictions

  \warning Currently, the array has to have \c get_min_index()==0 at each dimension.
*/
template <typename T>
void 
fourier(T& c, const int sign = 1);
//fourier(VectorWithOffset<elemT>& c, const int sign = 1);


/*! \ingroup DFT
  \brief Compute the inverse of the multi-dimensional discrete fourier transform.

  The scale factor is such that <tt>inverse_fourier(fourier(c,sign),sign)==c</tt>, aside
  from numerical error of course.
  \see fourier
*/
template <typename T>
inline void inverse_fourier(T& c, const int sign=1)
{
  fourier(c,-sign);
#ifdef _MSC_VER
  // disable warning about conversion
  #pragma warning(disable:4244)
#endif
  c /= c.size_all();
#ifdef _MSC_VER
  // disable warning about conversion
  #pragma warning(default:4244)
#endif
}

/*! \ingroup DFT

  \brief Compute one-dimensional discrete fourier transform of an array.

  \param[in,out] c The type of \a c should normally be <code>Array\<n,std::complex\<T\> \></code> (but see below).
  The function will then compute the one-dimensional fourier transform (i.e. on the
  'outer' index) of the data, and store the result in \a c.

  \param[in] sign This can be used to implement a different convention for the DFT

  \warning Currently, the array has to be indexed from 0.
  \warning Currently, the length of the array has to be a power of 2.
   
  The convention used is as follows.
  For a vector of length \a n, the result is
  \f[
    r_s = \sum_{s=0}^{n-1} c_r e^{\mathrm{sign} 2\pi i r s/n}
  \f]
  This means that the zero-frequency will be returned in <tt>c[0]</tt>
   
  This function can be used with more general type of \a c (if instantiated in fourier.cxx).
  The type \a T has to be such that \a T::value_type, \a T::reference and 
  <tt> T::reference T::operator[](const int)</tt> exist. Moreover, numerical
  operations
  <tt>operator*=(T::reference, std::complex\<float\>)</tt>,
  <tt>operator+=(T::reference, T::value_type)</tt> and
  <tt>operator*=(T::reference, int)</tt>,
   have to be defined as well.
*/
template <typename T>
void fourier_1d(T& c, const int sign);

/*! \ingroup DFT
  \brief Compute the inverse of the one-dimensional discrete fourier transform.

  The scale factor is such that <tt>inverse_fourier_1d(fourier_1d(c,sign),sign)==c</tt>, aside
  from numerical error of course.
  \see fourier_1d()
*/
template <typename T>
inline void inverse_fourier_1d(T& c, const int sign=1)
{
  fourier_1d(c,-sign);
#ifdef _MSC_VER
  // disable warning about conversion
  #pragma warning(disable:4244)
#endif
  c /= c.size();
#ifdef _MSC_VER
  #pragma warning(default:4244)
#endif
}

/*! \ingroup DFT

  \brief Compute one-dimensional discrete fourier transform of a real array (of even size).

  \param[in] c The type of \a c should normally be <code>Array\<1,float\></code> 
          (or \c double).
  \param[in] sign  This can be used to implement a different convention for the DFT.
  \return The positive frequencies of the DFT as an array of complex numbers. That is,
  for <tt>c.size()==2*n</tt>, the returned array will have indices from <tt>0</tt> 
  to <tt>n</tt>, with the 0 frequency at 0.

  For a real array, the DFT is such that the values at negative frequencies are the
  complex conjugate of the value at the corresponding positive frequency.
  In addition, if the length of the real array is even, the DFT can be computed
  efficiently by creating a complex array of half the length (by putting the
  even-numbered elements as the real part, and the odd-numbered as the imaginary part).

  This function implements the above and is hence (probably) a faster way to compute
  DFTs of real arrays. Note however, that in contrast to the Numerical Recipes routines,
  the result is not stored in the same memory location as the input. For a recent
  compiler that implements the Named-Return-Value-Optimisation, this should not be a problem
  in most cases.

  Result is such that  
  <tt>pos_frequencies_to_all(fourier_1d_for_real_data(v,sign))==fourier(v,sign)</tt> 
  (this is meant symbolically. In code you'd have to make \c v into an array of complex
  numbers before passing to \c fourier).
    
  \see pos_frequencies_to_all()
  \see fourier_1d() for conventions and restrictions

  \warning Currently, the array has to have \c get_min_index()==0.
*/
template <typename T>
Array<1,std::complex<T> >
fourier_1d_for_real_data(const Array<1,T>& c, const int sign = 1);


/*! \ingroup DFT

  \brief Compute the inverse of the one-dimensional discrete fourier transform of a real array (of even size).
  \warning Because of implementation issues, a temporary copy of the input data \c c has to be made. This
  obviously affects performance.
  \see fourier_1d_for_real_data()
*/
template <typename T>
Array<1,T>
inverse_fourier_1d_for_real_data(const Array<1,std::complex<T> >& c, const int sign = 1);

/*! \ingroup DFT

  \brief As inverse_fourier_1d_for_real_data(), but avoiding the copy of the input array.

  \warning destroys values in (and resizes) first argument \a c
  \see inverse_fourier_1d_for_real_data()
*/
template <typename T>
Array<1,T>
  inverse_fourier_1d_for_real_data_corrupting_input(Array<1,std::complex<T> >& c, const int sign);

/*! \ingroup DFT

  \brief Compute discrete fourier transform of a real array (with the last dimensions of even size).

  \param[in] c The type of \a c should normally be <code>Array\<d,float\></code> 
          (or \c double).
  \param[in] sign  This can be used to implement a different convention for the DFT.
  \return The positive frequencies of the DFT as an array of complex numbers. That is,
  if \c c has sizes <tt>(n1,n2,...,nd)</tt>, with <tt>nd</tt> even,
  the returned array will have sizes
  <tt>(n1,n2,...,(nd/2)+1)</tt>, with the 0 frequency at index <tt>(0,0,...0)</tt>.

  For a real array, the DFT is such that the values at frequency
  <tt>(k1,k2,...,kd)</tt> are the complex conjugate of the value at the corresponding 
  frequency <tt>(-k1,-k2,...,-kd)</tt>. 
  \see fourier_1d_for_real_data()

  This can be used to compute only the 'positive' half of the frequencies. For this 
  implementation, this means that only results for frequencies <tt>(k1,k2,...,kd)</tt>
  with <code>0\<=kd\<=(nd/2)</code>, i.e. the 'last' dimension has positive frequencies.

  \warning At present, \c c has to be a regular array with all indices starting from 0.
  \see pos_frequencies_to_all()
*/
template <int num_dimensions, typename T>
  Array<num_dimensions,std::complex<T> >
  fourier_for_real_data(const Array<num_dimensions,T>& c, const int sign = 1);

/*! \ingroup DFT

  \brief Compute the inverse of the discrete fourier transform of a real array (with the last dimension of even size).

  \warning Because of implementation issues, a temporary copy of the input data \c c has to be made. This
  \see fourier_for_real_data()
*/
template <int num_dimensions, typename T>
  Array<num_dimensions,T >
  inverse_fourier_for_real_data(const Array<num_dimensions,std::complex<T> >& c, const int sign = 1);


/*! \ingroup DFT

  \brief As inverse_fourier_for_real_data(), but avoiding the copy of the input array.

  \warning destroys values in (and resizes) first argument \a c
  \see inverse_fourier_for_real_data()
*/
template <int num_dimensions, typename T>
Array<num_dimensions,T >
  inverse_fourier_for_real_data_corrupting_input(Array<num_dimensions,std::complex<T> >& c, const int sign=1);

/*! \ingroup DFT
  \brief Adds negative frequencies to the last dimension of a complex array by complex conjugation.

  \see fourier_for_real_data()
*/
template <int num_dimensions, typename T>
Array<num_dimensions, std::complex<T> > 
pos_frequencies_to_all(const Array<num_dimensions, std::complex<T> >& c);


END_NAMESPACE_STIR

#endif
