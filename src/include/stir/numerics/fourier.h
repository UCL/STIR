//
// $Id$
//

/*!
  \file 
  \ingroup DFT
  \brief Functions for computing FFTs

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_numerics_stir_fourier_h__
#define  __stir_numerics_stir_fourier_h__
#include "stir/VectorWithOffset.h"
#include "stir/Array_complex_numbers.h"
START_NAMESPACE_STIR



/*! \ingroup DFT
  \brief Compute multi-dimensional discrete fourier transform.

  \param[in,out] c The type of \a c should normally be \verbatim Array<n,std::complex<T> > \endverbatim.
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

  The scale factor is such that <tt>inverse(fourier(c,sign),sign)==c</tt>, aside
  from numerical error of course.
*/
template <typename T>
inline void inverse_fourier(T& c, const int sign=1)
{
  fourier(c,-sign);
  c /= c.size_all();
}

/*! \ingroup DFT

  \brief Compute one-dimensional discrete fourier transform of an array.

  \param[in,out] c The type of \a c should normally be \verbatim Array<n,std::complex<T> > \endverbatim (but see below).
  The function will then compute the one-dimensional fourier transform (i.e. on the
  'outer' index) of the data, and store the result in \a c.

  \param sign This can be used to implement a different convention for the DFT

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
  <tt>operator*=(T::reference, std::complex&lt;float&gt;)</tt>,
  <tt>operator+=(T::reference, T::value_type)</tt> and
  <tt>operator*=(T::reference, int)</tt>,
   have to be defined as well.
*/
template <typename T>
void fourier_1d(T& c, const int sign);

/*! \ingroup DFT

  \brief Compute one-dimensional discrete fourier transform of a real array (of even size).

  \param[in] c The type of \a c should normally be \verbatim Array<1,float > \endverbatim 
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
  compiler implement the Named-Return-Value-Optimisation, this should not be a problem
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
Array<1,std::complex<typename T::value_type> >
fourier_1d_for_real_data(const T& v, const int sign = 1);

/*! \ingroup DFT

  \brief Compute the inverse of the one-dimensional discrete fourier transform of a real array (of even size).

  \warning destroys values in (and resizes) first argument \a c
  \see fourier_1d_for_real_data()
*/
template <typename T>
Array<1,T>
  inverse_fourier_1d_for_real_data(Array<1,std::complex<T> >& c, const int sign = 1);

/*! \ingroup DFT
  \brief Adds negative frequencies to a complex array of positive frequences by complex conjugation.

  \see fourier_1d_for_real_data()
*/
template <typename T>
Array<1,std::complex<T> > 
pos_frequencies_to_all(const VectorWithOffset<std::complex<T> >& c)
{
  const unsigned int n = (c.get_length()-1)*2;
  Array<1,std::complex<T> > result(n);
  for (int i=1; i<c.get_length()-1; ++i)
    {
      result[i] = c[i];
      result[n-i] = std::conj(c[i]);
    }
  result[0] = c[0];
  result[n/2] = c[n/2];
  return result;
}

#if 0
void real_to_complex(const VectorWithOffset< std::complex<float> >& c,
		VectorWithOffset<float>& nr_data)
{
      Array<2,std::complex<float> >::const_full_iterator iter=
        c2d.begin_all();

      while(iter != c2d.end_all())
      {
        *nr_iter++ = iter->real();
        *nr_iter++ = iter->imag();
        ++iter;
      }
  for (int i=0; i<c.get_length(); ++i)
  {
    nr_data[2*i+1] = c[i].real();
    nr_data[2*i+2] = c[i].imag();
  }
}
#endif


END_NAMESPACE_STIR

#endif
