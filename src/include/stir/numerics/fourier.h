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

  \param[in] sign This can be used to implement a different convention for the DFT

  \see fourier_1d for conventions

  \warning Currently, the array has to have \c get_min_index()==0 at each dimension.
*/
template <typename T>
void 
fourier(T& c, const int sign = 1);
//fourier(VectorWithOffset<elemT>& c, const int sign = 1);


template <typename T>
inline void inverse_fourier(T& c, const int sign=1)
{
  fourier(c,-sign);
  c /= c.size_all();
}

/*! \ingroup DFT

  \brief Compute one-dimensional discrete fourier transform of an array.

  \param[in,out] c The type of \a c should normally be \verbatim Array&lt;n,std::complex<T> > \endverbatim (but see below).
  The function will then compute the one-dimensional fourier transform (i.e. on the
  'outer' index) of the data, and store the result in \a c.

  \param sign This can be used to implement a different convention for the DFT

  \warning Currently, the array has to be indexed from 0.
   
  The convention used is as follows.
  For a vector of length \a n, the result is
  \f[
    r_s = \sum_{s=0}^{n-1} c_r e^{\mathrm{sign} 2\pi i r s/n}
  \f]
   
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

template <typename T>
Array<1,std::complex<typename T::value_type> >
fourier_1d_for_real_data(const T& v, const int sign = 1);

/*! \warning destroys values in (and resizes) first argument \a c */
template <typename T>
Array<1,T>
  inverse_fourier_1d_for_real_data(Array<1,std::complex<T> >& c, const int sign = 1);

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
