//
//

/*!
  \file 
  \ingroup numerics
  \brief functions to convert from data in Numerical Recipes format to STIR arrays.

  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_numerics_stir_NumericalRecipes_h__
#define __stir_numerics_stir_NumericalRecipes_h__

#include "stir/VectorWithOffset.h"
#include "stir/Array.h"
#include <complex>

START_NAMESPACE_STIR

/*!
  \ingroup numerics
  \name functions to convert from data in Numerical Recipes format to STIR arrays.
*/
//@{

inline void stir_to_nr(const VectorWithOffset< std::complex<float> >& c,
		VectorWithOffset<float>& nr_data)
{
  for (int i=0; i<c.get_length(); ++i)
  {
    nr_data[2*i+1] = c[i].real();
    nr_data[2*i+2] = c[i].imag();
  }
}

inline void stir_to_nr(const Array<2,std::complex<float> >& c2d,
		VectorWithOffset<float>& nr_data)
{
		
  VectorWithOffset<float>::iterator nr_iter = nr_data.begin();
      
  Array<2,std::complex<float> >::const_full_iterator iter=
    c2d.begin_all();

  while(iter != c2d.end_all())
    {
      *nr_iter++ = iter->real();
      *nr_iter++ = iter->imag();
      ++iter;
    }
}


inline void stir_to_nr(const VectorWithOffset<Array<1,std::complex<float> > >& c2d,
		VectorWithOffset<float>& nr_data)
{
		
  VectorWithOffset<float>::iterator nr_iter = nr_data.begin();

  VectorWithOffset<Array<1,std::complex<float> > >::const_iterator iter = 
    c2d.begin();
  while(iter != c2d.end())
    {
      Array<1,std::complex<float> >::const_iterator row_iter = iter->begin();
      while(row_iter != iter->end())
        {
          *nr_iter++ = row_iter->real();
          *nr_iter++ = row_iter->imag();
          ++row_iter;
        }
      ++iter;
    }
}

void nr_to_stir(const VectorWithOffset<float>& nr_data,
		VectorWithOffset< std::complex<float> >& c)
{
  for (int i=0; i<c.get_length(); ++i)
  {
    c[i]=std::complex<float>(nr_data[2*i+1], nr_data[2*i+2]);
  }
}

inline void nr_to_stir(const VectorWithOffset<float>& nr_data,
		Array<2,std::complex<float> > & c2d)
{
  VectorWithOffset<float>::const_iterator nr_iter = nr_data.begin();
  Array<2,std::complex<float> >::full_iterator iter=
    c2d.begin_all();
  while(iter != c2d.end_all())
    {
      *iter = std::complex<float>(*nr_iter, *(nr_iter+1));
      nr_iter+= 2;
      ++iter;
    }     
}

inline void nr_to_stir(const VectorWithOffset<float>& nr_data,
		VectorWithOffset<Array<1,std::complex<float> > >& c2d)
{
  VectorWithOffset<float>::const_iterator nr_iter = nr_data.begin();
  VectorWithOffset<Array<1,std::complex<float> > >::iterator iter = 
    c2d.begin();
  while(iter != c2d.end())
    {
      Array<1,std::complex<float> >::iterator row_iter = iter->begin();
      while(row_iter != iter->end())
        {
          *row_iter = std::complex<float>(*nr_iter, *(nr_iter+1));
          nr_iter+= 2;
          ++row_iter;
        }
      ++iter;
    }
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

//@}

END_NAMESPACE_STIR
#endif
