//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
/*!
  \file
  \ingroup Array
  
  \brief Implementation of functions for constructing arrays stir::make_1d_array etc
    
  \author Kris Thielemans

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

template <class T>
VectorWithOffset<T>
make_vector(const T& a0)
{
  VectorWithOffset<T> a(1);
  a[0]=a0;
  return a;
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1)
{
  VectorWithOffset<T> a(2);
  a[0]=a0; a[1]=a1;
  return a;
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2)
{
  VectorWithOffset<T> a(3);
  a[0]=a0; a[1]=a1; a[2]=a2;
  return a;
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3)
{
  VectorWithOffset<T> a(4);
  a[0]=a0; a[1]=a1; a[2]=a2; a[3]=a3;
  return a;
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4)
{
  VectorWithOffset<T> a(5);
  a[0]=a0; a[1]=a1; a[2]=a2; a[3]=a3; a[4]=a4;
  return a;
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5)
{
  VectorWithOffset<T> a(6);
  a[0]=a0; a[1]=a1; a[2]=a2; a[3]=a3; a[4]=a4; a[5]=a5;
  return a;
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6)
{
  VectorWithOffset<T> a(7);
  a[0]=a0; a[1]=a1; a[2]=a2; a[3]=a3; a[4]=a4; a[5]=a5; a[6]=a6;
  return a;
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6, const T& a7)
{
  VectorWithOffset<T> a(8);
  a[0]=a0; a[1]=a1; a[2]=a2; a[3]=a3; a[4]=a4; a[5]=a5; a[6]=a6; a[7]=a7;
  return a;
}


template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6, const T& a7, const T& a8)
{
  VectorWithOffset<T> a(9);
  a[0]=a0; a[1]=a1; a[2]=a2; a[3]=a3; a[4]=a4; 
  a[5]=a5; a[6]=a6; a[7]=a7; a[8]=a8;
  return a;
}

template <class T>
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6, const T& a7, const T& a8, const T& a9)
{
  VectorWithOffset<T> a(10);
  a[0]=a0; a[1]=a1; a[2]=a2; a[3]=a3; a[4]=a4; 
  a[5]=a5; a[6]=a6; a[7]=a7; a[8]=a8; a[9]=a9;
  return a;
}


template <class T>
Array<1,T>
make_1d_array(const T& a0)
{
  const Array<1,T> a = NumericVectorWithOffset<T,T>(make_vector(a0));
  return a;
}

template <class T>
Array<1,T>
make_1d_array(const T& a0, const T& a1)
{
  const Array<1,T> a = NumericVectorWithOffset<T,T>(make_vector(a0, a1));
  return a;
}


template <class T>
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2)
{
  const Array<1,T> a = NumericVectorWithOffset<T,T>(make_vector(a0, a1, a2));
  return a;
}


template <class T>
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3)
{
  const Array<1,T> a = NumericVectorWithOffset<T,T>(make_vector(a0, a1, a2, a3));
  return a;
}

template <class T>
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4)
{
  const Array<1,T> a = 
    NumericVectorWithOffset<T,T>(make_vector(a0, a1, a2, a3, a4));
  return a;
}


template <class T>
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5)
{
  const Array<1,T> a = 
    NumericVectorWithOffset<T,T> (make_vector(a0, a1, a2, a3, a4, a5));
  return a;
}


template <class T>
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6)
{
  const Array<1,T> a = 
    NumericVectorWithOffset<T,T>(make_vector(a0, a1, a2, a3, a4,
					     a5, a6));
  return a;
}


template <class T>
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6, const T& a7)
{
  const Array<1,T> a = 
    NumericVectorWithOffset<T,T>(make_vector(a0, a1, a2, a3, a4,
					     a5, a6, a7));
  return a;
}



template <class T>
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6, const T& a7, const T& a8)
{
  const Array<1,T> a = 
    NumericVectorWithOffset<T,T>(make_vector(a0, a1, a2, a3, a4,
					     a5, a6, a7, a8));
  return a;
}


template <class T>
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6, const T& a7, const T& a8, const T& a9)
{
  const Array<1,T> a = 
    NumericVectorWithOffset<T,T>(make_vector(a0, a1, a2, a3, a4,
					     a5, a6, a7, a8, a9));
  return a;
}


template <int num_dimensions, class T>
Array<num_dimensions,T>
make_array(const Array<num_dimensions,T>& a0)
{
  const Array<1,T> a = NumericVectorWithOffset<T,T>(make_vector(a0));
  return a;
}

template <int num_dimensions, class T>
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1)
{
  const Array<num_dimensions+1,T> a = NumericVectorWithOffset<Array<num_dimensions,T>,T>(make_vector(a0, a1));
  return a;
}


template <int num_dimensions, class T>
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2)
{
  const Array<num_dimensions+1,T> a = NumericVectorWithOffset<Array<num_dimensions,T>,T>(make_vector(a0, a1, a2));
  return a;
}


template <int num_dimensions, class T>
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3)
{
  const Array<num_dimensions+1,T> a = NumericVectorWithOffset<Array<num_dimensions,T>,T>(make_vector(a0, a1, a2, a3));
  return a;
}

template <int num_dimensions, class T>
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3, const Array<num_dimensions,T>& a4)
{
  const Array<num_dimensions+1,T> a = 
    NumericVectorWithOffset<Array<num_dimensions,T>,T>(make_vector(a0, a1, a2, a3, a4));
  return a;
}


template <int num_dimensions, class T>
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3, const Array<num_dimensions,T>& a4,
	    Array<num_dimensions,T>& a5)
{
  const Array<num_dimensions+1,T> a = 
    NumericVectorWithOffset<Array<num_dimensions,T>,T> (make_vector(a0, a1, a2, a3, a4, a5));
  return a;
}


template <int num_dimensions, class T>
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3, const Array<num_dimensions,T>& a4,
	    Array<num_dimensions,T>& a5, const Array<num_dimensions,T>& a6)
{
  const Array<num_dimensions+1,T> a = 
    NumericVectorWithOffset<Array<num_dimensions,T>,T>(make_vector(a0, a1, a2, a3, a4,
					     a5, a6));
  return a;
}


template <int num_dimensions, class T>
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3, const Array<num_dimensions,T>& a4,
	    Array<num_dimensions,T>& a5, const Array<num_dimensions,T>& a6, const Array<num_dimensions,T>& a7)
{
  const Array<num_dimensions+1,T> a = 
    NumericVectorWithOffset<Array<num_dimensions,T>,T>(make_vector(a0, a1, a2, a3, a4,
					     a5, a6, a7));
  return a;
}



template <int num_dimensions, class T>
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3, const Array<num_dimensions,T>& a4,
	    Array<num_dimensions,T>& a5, const Array<num_dimensions,T>& a6, const Array<num_dimensions,T>& a7, const Array<num_dimensions,T>& a8)
{
  const Array<num_dimensions+1,T> a = 
    NumericVectorWithOffset<Array<num_dimensions,T>,T>(make_vector(a0, a1, a2, a3, a4,
					     a5, a6, a7, a8));
  return a;
}


template <int num_dimensions, class T>
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3, const Array<num_dimensions,T>& a4,
	    Array<num_dimensions,T>& a5, const Array<num_dimensions,T>& a6, const Array<num_dimensions,T>& a7, const Array<num_dimensions,T>& a8, const Array<num_dimensions,T>& a9)
{
  const Array<num_dimensions+1,T> a = 
    NumericVectorWithOffset<Array<num_dimensions,T>,T>(make_vector(a0, a1, a2, a3, a4,
					     a5, a6, a7, a8, a9));
  return a;
}


END_NAMESPACE_STIR
