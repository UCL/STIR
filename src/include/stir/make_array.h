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
#ifndef __stir_make_array_H__
#define __stir_make_array_H__
/*!
  \file
  \ingroup Array
  
  \brief Declaration of functions for constructing arrays stir::make_1d_array etc
    
  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/Array.h"

START_NAMESPACE_STIR


template <class T>
inline
VectorWithOffset<T>
make_vector(const T& a0);

template <class T>
inline
VectorWithOffset<T>
make_vector(const T& a0, const T& a1);

template <class T>
inline
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2);

template <class T>
inline
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3);

template <class T>
inline
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4);

template <class T>
inline
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5);

template <class T>
inline
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6);

template <class T>
inline
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6, const T& a7);


template <class T>
inline
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6, const T& a7, const T& a8);

template <class T>
inline
VectorWithOffset<T>
make_vector(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6, const T& a7, const T& a8, const T& a9);


template <class T>
inline
Array<1,T>
make_1d_array(const T& a0);

template <class T>
inline
Array<1,T>
make_1d_array(const T& a0, const T& a1);

template <class T>
inline
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2);

template <class T>
inline
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3);

template <class T>
inline
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4);

template <class T>
inline
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5);

template <class T>
inline
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6);

template <class T>
inline
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6, const T& a7);


template <class T>
inline
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6, const T& a7, const T& a8);

template <class T>
inline
Array<1,T>
make_1d_array(const T& a0, const T& a1, const T& a2, const T& a3, const T& a4,
	    const T& a5, const T& a6, const T& a7, const T& a8, const T& a9);



template <int num_dimensions, class T>
inline
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0);

template <int num_dimensions, class T>
inline
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1);

template <int num_dimensions, class T>
inline
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2);

template <int num_dimensions, class T>
inline
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3);

template <int num_dimensions, class T>
inline
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3, const Array<num_dimensions,T>& a4);

template <int num_dimensions, class T>
inline
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3, const Array<num_dimensions,T>& a4,
	    Array<num_dimensions,T>& a5);

template <int num_dimensions, class T>
inline
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3, const Array<num_dimensions,T>& a4,
	    Array<num_dimensions,T>& a5, const Array<num_dimensions,T>& a6);

template <int num_dimensions, class T>
inline
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3, const Array<num_dimensions,T>& a4,
	    Array<num_dimensions,T>& a5, const Array<num_dimensions,T>& a6, const Array<num_dimensions,T>& a7);


template <int num_dimensions, class T>
inline
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions,T>& a0, const Array<num_dimensions,T>& a1, const Array<num_dimensions,T>& a2, const Array<num_dimensions,T>& a3, const Array<num_dimensions,T>& a4,
	    Array<num_dimensions,T>& a5, const Array<num_dimensions,T>& a6, const Array<num_dimensions,T>& a7, const Array<num_dimensions,T>& a8);

template <int num_dimensions, class T>
inline
Array<num_dimensions+1,T>
make_array(const Array<num_dimensions-1,T>& a0, const Array<num_dimensions-1,T>& a1, const Array<num_dimensions-1,T>& a2, const Array<num_dimensions-1,T>& a3, const Array<num_dimensions-1,T>& a4,
	   Array<num_dimensions-1,T>& a5, const Array<num_dimensions-1,T>& a6, const Array<num_dimensions-1,T>& a7, const Array<num_dimensions-1,T>& a8, const Array<num_dimensions-1,T>& a9);


END_NAMESPACE_STIR

#include "stir/make_array.inl"

#endif
