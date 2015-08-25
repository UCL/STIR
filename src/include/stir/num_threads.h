/*
    Copyright (C) 2015, University College London
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
  \ingroup threads

  \brief Implementation of functions related to setting/getting the number of threads

  \author Kris Thielemans  
*/

#include "stir/common.h"

START_NAMESPACE_STIR

//! Get current maximum number of threads
/*! \ingroup threads
  This returns the maxmimum number of threads to be used by STIR.
  Usually this should be equal to what you set earlier via set_num_threads().
  
  Currently only useful when compiled with OpenMP support. Corresponds then
  to omp_get_max_threads()
*/
int get_max_num_threads();

//! Set current number of threads
/*! \ingroup threads
  This can be used to increase/decrease the number of threads used by STIR
  from the default value (see get_default_num_threads()).

  If \a num_threads is zero (and therefore when no arguments are passed) and
  if this is the first time this function is called, it will call
  set_default_num_threads().

  Therefore, after calling <code>set_num_threads(5)</code>, future calls to
  <code>set_num_threads()</code> will keep using 5 threads. This is used
  internally in STIR (e.g. in distributable_computation) to normally
  use the default number of threads, but let the user change it.
*/
void set_num_threads(const int num_threads = 0);

//! Get default number of threads
/*! \ingroup threads
 If OpenMP support is enabled, the default is normally set from the
 \c OMP_NUM_THREADS environment variable. However, if this is is
 not set, we use ~90% of the available processors.
  
  Currently only useful when compiled with OpenMP support. 
*/
int get_default_num_threads();

//! set current number of threads to the default
/*! \ingroup threads
  \see get_default_num_threads()

  Currently only useful when compiled with OpenMP support. 
*/
void set_default_num_threads();

END_NAMESPACE_STIR
