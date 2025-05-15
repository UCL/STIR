/*
    Copyright (C) 2015, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup buildblock

  \brief Implementation of functions related to setting/getting the number of threads

  \author Kris Thielemans
*/

#include "stir/num_threads.h"
#include "stir/warning.h"
#include "stir/info.h"
#include "stir/warning.h"

#include <stdlib.h>

#ifdef STIR_OPENMP
#  include <omp.h>
#endif

START_NAMESPACE_STIR

int
get_max_num_threads()
{
#ifdef STIR_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void
set_num_threads(const int num_threads)
{
  static bool already_set_once = false;

  if (num_threads == 0)
    {
      if (!already_set_once)
        {
          set_default_num_threads();
        }
    }
  else
    {
#ifdef STIR_OPENMP
      omp_set_num_threads(num_threads);

      if (omp_get_max_threads() == 1)
        warning("Using OpenMP with number of threads=1 produces parallel overhead. You should compile without OPENMP support");
#else
      if (num_threads != 1)
        warning("You have asked for more than 1 thread but this is ignored as STIR was not compiled with OpenMP support.");
#endif
    }
  already_set_once = true;
}

int
get_default_num_threads()
{
#ifdef STIR_OPENMP
  int default_num_threads = std::max((int)floor(omp_get_num_procs() * .9), 2);
  if (omp_get_num_procs() == 1)
    default_num_threads = 1;

  if (getenv("OMP_NUM_THREADS") != NULL)
    {
      default_num_threads = atoi(getenv("OMP_NUM_THREADS"));
    }
  return default_num_threads;
#else
  return 1;
#endif
}

void
set_default_num_threads()
{
  set_num_threads(get_default_num_threads());
}

END_NAMESPACE_STIR
