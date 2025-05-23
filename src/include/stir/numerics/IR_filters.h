//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup numerics
  \brief Implementation of the IIR and FIR filters

  \author Charalampos Tsoumpas
  \author Kris Thielemans

*/

#include "stir/shared_ptr.h"
#include <vector>
#include <iostream>
#include <cmath>

START_NAMESPACE_STIR

/*\ingroup numerics
  \brief IIR filter
*/
template <class RandIter1,
  class RandIter2,
  class RandIter3,
  class RandIter4>
void 
inline 
IIR_filter(RandIter1 output_begin_iterator, 
           RandIter1 output_end_iterator,
           const RandIter2 input_begin_iterator, 
           const RandIter2 input_end_iterator,
           const RandIter3 input_factor_begin_iterator,
           const RandIter3 input_factor_end_iterator,
           const RandIter4 pole_begin_iterator,
           const RandIter4 pole_end_iterator,
           const bool if_initial_exists);

/*\ingroup numerics
  \brief IIR filter
*/
template <class RandIter1,
  class RandIter2,
  class RandIter3>               
void 
inline 
FIR_filter(RandIter1 output_begin_iterator, 
           RandIter1 output_end_iterator,
           const RandIter2 input_begin_iterator, 
           const RandIter2 input_end_iterator,
           const RandIter3 input_factor_begin_iterator,
           const RandIter3 input_factor_end_iterator,
           const bool if_initial_exists);
END_NAMESPACE_STIR

#include "stir/numerics/IR_filters.inl"
