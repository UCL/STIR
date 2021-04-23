//
//
/*!

  \file
  \ingroup projection

  \brief Implementations of inline functions for class stir::ProjMatrixByBin

  \author Mustapha Sadki 
  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2013, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/SymmetryOperation.h"

START_NAMESPACE_STIR

const DataSymmetriesForBins*
ProjMatrixByBin:: get_symmetries_ptr() const
{
  return  symmetries_sptr.get();
}

const shared_ptr<DataSymmetriesForBins>
ProjMatrixByBin:: get_symmetries_sptr() const
{
  return  symmetries_sptr;
}

inline void 
ProjMatrixByBin::
get_proj_matrix_elems_for_one_bin(
                                  ProjMatrixElemsForOneBin& probabilities,
                                  const Bin& bin) STIR_MUTABLE_CONST
{  
  // start_timers(); TODO, can't do this in a const member

  // set to empty
  probabilities.erase();
  
  if (cache_stores_only_basic_bins)
  {
    // find basic bin
    Bin basic_bin = bin;    
    unique_ptr<SymmetryOperation> symm_ptr = 
      symmetries_sptr->find_symmetry_operation_from_basic_bin(basic_bin);
    
    probabilities.set_bin(basic_bin);
    // check if basic bin is in cache  
    if (get_cached_proj_matrix_elems_for_one_bin(probabilities) ==
      Succeeded::no)
    {
      // call 'calculate' just for the basic bin
      calculate_proj_matrix_elems_for_one_bin(probabilities);
#ifndef NDEBUG
      probabilities.check_state();
#endif
      cache_proj_matrix_elems_for_one_bin(probabilities);		
    }
    
    // now transform to original bin
    symm_ptr->transform_proj_matrix_elems_for_one_bin(probabilities);  
  }
  else // !cache_stores_only_basic_bins
  {
    probabilities.set_bin(bin);
    // check if in cache  
    if (get_cached_proj_matrix_elems_for_one_bin(probabilities) ==
      Succeeded::no)
    {
      // find basic bin
      Bin basic_bin = bin;  
      unique_ptr<SymmetryOperation> symm_ptr = 
        symmetries_sptr->find_symmetry_operation_from_basic_bin(basic_bin);

      probabilities.set_bin(basic_bin);
      // check if basic bin is in cache
      if (get_cached_proj_matrix_elems_for_one_bin(probabilities) ==
        Succeeded::no)
      {
        // call 'calculate' just for the basic bin
        calculate_proj_matrix_elems_for_one_bin(probabilities);
#ifndef NDEBUG
        probabilities.check_state();
#endif
        cache_proj_matrix_elems_for_one_bin(probabilities);
      }
      symm_ptr->transform_proj_matrix_elems_for_one_bin(probabilities);
      cache_proj_matrix_elems_for_one_bin(probabilities);      
    }
  }  
  // stop_timers(); TODO, can't do this in a const member
}

END_NAMESPACE_STIR
