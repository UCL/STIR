//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Implementations of inline functions for class ProjMatrixByBin

  \author Mustapha Sadki 
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/SymmetryOperation.h"

START_NAMESPACE_STIR

const DataSymmetriesForBins*
ProjMatrixByBin:: get_symmetries_ptr() const
{
  return  symmetries_ptr.get();
}


inline void ProjMatrixByBin::get_proj_matrix_elems_for_one_bin(
                                                               ProjMatrixElemsForOneBin& probabilities,
                                                               const Bin& bin) STIR_MUTABLE_CONST
{  
  // set to empty
  probabilities.erase();
  

  // find basic bin
  Bin basic_bin = bin;
    
  auto_ptr<SymmetryOperation> symm_ptr = 
    symmetries_ptr->find_symmetry_operation_to_basic_bin(basic_bin);
  
  probabilities.set_bin(basic_bin);

  // check if in cache  
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

/*!
\warning   Preconditions: 
  <ul><li>all coordinates non-negative
      <li>segment_num coded in 8 bits
      <li>view coded in 10 bits
      <li>axial_pos_num  in 6 bits
      <li>tangential_pos_num in  8 bits   
  </ul>
 */
ProjMatrixByBin::CacheKey
ProjMatrixByBin::cache_key(const Bin& bin) 
{ assert(bin.segment_num()>=0);
  assert(bin.segment_num() < (1<<8));  
  assert(bin.view_num() >= 0);
  assert(bin.view_num() < (1<<10));
  assert(bin.axial_pos_num() >= 0);
  assert(bin.axial_pos_num() < (1<<6));
  assert(bin.tangential_pos_num() >= 0);
  assert(bin.tangential_pos_num() < (1<<8));
  return (CacheKey)(
      (static_cast<unsigned int>(bin.axial_pos_num())<<26) 
      | (static_cast<unsigned int>(bin.view_num()) << 16) 
      | (static_cast<unsigned int>(bin.segment_num())) << 8) 
      |  static_cast<unsigned int>(bin.tangential_pos_num() );    	
} 



//! insert matrix elements for one bin into the cache collection	
void  
ProjMatrixByBin::
cache_proj_matrix_elems_for_one_bin(
                                    const ProjMatrixElemsForOneBin& probabilities) STIR_MUTABLE_CONST
{ 
  if ( cache_disabled ) return;
  
  // insert probabilities into the collection	
  cache_collection.insert(MapProjMatrixElemsForOneBin::value_type( cache_key(probabilities.get_bin()), probabilities));    
  
}


END_NAMESPACE_STIR
