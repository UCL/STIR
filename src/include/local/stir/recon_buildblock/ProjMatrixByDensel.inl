//
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Implementations of inline functions for class ProjMatrixByDensel

  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2002, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/SymmetryOperation.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

void ProjMatrixByDensel::
get_proj_matrix_elems_for_one_densel(
				     ProjMatrixElemsForOneDensel& probabilities,
				     const Densel& densel) STIR_MUTABLE_CONST
{  
  // set to empty
  probabilities.erase();
  

  // find basic densel
  Densel basic_densel = densel;
    
  std::unique_ptr<SymmetryOperation> symm_ptr = 
    get_symmetries_ptr()->find_symmetry_operation_from_basic_densel(basic_densel);
  
  probabilities.set_densel(basic_densel);

  // check if in cache  
  if (get_cached_proj_matrix_elems_for_one_densel(probabilities) ==
      Succeeded::no)
    
  {
    // call 'calculate' just for the basic densel
    calculate_proj_matrix_elems_for_one_densel(probabilities);
#ifndef NDEBUG
    probabilities.check_state();
#endif
    cache_proj_matrix_elems_for_one_densel(probabilities);		
  }   
  
  // now transform to original densel
  symm_ptr->transform_proj_matrix_elems_for_one_densel(probabilities);  
  
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
ProjMatrixByDensel::CacheKey
ProjMatrixByDensel::cache_key(const Densel& densel) 
{
  assert(densel[1] >= 0);
  assert(densel[1] < (1<<10));
  assert(densel[2] >= 0);
  assert(densel[2] < (1<<10));
  assert(densel[3] >= 0);
  assert(densel[3] < (1<<10));
  return (CacheKey)(
      (static_cast<unsigned int>(densel[1])<< 20) 
      | (static_cast<unsigned int>(densel[2]) << 10) 
      | (static_cast<unsigned int>(densel[3])) );    	
} 



//! insert matrix elements for one densel into the cache collection	
void  
ProjMatrixByDensel::
cache_proj_matrix_elems_for_one_densel(
                                    const ProjMatrixElemsForOneDensel& probabilities) STIR_MUTABLE_CONST
{ 
  if ( cache_disabled ) return;
  
  // insert probabilities into the collection	
  cache_collection.insert(MapProjMatrixElemsForOneDensel::value_type( cache_key(probabilities.get_densel()), probabilities));    
  
}


END_NAMESPACE_STIR
