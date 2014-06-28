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
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/SymmetryOperation.h"

START_NAMESPACE_STIR

const DataSymmetriesForBins*
ProjMatrixByBin:: get_symmetries_ptr() const
{
  return  symmetries_ptr.get();
}

#if 0
  \warning   Preconditions when cache_stores_only_basic_bins==true
  <ul><li>all coordinates non-negative
      <li>segment_num coded in 8 bits
      <li>view coded in 9 bits
      <li>axial_pos_num  in 6 bits
      <li>tangential_pos_num in 9 bits   
  </ul>
  Preconditions when cache_stores_only_basic_bins==false
#endif
/*!
  \warning Preconditions
  <ul><li>abs(segment_num) coded in 7 bits
      <li>view non-negative and coded in 9 bits
      <li>axial_pos_num non-negative and in 6 bits
      <li>abs(tangential_pos_num) in  8 bits   
 */
ProjMatrixByBin::CacheKey
ProjMatrixByBin::cache_key(const Bin& bin) const
{
#if 0
  // this was an attempt to allow more bits for certain numbers by relying on 
  // the fact that segment_num and tangetnail_pos_num was always positive.
  // However, this depends on which symmetries you are using.
  if (cache_stores_only_basic_bins)
  {
    assert(bin.segment_num()>=0);
    assert(bin.segment_num() < (1<<8));  
    assert(bin.view_num() >= 0);
    assert(bin.view_num() < (1<<9));
    assert(bin.axial_pos_num() >= 0);
    assert(bin.axial_pos_num() < (1<<6));
    assert(bin.tangential_pos_num() >= 0);
    assert(bin.tangential_pos_num() < (1<<9));
    return (CacheKey)(
        (static_cast<unsigned int>(bin.axial_pos_num())<<26) 
        | (static_cast<unsigned int>(bin.view_num()) << 17) 
        | (static_cast<unsigned int>(bin.segment_num())) << 9) 
        |  static_cast<unsigned int>(bin.tangential_pos_num() );    	
  }
  else
#endif
{
 //KTBMCHANGE allow more entries by going to 64-bit
#if 0
    assert(abs(bin.segment_num()) < (1<<7));
    assert(bin.view_num() >= 0);
    assert(bin.view_num() < (1<<9));
    assert(bin.axial_pos_num() >= 0);
    assert(bin.axial_pos_num() < (1<<6));
    assert(abs(bin.tangential_pos_num()) < (1<<8));
    return (CacheKey)(
        (static_cast<unsigned int>(bin.axial_pos_num())<<26) 
        | (static_cast<unsigned int>(bin.view_num()) << 17) 
        | (static_cast<unsigned int>(bin.segment_num()>=0?0:1) << 16)
        | (static_cast<unsigned int>(abs(bin.segment_num())) << 9) 
        | (static_cast<unsigned int>(bin.tangential_pos_num()>=0?0:1) << 8)
        |  static_cast<unsigned int>(abs(bin.tangential_pos_num())) );    	
  }
#else
      assert(abs(bin.segment_num()) < (1<<7));
    assert(bin.view_num() >= 0);
    assert(bin.view_num() < (1<<9));
    assert(bin.axial_pos_num() >= 0);
    assert(static_cast<boost::uint64_t>(bin.axial_pos_num()) < (static_cast<boost::uint64_t>(1)<<38));
    assert(abs(bin.tangential_pos_num()) < (1<<8));
    return (CacheKey)( 
        (static_cast<boost::uint64_t>(bin.axial_pos_num())<<26) 
        | (static_cast<boost::uint64_t>(bin.view_num()) << 17) 
        | (static_cast<boost::uint64_t>(bin.segment_num()>=0?0:1) << 16)
        | (static_cast<boost::uint64_t>(abs(bin.segment_num())) << 9) 
        | (static_cast<boost::uint64_t>(bin.tangential_pos_num()>=0?0:1) << 8)
        |  static_cast<boost::uint64_t>(abs(bin.tangential_pos_num())) );    	
  }
#endif
  } 

void  
ProjMatrixByBin::
cache_proj_matrix_elems_for_one_bin(
                                    const ProjMatrixElemsForOneBin& probabilities) STIR_MUTABLE_CONST
{ 
  if ( cache_disabled ) return;
  
  //std::cerr << "cached lor size " << probabilities.size() << " capacity " << probabilities.capacity() << std::endl;    
  // insert probabilities into the collection	
  cache_collection.insert(MapProjMatrixElemsForOneBin::value_type( cache_key(probabilities.get_bin()), 
                                                                   probabilities));  
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
    std::auto_ptr<SymmetryOperation> symm_ptr = 
      symmetries_ptr->find_symmetry_operation_from_basic_bin(basic_bin);
    
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
      std::auto_ptr<SymmetryOperation> symm_ptr = 
        symmetries_ptr->find_symmetry_operation_from_basic_bin(basic_bin);

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
