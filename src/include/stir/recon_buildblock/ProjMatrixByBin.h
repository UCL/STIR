#ifndef __stir_recon_buildblock_ProjMatrixByBin_H__
#define __stir_recon_buildblock_ProjMatrixByBin_H__

/*!

 \file
  \ingroup projection 
  \brief declaration of stir::ProjMatrixByBin and its helpers classes
  
  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2015 University College London

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

#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/DataSymmetriesForBins.h"
#include "stir/shared_ptr.h"
#include "stir/VectorWithOffset.h"
#include "stir/TimedObject.h"
#include <boost/cstdint.hpp>
//#include <map>
#include <boost/unordered_map.hpp>

// define a local preprocessor symbol to keep code relatively clean
#ifdef STIR_NO_MUTABLE
#define STIR_MUTABLE_CONST
#else
#define STIR_MUTABLE_CONST const
#endif

START_NAMESPACE_STIR

/* TODO 
class ProjMatrixElemsForOneViewgram;
class SubsetInfo;
*/
	    
class Bin;	    
	    
/*!
\ingroup projection
\brief 
  This is the (abstract) base class for all projection matrices 
  which are organised by 'bin'.

  This class provides essentially only 2 public members: a method to get a 
  'row' of the matrix, and a method to get information on the symmetries.  

  Currently, the class provides for some (basic) caching.
  This functionality will probably be moved to a new class 
  ProjMatrixByBinWithCache. (TODO)

  \par Parsing parameters

  The following parameters can be set (default values are indicated):
  \verbatim
  disable caching := false
  store only basic bins in cache := true
  \endverbatim
  The 2nd option allows to cache the whole matrix. This results in the fastest
  behaviour IF your system does not start swapping. The default choice caches 
  only the 'basic' bins, and computes symmetry related bins from the 'basic' ones.
*/
class ProjMatrixByBin :  
  public RegisteredObject<ProjMatrixByBin>,  
  public ParsingObject,
  public TimedObject
{
public:
  
  virtual ~ProjMatrixByBin() {}

  //! To be called before any calculation is performed
  /*! Note that get_proj_matrix_elems_for_one_bin() will expect objects of
      compatible sizes and other info.

      \warning: Any implementation of set_up in a derived class has to 
      call ProjMatrixByBin::set_up first.
  */
  virtual void set_up(
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
  ) = 0;

  //! get a pointer to an object encoding all symmetries that are used by this ProjMatrixByBin
  inline const  DataSymmetriesForBins* get_symmetries_ptr() const;
  
  
  //! The main method for getting a row of the matrix.
  /*! 
  The ProjMatrixElemsForOneBin argument will be overwritten
  (i.e. data is NOT appended).
  
  The implementation is inline as it just gets it in
  terms of the cached_proj_matrix_elems_for_one_bin or 
  calculate_proj_matrix_elems_for_one_bin.*/
  inline void 
    get_proj_matrix_elems_for_one_bin(
       ProjMatrixElemsForOneBin&,
       const Bin&) STIR_MUTABLE_CONST;
  
#if 0
  // TODO
  /*! \brief Facility to write the 'independent' part of the matrix to file.
  
   This is useful for versions for which computation is very slow.
   \warning Format has to be compatible with ProjMatrixByBinFromFile
  */
  virtual void write_to_file_by_bin(
    const char * const file_name_without_extension) const;
#endif  
  // TODO implement this one at some point ?
  /*
  virtual void write_to_file_by_voxel(
  const char * const file_name_without_extension);
  */
  
  //void set_maximum_cache_size(const unsigned long size){;}        
  /* TODO
  void set_subset_usage(const SubsetInfo&, const int num_access_times);
  */
  void enable_cache(const bool v = true);
  void store_only_basic_bins_in_cache(const bool v = true) ;

  bool is_cache_enabled() const;
  bool does_cache_store_only_basic_bins() const;

  // void reserve_num_elements_in_cache(const std::size_t);
  //! Remove all elements from the cache
  void clear_cache() STIR_MUTABLE_CONST;

  
protected:
  shared_ptr<DataSymmetriesForBins> symmetries_ptr;
  
  //! default ctor (calls set_defaults())
  /*! Note that due to the C++ definition (and some good reasons), 
      ProjMatrixByBin::set_defaults() is called,
      even though this is a virtual function.
   */
  ProjMatrixByBin();  
  
  /*! \brief This method needs to be implemented in the derived class.
  
    bin-coordinates are obtained via the ProjMatrixElemsForOneBin::get_bin() method.

    Note that 'calculate' could just as well mean 'get from file'
  */
  virtual void 
    calculate_proj_matrix_elems_for_one_bin(
    ProjMatrixElemsForOneBin& 
    ) const = 0;

  /////////////////////////////// parsing stuff //////////////////////
  
  //! sets value for caching configuration (enables caching, but for 'basic' bins only)
  /*! Has to be called by set_defaults in the leaf-class */
  virtual void set_defaults();
  //! sets keys for caching configuration
  /*! Has to be called by initialise_keymap in the leaf-class */
  virtual void initialise_keymap();
  //! Checks if parameters have sensible values
  /*! Has to be called by post_processing in the leaf-class */
  virtual bool post_processing();
  
  /////////////////////////////// caching stuff //////////////////////

  bool cache_disabled;  
  bool cache_stores_only_basic_bins;

  /*! \brief The method that tries to get data from the cache.
  
   If it succeeds, it overwrites the ProjMatrixElemsForOneBin parameter and
   returns Succeeded::yes, otherwise it does not touch the ProjMatrixElemsForOneBin
   and returns Succeeded::false.
  */
  Succeeded get_cached_proj_matrix_elems_for_one_bin(
	 	 ProjMatrixElemsForOneBin&
                 ) const;		
  
  //! The method to store data in the cache.
  void  cache_proj_matrix_elems_for_one_bin( const ProjMatrixElemsForOneBin&)
    STIR_MUTABLE_CONST;

private:
  
  typedef boost::uint32_t CacheKey;

	//  typedef std::map<CacheKey, ProjMatrixElemsForOneBin>   MapProjMatrixElemsForOneBin;
  typedef boost::unordered_map<CacheKey, ProjMatrixElemsForOneBin>   MapProjMatrixElemsForOneBin;
  typedef MapProjMatrixElemsForOneBin::iterator MapProjMatrixElemsForOneBinIterator;
  typedef MapProjMatrixElemsForOneBin::const_iterator const_MapProjMatrixElemsForOneBinIterator;
 
  //! collection of  ProjMatrixElemsForOneBin (internal cache )   
#ifndef STIR_NO_MUTABLE
  mutable
#endif
    VectorWithOffset<VectorWithOffset<MapProjMatrixElemsForOneBin> > cache_collection;
         
  //! create the key for caching
  // KT 15/05/2002 not static anymore as it uses cache_stores_only_basic_bins
  CacheKey cache_key(const Bin& bin) const;

   
};



END_NAMESPACE_STIR

#include "stir/recon_buildblock/ProjMatrixByBin.inl"

#undef STIR_MUTABLE_CONST

#endif // __ProjMatrixByBin_H__



