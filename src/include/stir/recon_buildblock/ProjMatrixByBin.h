//
// $Id$
//

#ifndef __stir_recon_buildblock_ProjMatrixByBin_H__
#define __stir_recon_buildblock_ProjMatrixByBin_H__

/*!

 \file
  \ingroup recon_buildblock 
  \brief declaration of ProjMatrixByBin and its helpers classes
  
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



#include "stir/RegisteredObject.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/DataSymmetriesForBins.h"
#include "stir/shared_ptr.h"

#include <map>

#ifndef STIR_NO_NAMESPACES
using std::map;
#endif

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
	    
//template <typename elemT> class RelatedViewgrams;	    
class Bin;	    
	    
/*!
\ingroup recon_buildblock
\brief 
  This is the (abstract) base class for all projection matrices 
  which are organised by 'bin'.

  This class provides essentially only 2 public members: a method to get a 
  'row' of the matrix, and a method to get information on the symmetries.  

  Currently, the class provides for some (basic) caching.
  This functionality will probably be moved to a new class 
  ProjMatrixByBinWithCache. (TODO)
*/
class ProjMatrixByBin :  public RegisteredObject<ProjMatrixByBin>
{
public:
  
  virtual ~ProjMatrixByBin() {}

  //! To be called before any calculation is performed
  /*! Note that get_proj_matrix_elems_for_one_bin() will expect objects of
      compatible sizes and other info.
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
  terms of the cached__proj_matrix_elems_for_one_bin or 
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
  void enable_cache(bool v){cache_disabled = v;}
  /* TODO
  void set_subset_usage(const SubsetInfo&, const int num_access_times);
  */
  
  
  
protected:
  shared_ptr<DataSymmetriesForBins> symmetries_ptr;
  

  //! default ctor (enables caching)
  ProjMatrixByBin();  
  
  /*! \brief This method needs to be implemented in the derived class.
  
    bin-coordinates are obtained via the ProjMatrixElemsForOneBin::get_bin() method.

    Note that 'calculate' could just as well mean 'get from file'
  */
  virtual void 
    calculate_proj_matrix_elems_for_one_bin(
    ProjMatrixElemsForOneBin& 
    ) const = 0;
  
  /////////////////////////////// caching stuff //////////////////////

  bool cache_disabled;  

  /*! \brief The method that tries to get data from the cache.
  
   If it succeeds, it overwrites the ProjMatrixElemsForOneBin parameter and
   returns Succeeded::yes, otherwise it does not touch the ProjMatrixElemsForOneBin
   and returns Succeeded::false.
  */
  Succeeded get_cached_proj_matrix_elems_for_one_bin(
	 	 ProjMatrixElemsForOneBin&
                 ) const;		
  
  //! The method to store data in the cache.
  inline void  cache_proj_matrix_elems_for_one_bin( const ProjMatrixElemsForOneBin&)
    STIR_MUTABLE_CONST;

private:
  
  typedef unsigned int CacheKey;

#ifndef STIR_NO_NAMESPACES
  typedef std::map<CacheKey, ProjMatrixElemsForOneBin>   MapProjMatrixElemsForOneBin;
#else
  typedef map<CacheKey, ProjMatrixElemsForOneBin>   MapProjMatrixElemsForOneBin;
 #endif
  typedef MapProjMatrixElemsForOneBin::iterator MapProjMatrixElemsForOneBinIterator;
  typedef MapProjMatrixElemsForOneBin::const_iterator const_MapProjMatrixElemsForOneBinIterator;
 
  //! collection of  ProjMatrixElemsForOneBin (internal cache )   
#ifndef STIR_NO_MUTABLE
  mutable
#endif
    MapProjMatrixElemsForOneBin cache_collection;
         
  //! create the key for caching
  inline static CacheKey cache_key(const Bin& bin);

   
};



END_NAMESPACE_STIR

#include "stir/recon_buildblock/ProjMatrixByBin.inl"

#undef STIR_MUTABLE_CONST

#endif // __ProjMatrixByBin_H__



