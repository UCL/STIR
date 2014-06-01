//
//

#ifndef __stir_recon_buildblock_ProjMatrixByDensel_H__
#define __stir_recon_buildblock_ProjMatrixByDensel_H__

/*!

 \file
  \ingroup recon_buildblock 
  \brief declaration of ProjMatrixByDensel and its helpers classes
  
  \author Kris Thielemans
   
*/
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/



#include "stir/RegisteredObject.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneDensel.h"
#include "stir/recon_buildblock/DataSymmetriesForDensels.h"
#include "stir/Densel.h"
#include "stir/ProjDataInfo.h" // todo replace by forward declaration
#include <map>
#include "stir/shared_ptr.h"

// define a local preprocessor symbol to keep code relatively clean
#ifdef STIR_NO_MUTABLE
#define STIR_MUTABLE_CONST
#else
#define STIR_MUTABLE_CONST const
#endif

START_NAMESPACE_STIR
	    
//template <typename elemT> class RelatedViewgrams;	    
//class Densel;	    
template <int num_dimensions, typename elemT> class DiscretisedDensity;
/*!
\ingroup recon_buildblock
\brief 
  This is the (abstract) base class for all projection matrices 
  which are organised by 'Densel'.

  This class provides essentially only 2 public members: a method to get a 
  'row' of the matrix, and a method to get information on the symmetries.  

  Currently, the class provides for some (basic) caching.
  This functionality will probably be moved to a new class 
  ProjMatrixByDenselWithCache. (TODO)
*/
class ProjMatrixByDensel :  public RegisteredObject<ProjMatrixByDensel>
{
public:
  
  virtual ~ProjMatrixByDensel() {}

  //! To be called before any calculation is performed
  /*! Note that get_proj_matrix_elems_for_one_Densel() will expect objects of
      compatible sizes and other info.
  */
  virtual void set_up(
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
  ) = 0;

  //! get a pointer to an object encoding all symmetries that are used by this ProjMatrixByDensel
  virtual const DataSymmetriesForDensels* get_symmetries_ptr() const = 0;
  
  
  //! The main method for getting a column of the matrix.
  /*! 
  The ProjMatrixElemsForOneDensel argument will be overwritten
  (i.e. data is NOT appended).
  
  The implementation is inline as it just gets it in
  terms of the cached__proj_matrix_elems_for_one_densel or 
  calculate_proj_matrix_elems_for_one_densel.*/
  inline void 
    get_proj_matrix_elems_for_one_densel(
       ProjMatrixElemsForOneDensel&,
       const Densel&) STIR_MUTABLE_CONST;
  
#if 0
  // TODO
  /*! \brief Facility to write the 'independent' part of the matrix to file.
  
   This is useful for versions for which computation is very slow.
   \warning Format has to be compatible with ProjMatrixByDenselFromFile
  */
  virtual void write_to_file_by_densel(
    const char * const file_name_without_extension) const;
#endif  
  // TODO implement this one at some point ?
  /*
  virtual void write_to_file_by_bin(
  const char * const file_name_without_extension);
  */
  
  //void set_maximum_cache_size(const unsigned long size){;}        
  void enable_cache(bool v){cache_disabled = v;}
  /* TODO
  void set_subset_usage(const SubsetInfo&, const int num_access_times);
  */
  
  
  
protected:
  

  //! default ctor (enables caching)
  ProjMatrixByDensel();  
  
  /*! \brief This method needs to be implemented in the derived class.
  
    Densel-coordinates are obtained via the ProjMatrixElemsForOneDensel::get_Densel() method.

    Note that 'calculate' could just as well mean 'get from file'
  */
  virtual void 
    calculate_proj_matrix_elems_for_one_densel(
    ProjMatrixElemsForOneDensel& 
    ) const = 0;
  
  /////////////////////////////// caching stuff //////////////////////

  bool cache_disabled;  

  /*! \brief The method that tries to get data from the cache.
  
   If it succeeds, it overwrites the ProjMatrixElemsForOneDensel parameter and
   returns Succeeded::yes, otherwise it does not touch the ProjMatrixElemsForOneDensel
   and returns Succeeded::false.
  */
  Succeeded get_cached_proj_matrix_elems_for_one_densel(
	 	 ProjMatrixElemsForOneDensel&
                 ) const;		
  
  //! The method to store data in the cache.
  inline void  cache_proj_matrix_elems_for_one_densel( const ProjMatrixElemsForOneDensel&)
    STIR_MUTABLE_CONST;

private:
  
  typedef unsigned int CacheKey;

#ifndef STIR_NO_NAMESPACES
  typedef std::map<CacheKey, ProjMatrixElemsForOneDensel>   MapProjMatrixElemsForOneDensel;
#else
  typedef map<CacheKey, ProjMatrixElemsForOneDensel>   MapProjMatrixElemsForOneDensel;
 #endif
  typedef MapProjMatrixElemsForOneDensel::iterator MapProjMatrixElemsForOneDenselIterator;
  typedef MapProjMatrixElemsForOneDensel::const_iterator const_MapProjMatrixElemsForOneDenselIterator;
 
  //! collection of  ProjMatrixElemsForOneDensel (internal cache )   
#ifndef STIR_NO_MUTABLE
  mutable
#endif
    MapProjMatrixElemsForOneDensel cache_collection;
         
  //! create the key for caching
  inline static CacheKey cache_key(const Densel& Densel);

   
};



END_NAMESPACE_STIR

#include "local/stir/recon_buildblock/ProjMatrixByDensel.inl"

#undef STIR_MUTABLE_CONST

#endif // __ProjMatrixByDensel_H__



