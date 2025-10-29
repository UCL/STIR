#ifndef __stir_recon_buildblock_ProjMatrixByBin_H__
#define __stir_recon_buildblock_ProjMatrixByBin_H__

/*!

 \file
  \ingroup projection
  \brief declaration of stir::ProjMatrixByBin and its helpers classes

  \author Nikos Efthimiou
  \author Mustapha Sadki
  \author Kris Thielemans
  \author Robert Twyman
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2015, 2022 University College London
    Copyright (C) 2016, University of Hull

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/DataSymmetriesForBins.h"
#include "stir/shared_ptr.h"
#include "stir/VectorWithOffset.h"
#include "stir/TimedObject.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/numerics/FastErf.h"
#include <cstdint>
//#include <map>
#include <unordered_map>
#ifdef STIR_OPENMP
#  include <omp.h>
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
class ProjMatrixByBin : public RegisteredObject<ProjMatrixByBin>, public TimedObject
{
public:
  ~ProjMatrixByBin() override {}

  //! To be called before any calculation is performed
  /*! Note that get_proj_matrix_elems_for_one_bin() will expect objects of
      compatible sizes and other info.

      \warning: Any implementation of set_up in a derived class has to
      call ProjMatrixByBin::set_up first.
  */
  virtual void set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
                      const shared_ptr<const DiscretisedDensity<3, float>>& density_info_ptr // TODO should be Info only
                      )
      = 0;

  virtual ProjMatrixByBin* clone() const = 0;

  //! get a pointer to an object encoding all symmetries that are used by this ProjMatrixByBin
  inline const DataSymmetriesForBins* get_symmetries_ptr() const;
  //! get a shared_ptr to an object encoding all symmetries that are used by this ProjMatrixByBin
  inline const shared_ptr<DataSymmetriesForBins> get_symmetries_sptr() const;

  //! The main method for getting a row of the matrix.
  /*!
  The ProjMatrixElemsForOneBin argument will be overwritten
  (i.e. data is NOT appended).

  The implementation is inline as it just gets it in
  terms of the cached_proj_matrix_elems_for_one_bin or
  calculate_proj_matrix_elems_for_one_bin.*/
  inline void get_proj_matrix_elems_for_one_bin(ProjMatrixElemsForOneBin&, const Bin&) const;

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

  // void set_maximum_cache_size(const unsigned long size){;}
  /* TODO
  void set_subset_usage(const SubsetInfo&, const int num_access_times);
  */
  void enable_cache(const bool v = true);
  void store_only_basic_bins_in_cache(const bool v = true);

  bool is_cache_enabled() const;
  bool does_cache_store_only_basic_bins() const;

  // void reserve_num_elements_in_cache(const std::size_t);
  //! Remove all elements from the cache
  void clear_cache() const;

protected:
  shared_ptr<DataSymmetriesForBins> symmetries_sptr;

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
  virtual void calculate_proj_matrix_elems_for_one_bin(ProjMatrixElemsForOneBin&) const = 0;

  /////////////////////////////// parsing stuff //////////////////////

  //! sets value for caching configuration (enables caching, but for 'basic' bins only)
  /*! Has to be called by set_defaults in the leaf-class */
  void set_defaults() override;
  //! sets keys for caching configuration
  /*! Has to be called by initialise_keymap in the leaf-class */
  void initialise_keymap() override;
  //! Checks if parameters have sensible values
  /*! Has to be called by post_processing in the leaf-class */
  bool post_processing() override;

  /////////////////////////////// caching stuff //////////////////////

  bool cache_disabled;
  bool cache_stores_only_basic_bins;
  //! If activated TOF reconstruction will be performed.
  bool tof_enabled;

  /*! \brief The method that tries to get data from the cache.

   If it succeeds, it overwrites the ProjMatrixElemsForOneBin parameter and
   returns Succeeded::yes, otherwise it does not touch the ProjMatrixElemsForOneBin
   and returns Succeeded::false.
  */
  Succeeded get_cached_proj_matrix_elems_for_one_bin(ProjMatrixElemsForOneBin&) const;

  //! We need a local copy of the discretised density in order to find the
  //! cartesian coordinates of each voxel.
  shared_ptr<const VoxelsOnCartesianGrid<float>> image_info_sptr;

  //! We need a local copy of the proj_data_info to get the integration boundaries and RayTracing
  shared_ptr<const ProjDataInfo> proj_data_info_sptr;

  //! The method to store data in the cache.
  void cache_proj_matrix_elems_for_one_bin(const ProjMatrixElemsForOneBin&) const;

private:
  typedef std::uint64_t CacheKey;
  //! \name bit-field sizes for the cache key
  // note: sum needs to be less than  64 - 3  (for the 3 sign bits)
  //@{
  const CacheKey tang_pos_bits = 12;
  const CacheKey axial_pos_bits = 28;
  const CacheKey timing_pos_bits = 20;
  //@}
  //  typedef std::map<CacheKey, ProjMatrixElemsForOneBin>   MapProjMatrixElemsForOneBin;
  typedef std::unordered_map<CacheKey, ProjMatrixElemsForOneBin> MapProjMatrixElemsForOneBin;
  typedef MapProjMatrixElemsForOneBin::iterator MapProjMatrixElemsForOneBinIterator;
  typedef MapProjMatrixElemsForOneBin::const_iterator const_MapProjMatrixElemsForOneBinIterator;

  //! collection of  ProjMatrixElemsForOneBin (internal cache )
  mutable VectorWithOffset<VectorWithOffset<MapProjMatrixElemsForOneBin>> cache_collection;
#ifdef STIR_OPENMP
  mutable VectorWithOffset<VectorWithOffset<omp_lock_t>> cache_locks;
#endif

  //! create the key for caching
  // KT 15/05/2002 not static anymore as it uses cache_stores_only_basic_bins
  CacheKey cache_key(const Bin& bin) const;

  //! Activates the application of the timing kernel to the LOR
  //! and performs initial set_up().
  //! \warning Must be called during set_up()
  void enable_tof(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr, const bool v = true);

  //! A local copy of the scanner's time resolution in mm.
  float gauss_sigma_in_mm;
  //! 1/(2*sigma_in_mm)
  float r_sqrt2_gauss_sigma;

  //! The function which actually applies the TOF kernel on the LOR.
  inline void apply_tof_kernel(ProjMatrixElemsForOneBin& probabilities) const;

  //! Get the interal value erf(m - v_j) - erf(m -v_j)
  inline float get_tof_value(const float d1, const float d2) const;

  //! erf map
  FastErf erf_interpolation;
};

END_NAMESPACE_STIR

#include "stir/recon_buildblock/ProjMatrixByBin.inl"

#endif // __ProjMatrixByBin_H__
