//
//
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2018, 2022 University College London
    Copyright (C) 2021, University of Pennsylvania
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \brief Declaration of class 
  stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeData

  \author Kris Thielemans
  \author Sanida Mustafovic

*/

#ifndef __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndListModeData_H__
#define __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndListModeData_H__


//#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
#include "stir/listmode/ListModeData.h"
#include "stir/ParseAndCreateFrom.h"
#include "stir/TimeFrameDefinitions.h"

START_NAMESPACE_STIR

/*!
  \ingroup GeneralisedObjectiveFunction
  \brief An objective function class appropriate for PET list mode data

  The statistics for list mode data is slightly different from having a set
  of counts, see the paper
  H. H. Barrett, L. Parra, and T. White, 
  <i>List-mode likelihood,</i> J. Optical Soc. Amer. A, vol. 14, no. 11, 1997.

  However, it is intuitive that the list mode likelihood can be
  derived from that 'binned' likelihood by taking smaller and smaller
  time bins. One can then see that the gradient of the list mode
  likelihood can be computed similar to the 'binned' case, but now
  with a sum over events. The sensitivity still needs a sum (or integral)
  over all possible detections.

  At present, STIR does not contain any classes to forward/back
  project list mode data explictly. So, the sum over events cannot
  be computed for arbitrary list mode data. This is currently
  done in derived classes.

  For list mode reconstructions, computing the sensitivity is sometimes
  conceptually very difficult (how to compute the integral) or impractical.
  So, even if we will be able to compute the sum over events in a generic way,
  the add_subset_sensitivity() function will have to be implemented by
  a derived class, specific for the measurement.

*/
template <typename TargetT>
class PoissonLogLikelihoodWithLinearModelForMeanAndListModeData: 
public PoissonLogLikelihoodWithLinearModelForMean<TargetT> 
{
private: 

  typedef PoissonLogLikelihoodWithLinearModelForMean<TargetT> base_type;

public:   

 
  PoissonLogLikelihoodWithLinearModelForMeanAndListModeData();  
 
  //virtual TargetT * construct_target_ptr();  
 
  Succeeded
   set_up_before_sensitivity(shared_ptr <const TargetT > const& target_sptr) override;
 
  //! time frame definitions
  /*! \todo This is currently used to be able to compute the gradient for 
    one time frame. However, it probably does not belong here.
    For instance when fitting kinetic model parameters from list
    mode data, time frames are in principle irrelevant. So, we will
    probably shift this to the derived class.
  */
    TimeFrameDefinitions frame_defs;

    void set_normalisation_sptr(const shared_ptr<BinNormalisation>&) override;
    void set_additive_proj_data_sptr(const shared_ptr<ExamData>&) override;

    void set_input_data(const shared_ptr<ExamData> &) override;
    const ListModeData& get_input_data() const override;
    //! set maximum segment_number (in listmode data) to process
    /*! minimum will be -max_segment_num_to_process

      Use -1 to process all.
     */
    void set_max_segment_num_to_process(int);
    //! get maximum segment_number (from listmode data) to process
    /*! \see set_max_segment_num_to_process */
    int get_max_segment_num_to_process() const;

    /*! \name caching-related methods
      These functions can be used to cache listmode events into memory, allowing
      parallelised processing.

      Currently, the cached data is written to one or more files (\see get_cache_filename)
      \warning This code is experimental and likely to change in future versions.
      \warning When re-using an existing cache, there is no check if time-frames etc are
      the same as what was used when creating the cache. This is therefore quite risky.
      \warning Cache-files are written in a binary format that likely depends on compiler, endianness etc.
      \todo It should be possible to read only part of the cache in memory.
    */
    //@{
    //! Set the directory where data will be cached
    /*!
      \param cache_path directory-name (defaults to current directory). The directory has to exist.
    */
    virtual void set_cache_path(const std::string& cache_path);
    //! Get the directory where data will be cached
    virtual std::string get_cache_path() const;

    //! Get the filename for a cache file
    /*!
      Filenames are currently "my_CACHE%d.bin" with get_cache_path() prepended.
    */
    virtual std::string get_cache_filename(unsigned int icache) const;

    //! Set if existing cache files should be used or not
    /*! defaults to \c true */
    void set_recompute_cache(bool);
    bool get_recompute_cache() const;

    //! Skip reading of listmode file entirely, only read from cache (broken!)
    /*!
      \warning This currently aborts, as functionality is broken. (We would need to be
      able to read proj_data_info and exam_info).
      \todo replace with reading from a custom-listmode file (although this
      would have to support the additive term).
    */
    void set_skip_lm_input_file(const bool arg);

    //! Set maximum size (in bytes) of cache in memory
    /*! When all events do not fit in the cache, several cache files will be used.

      As multi-threading occurs over cached events, it is likely that better performance
      will be obtained with a large cache size.
    */
    virtual void set_cache_max_size(const unsigned long int arg);

    //! Get maximum size (in bytes) of cache
    virtual unsigned long int get_cache_max_size() const;

    //@}
protected:
  std::string frame_defs_filename;

  //! Filename with input projection data
  std::string list_mode_filename;

  shared_ptr<ProjData> additive_proj_data_sptr;
  //! filename for additive data (only used when parsing)
  std::string additive_projection_data_filename;
  //! If \c true, the additive sinogram will not be read in memory
  bool reduce_memory_usage;

  shared_ptr<BinNormalisation> normalisation_sptr;
 
  //! Listmode pointer
  shared_ptr<ListModeData> list_mode_data_sptr;
 
  unsigned int current_frame_num;

  //! This is part of some functionality I transfer from LmToProjData.
  long int num_events_to_use;
   //! Reconstruct based on time frames
   bool do_time_frame;
 
  //! sets any default values
  /*! Has to be called by set_defaults in the leaf-class */
  void set_defaults() override;
  //! sets keys
  /*! Has to be called by initialise_keymap in the leaf-class */
  void initialise_keymap() override; 

  bool post_processing() override; 

   //! will be called when a new time frame starts
   /*! The frame numbers start from 1. */
   virtual void start_new_time_frame(const unsigned int new_frame_num);

   ParseAndCreateFrom<TargetT, ListModeData> target_parameter_parser;

   //! This is the number of records to be cached. If this parameter is more than zero, then the
   //! flag cache_lm_file will be set to true. The listmode file up to this size will be loaded in
   //! the RAM, alongside with any additive sinograms.
   unsigned long int cache_size;
   //! This flag is true when cache_size is more than zero.
   bool cache_lm_file;
   //! On the first cached run, the cache will be written in the cache_path.
   //! If recompute_cache is set to zero then every consecutive reconstruction will use that cache file.
   //! If you want to create a new, either delete the previous or set this 1. \todo multiple cache files
   //! need to be supported!
   bool recompute_cache;
   //! This flag is set when we don't set an input lm filename and rely only on the cache file.
   bool skip_lm_input_file;
   //! Path to read/write the cached listmode file. \todo add the ability to set a filename.
   std::string cache_path;
   //! The data set has additive corrections
   bool has_add;
  //! ProjDataInfo
  /*! normally a copy of the one from the listmode file, but could be reduced in size */
  shared_ptr<ProjDataInfo> proj_data_info_sptr;
 private:
  //! maximum segment_number (from listmode data) to process
  /*! \see set_max_segment_num_to_process */
  int max_segment_num_to_process;

};

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.inl"

#endif
