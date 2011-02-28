//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

#ifndef __stir_recon_buildblock_DISTRIBUTABLE_H__
#define __stir_recon_buildblock_DISTRIBUTABLE_H__

/*!
  \file
  \ingroup distributable

  \brief Declaration of the main functions that perform parallel processing

  \author Alexey Zverovich
  \author Kris Thielemans
  \author Matthew Jacobson
  \author PARAPET project

   $Date$
   $Revision$
*/
#include "stir/common.h"

START_NAMESPACE_STIR

template <typename T> class shared_ptr;
template <typename elemT> class RelatedViewgrams;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
class BinNormalisation;
class ProjData;
class ProjDataInfo;
class DataSymmetriesForViewSegmentNumbers;
class ForwardProjectorByBin;
class BackProjectorByBin;
class ProjectorByBinPair;
class DistributedCachingInformation;
//!@{
//! \ingroup distributable

//! \name Task-ids current understood by STIR_MPI
//@{
const int task_stop_processing=0;
const int task_setup_distributable_computation=200;
const int task_do_distributable_gradient_computation=42;
//@}

//! set-up parameters before calling distributable_computation()
/*! Empty unless STIR_MPI is defined, in which case it sends parameters to the 
    slaves (see stir::DistributedWorker).

    \todo currently uses some global variables for configuration in the distributed
    namespace. This needs to be converted to a class, e.g. \c DistributedMaster
*/
void setup_distributable_computation(
                                     const shared_ptr<ProjectorByBinPair>& proj_pair_sptr,
                                     const ProjDataInfo * const proj_data_info_ptr,
                                     const shared_ptr<DiscretisedDensity<3,float> >& target_sptr,
                                     const bool zero_seg0_end_planes,
                                     const bool distributed_cache_enabled);

//! clean-up after a sequence of computations
/*!  Empty unless STIR_MPI is defined, in which case it sends the "stop" task to 
     the slaves (see stir::DistributedWorker)
*/
void end_distributable_computation();

//! typedef for callback functions for distributable_computation()
/*! Pointers will be NULL when they are not to be used by the callback function.

    \a count and \a count2 are normally incremental counters that accumulate over the loop
    in distributable_computation().

    \warning The data in *measured_viewgrams_ptr are allowed to be overwritten, but the new data 
    will not be used. 
*/
typedef  void RPC_process_related_viewgrams_type (
                                                  const shared_ptr<ForwardProjectorByBin>& forward_projector_sptr,
                                                  const shared_ptr<BackProjectorByBin>& back_projector_sptr,
                                                  DiscretisedDensity<3,float>* output_image_ptr, 
                                                  const DiscretisedDensity<3,float>* input_image_ptr, 
                                                  RelatedViewgrams<float>* measured_viewgrams_ptr,
                                                  int& count, int& count2, double* log_likelihood_ptr,
                                                  const RelatedViewgrams<float>* additive_binwise_correction_ptr,
                                                  const RelatedViewgrams<float>* mult_viewgrams_ptr);

/*!
  \brief This function essentially implements a loop over segments and all views in the current subset.
  
  Output is in output_image_ptr and in float_out_ptr (but only if they are not NULL).
  What the output is, is really determined by the call-back function
  RPC_process_related_viewgrams.

  If STIR_MPI is defined, this function distributes the computation over the slaves.

  Subsets are currently defined on views. A particular \a subset_num contains all views  which are symmetry related to 
  \code 
  proj_data_ptr->min_view_num()+subset_num + n*num_subsets
  \endcode
  for n=0,1,,.. \c and for which the above view_num is 'basic' (for some segment_num in the range).

  Symmetries are determined by using the 3rd argument to set_projectors_and_symmetries().

  \par Usage

  You first need to call setup_distributable_computation(), then you can do multiple calls
  to distributable_computation() with different images (but the same projection data, as
  this is potentially cached). If you want to change the image characteristics (e.g. 
  size, or origin so), you have to call setup_distributable_computation() again. Finally,
  end the sequence of computations by a call to end_distributable_computation().

  \param output_image_ptr will store the output image if non-zero.
  \param input_image_ptr input when non-zero.
  \param proj_data_ptr input projection data
  \param read_from_proj_data if true, the \a measured_viewgrams_ptr argument of the call_back function 
         will be constructed using ProjData::get_related_viewgrams, otherwise 
         ProjData::get_empty_related_viewgrams is used.
  \param subset_num the number of the current subset (see above). Should be between 0 and num_subsets-1.
  \param num_subsets the number of subsets to consider. 1 will process all data.
  \param min_segment_num Minimum segment_num to process.
  \param max_segment_num Maximum segment_num to process.
  \param zero_seg0_end_planes if true, the end planes for segment_num=0 in measured_viewgrams_ptr
         (and additive_binwise_correction_ptr when applicable) will be set to 0.
  \param double_out_ptr a potential double output parameter for the call back function (which needs to be accumulated).
  \param additive_binwise_correction Additional input projection data (when the shared_ptr is not 0).
  \param normalise_sptr normalisation pointer that, if non-zero, will be used to construct the "multiplicative" viewgrams
         (by using normalise_sptr->undo() on viewgrams filled with 1) that are then passed to RPC_process_related_viewgrams.
         This is useful for e.g. log-likelihood computations.
  \param start_time_of_frame is passed to normalise_sptr
  \param end_time_of_frame is passed to normalise_sptr
  \param RPC_process_related_viewgrams function that does the actual work.
  \param caching_info_ptr ignored unless STIR_MPI=1, in which case it enables caching of viewgrams at the slave side  
  \warning There is NO check that the resulting subsets are balanced.

  \warning The function assumes that \a min_segment_num, \a max_segment_num are such that
  symmetries map this range onto itself (i.e. no segment_num is obtained outside the range). 
  This usually means that \a min_segment_num = -\a max_segment_num. This assumption is checked with 
  assert().

  \todo The subset-scheme should be moved somewhere else (a Subset class?).

  \warning If STIR_MPI is defined, there can only be one set_up active, as the 
  slaves use only one set of variabiles to store projectors etc.

  \see DistributedWorker for how the slaves perform the computation if STIR_MPI is defined.
 */
void distributable_computation(
                               const shared_ptr<ForwardProjectorByBin>& forward_projector_sptr,
                               const shared_ptr<BackProjectorByBin>& back_projector_sptr,
                               const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_sptr,
                               DiscretisedDensity<3,float>* output_image_ptr,
                               const DiscretisedDensity<3,float>* input_image_ptr,
                               const shared_ptr<ProjData>& proj_data_ptr,
                               const bool read_from_proj_data,
                               int subset_num, int num_subsets,
                               int min_segment_num, int max_segment_num,
                               bool zero_seg0_end_planes,
                               double* double_out_ptr,
                               const shared_ptr<ProjData>& additive_binwise_correction,
                               const shared_ptr<BinNormalisation> normalise_sptr,
                               const double start_time_of_frame,
                               const double end_time_of_frame,
                               RPC_process_related_viewgrams_type * RPC_process_related_viewgrams,
                               DistributedCachingInformation* caching_info_ptr);


  /*! \name Tag-names currently used by stir::distributable_computation and related functions
   */
  //!@{
  const int AVAILABLE_NOTIFICATION_TAG=2;
  const int END_ITERATION_TAG=3;
  const int END_RECONSTRUCTION_TAG=4;
  const int END_NOTIFICATION_TAG=5;
  const int BINWISE_CORRECTION_TAG=6;
  const int BINWISE_MULT_TAG=66;
  const int REUSE_VIEWGRAM_TAG=10;
  const int NEW_VIEWGRAM_TAG=11;
  //!@}

//!@}

END_NAMESPACE_STIR

#endif // __stir_recon_buildblock_DISTRIBUTABLE_H__

