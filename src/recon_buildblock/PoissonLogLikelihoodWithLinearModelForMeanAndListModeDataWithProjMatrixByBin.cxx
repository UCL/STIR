/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2014, 2016, 2018, 2022 University College London
    Copyright (C) 2016, University of Hull
    Copyright (C) 2021, University of Pennsylvania
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \brief Implementation of class 
  stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin

  \author Nikos Efthimiou
  \author Kris Thielemans
  \author Sanida Mustafovic
*/

#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin.h" 
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h" 
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/ProjData.h"
#include "stir/listmode/ListRecord.h"
#include "stir/Viewgram.h"
#include "stir/info.h"
#include <boost/format.hpp>
#include "stir/HighResWallClockTimer.h"
#include "stir/Viewgram.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/recon_array_functions.h"
#include "stir/FilePath.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include "stir/stream.h"
#include "stir/listmode/ListModeData_dummy.h"

#include <fstream>
#include <string>

#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"
#include "stir/recon_buildblock/BinNormalisationWithCalibration.h"

#include "stir/recon_buildblock/PresmoothingForwardProjectorByBin.h"
#include "stir/recon_buildblock/PostsmoothingBackProjectorByBin.h"
#ifdef STIR_MPI
#include "stir/recon_buildblock/distributed_functions.h"
#endif

#ifdef STIR_OPENMP
#include <omp.h>
#endif

#include <vector>
START_NAMESPACE_STIR

template<typename TargetT>
const char * const 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
registered_name = 
"PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin";
 
template <typename TargetT> 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>:: 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin() 
{ 
  this->set_defaults(); 
} 

template <typename TargetT> 
void  
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_defaults()
{
  base_type::set_defaults();
#if STIR_VERSION < 060000
  this->max_ring_difference_num_to_process =-1;
#endif
  this->PM_sptr.reset(new  ProjMatrixByBinUsingRayTracing());

  this->use_tofsens = false;
  skip_balanced_subsets = false;
} 
 
template <typename TargetT> 
void  
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
initialise_keymap() 
{ 
  base_type::initialise_keymap(); 
  this->parser.add_start_key("PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin Parameters"); 
  this->parser.add_stop_key("End PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin Parameters");
  this->parser.add_key("use time-of-flight sensitivities", &this->use_tofsens);
#if STIR_VERSION < 060000
this->parser.add_key("max ring difference num to process", &this->max_ring_difference_num_to_process);
#endif
  this->parser.add_parsing_key("Matrix type", &this->PM_sptr); 

  this->parser.add_key("num_events_to_use",&this->num_events_to_use);
  this->parser.add_key("skip checking balanced subsets", &skip_balanced_subsets);
} 

template <typename TargetT> 
int 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_num_subsets(const int new_num_subsets)
{
  this->num_subsets = new_num_subsets;
  return this->num_subsets;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_proj_matrix(const shared_ptr<ProjMatrixByBin>& arg)
{
    this->PM_sptr = arg;
}

#if 0
template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_proj_data_info(const ProjData& arg)
{
  // this will be broken now. Why did we need this?
    this->proj_data_info_sptr = arg.get_proj_data_info_sptr()->create_shared_clone();
    if(this->skip_lm_input_file)
    {
        std::cout << "Dummy LM file" << std::endl;
        this->list_mode_data_sptr.reset(new ListModeData_dummy(
                                            arg.get_exam_info_sptr(),
                                            proj_data_info_sptr));
        this->frame_defs = arg.get_exam_info_sptr()->get_time_frame_definitions();
    }
}
#endif

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_skip_balanced_subsets(const bool arg)
{
  skip_balanced_subsets = arg;
}

#if STIR_VERSION < 060000
template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_max_ring_difference(const int arg)
{
  if (is_null_ptr(this->proj_data_info_sptr))
    error("set_max_ring_difference can only be called after setting the listmode"
          " (but is obsolete anyway. Use set_max_segment_num_to_process() instead).");
  auto pdi_cyl_ptr = dynamic_cast<ProjDataInfoCylindrical const *>(this->proj_data_info_sptr.get());
  if (is_null_ptr(pdi_cyl_ptr))
    error("set_max_ring_difference can only be called for listmode data with cylindrical proj_data_info"
          " (but is obsolete anyway. Use set_max_segment_num_to_process() instead).");
  if (pdi_cyl_ptr->get_max_ring_difference(0) != 0 ||
      pdi_cyl_ptr->get_min_ring_difference(0) != 0)
    error("set_max_ring_difference can only be called for listmode data with span=1"
          " (but is obsolete anyway. Use set_max_segment_num_to_process() instead).");
  this->set_max_segment_num_to_process(arg);
}
#endif

template<typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
actual_subsets_are_approximately_balanced(std::string& warning_message) const
{
  if (this->num_subsets == 1)
    return true;

  if (skip_balanced_subsets)
    {
        warning("We skip the check on balanced subsets and presume they are balanced!");
        return true;
    }

    assert(this->num_subsets>0);
        const DataSymmetriesForBins& symmetries =
                *this->PM_sptr->get_symmetries_ptr();

        Array<1,int> num_bins_in_subset(this->num_subsets);
        num_bins_in_subset.fill(0);

        for (int subset_num=0; subset_num<this->num_subsets; ++subset_num)
        {
          for (int segment_num = this->proj_data_info_sptr->get_min_segment_num();
                 segment_num <= this->proj_data_info_sptr->get_max_segment_num(); ++segment_num)
            {
                for (int axial_num = this->proj_data_info_sptr->get_min_axial_pos_num(segment_num);
                     axial_num < this->proj_data_info_sptr->get_max_axial_pos_num(segment_num);
                     axial_num ++)
                {
                    // For debugging.
                    //                std::cout <<segment_num << " "<<  axial_num  << std::endl;

                    for (int tang_num= this->proj_data_info_sptr->get_min_tangential_pos_num();
                         tang_num < this->proj_data_info_sptr->get_max_tangential_pos_num();
                         tang_num ++ )
                    {
                        for(int view_num = this->proj_data_info_sptr->get_min_view_num() + subset_num;
                            view_num <= this->proj_data_info_sptr->get_max_view_num();
                            view_num += this->num_subsets)
                        {
                          for (int timing_pos_num = this->proj_data_info_sptr->get_min_tof_pos_num();
                               timing_pos_num <= this->proj_data_info_sptr->get_max_tof_pos_num();
                               ++timing_pos_num)
                            {
                              const Bin tmp_bin(segment_num,
                                                view_num,
                                                axial_num,
                                                tang_num,
                                                timing_pos_num,
                                                1);

                              if (!this->PM_sptr->get_symmetries_ptr()->is_basic(tmp_bin) )
                                continue;

                              num_bins_in_subset[subset_num] +=
                                symmetries.num_related_bins(tmp_bin);
                            }
                        }
                    }
                }
            }
        }

        for (int subset_num=1; subset_num<this->num_subsets; ++subset_num)
        {
            if(num_bins_in_subset[subset_num] != num_bins_in_subset[0])
            {
                std::stringstream str(warning_message);
                str <<"Number of subsets is such that subsets will be very unbalanced.\n"
                   << "Number of Bins in each subset would be:\n"
                   << num_bins_in_subset
                   << "\nEither reduce the number of symmetries used by the projector, or\n"
                      "change the number of subsets. It usually should be a divisor of\n"
                   << this->proj_data_info_sptr->get_num_views()
                   << "/4 (or if that's not an integer, a divisor of "
                   << this->proj_data_info_sptr->get_num_views()
                   << "/2 or "
                   << this->proj_data_info_sptr->get_num_views()
                   << ").\n";
                warning_message = str.str();
                return false;
            }
        }
        return true;
}

template <typename TargetT>  
Succeeded 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_up_before_sensitivity(shared_ptr <const TargetT > const& target_sptr) 
{ 
  if ( base_type::set_up_before_sensitivity(target_sptr) != Succeeded::yes)
    return Succeeded::no;
#ifdef STIR_MPI
    //broadcast objective_function (100=PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin)
    distributed::send_int_value(100, -1);
#endif
    
  if (is_null_ptr(this->PM_sptr)) 
    { error("You need to specify a projection matrix"); } 

    // set projector to be used for the calculations
    this->PM_sptr->set_up(this->proj_data_info_sptr->create_shared_clone(),target_sptr);

    shared_ptr<ForwardProjectorByBin> forward_projector_ptr(new ForwardProjectorByBinUsingProjMatrixByBin(this->PM_sptr));
    shared_ptr<BackProjectorByBin> back_projector_ptr(new BackProjectorByBinUsingProjMatrixByBin(this->PM_sptr));

    this->projector_pair_sptr.reset(
                new ProjectorByBinPairUsingSeparateProjectors(forward_projector_ptr, back_projector_ptr));

    this->projector_pair_sptr->set_up(this->proj_data_info_sptr->create_shared_clone(),target_sptr);

    if (!this->use_tofsens && (this->proj_data_info_sptr->get_num_tof_poss()>1)) // TODO this check needs to cover the case if we reconstruct only TOF bin 0
      {
	// sets non-tof backprojector for sensitivity calculation (clone of the back_projector + set projdatainfo to non-tof)
        this->sens_proj_data_info_sptr = this->proj_data_info_sptr->create_non_tof_clone();
        // TODO disable caching of the matrix
        this->sens_backprojector_sptr.reset(projector_pair_sptr->get_back_projector_sptr()->clone());
        this->sens_backprojector_sptr->set_up(this->sens_proj_data_info_sptr, target_sptr);
      }
    else
      {
        // just use the normal backprojector
        this->sens_proj_data_info_sptr = this->proj_data_info_sptr;
        this->sens_backprojector_sptr = projector_pair_sptr->get_back_projector_sptr();
      }
    if (this->normalisation_sptr->set_up(this->list_mode_data_sptr->get_exam_info_sptr(),
                                         this->sens_proj_data_info_sptr) == Succeeded::no)
      {
        warning("PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin: "
                "set-up of normalisation failed.");
        return Succeeded::no;
      }


    if (this->current_frame_num<=0)
    {
        warning("frame_num should be >= 1");
        return Succeeded::no;
    }

    if (this->current_frame_num > this->frame_defs.get_num_frames())
        {
            warning("frame_num is %d, but should be less than the number of frames %d.",
                    this->current_frame_num, this->frame_defs.get_num_frames());
            return Succeeded::no;
        }

    if(this->cache_size > 0 || this->skip_lm_input_file)
    {
        this->cache_lm_file = true;
        return cache_listmode_file();
    }
    else if (this->cache_size == 0 && this->skip_lm_input_file)
    {
        warning("Please set the max cache size for the listmode file");
        this->cache_lm_file = true;
        return cache_listmode_file();
    }

    return Succeeded::yes;
} 
 
 
template <typename TargetT>  
bool  
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::post_processing() 
{

  if (base_type::post_processing() == true)
    return true;
 
#if STIR_VERSION < 060000
   this->proj_data_info_sptr = this->list_mode_data_sptr->get_proj_data_info_sptr()->create_shared_clone();

   if (this->get_max_segment_num_to_process() < 0)
     this->set_max_ring_difference(this->max_ring_difference_num_to_process);
   else
     {
       if (this->max_ring_difference_num_to_process >= 0)
         warning("You've set \"max_ring_difference_num_to_process\", which is obsolete.\n"
                 "Replace by \"maximum segment number to process\" for future compatibility and to avoid this warning");
     }
#endif

   return false;

}

template<typename TargetT>
Succeeded
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::load_listmode_cache_file(unsigned int file_id)
{
    FilePath icache(this->get_cache_filename(file_id), false);

    record_cache.clear();

    if (icache.is_regular_file())
    {
        info( boost::format("Loading Listmode cache from disk %1%") % icache.get_as_string());
        std::ifstream fin(icache.get_as_string(), std::ios::in | std::ios::binary
                          | std::ios::ate);

        const std::size_t num_records = fin.tellg()/sizeof (Bin);
        try
          {
            record_cache.reserve(num_records + 1); // add 1 to avoid reallocation when overruning (see below)
          }
        catch (...)
          {
            error("Listmode: cannot allocate cache for " + std::to_string(num_records + 1) + " records");
          }
        if (!fin)
          error("Error opening cache file \"" + icache.get_as_string() + "\" for reading.");

        fin.clear();
        fin.seekg(0);

        while(!fin.eof())
        {
            BinAndCorr tmp;
            fin.read((char*)&tmp, sizeof(Bin));
            if (this->has_add)
            {
                tmp.my_corr = tmp.my_bin.get_bin_value();
                tmp.my_bin.set_bin_value(1);
            }
            record_cache.push_back(tmp);
        }
        //The while will push one junk record
        record_cache.pop_back();
        fin.close();
    }
    else
    {
        error("Cannot find Listmode cache on disk. Please recompute it or do not set the  max cache size. Abort.");
        return Succeeded::no;
    }

    info( boost::format("Cached Events: %1% ") % record_cache.size());
    return Succeeded::yes;
}

template<typename TargetT>
Succeeded
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::write_listmode_cache_file(unsigned int file_id) const
{
    const auto cache_filename = this->get_cache_filename(file_id);
    const bool with_add = !is_null_ptr(this->additive_proj_data_sptr);

    {
        info("Storing Listmode cache to file \"" + cache_filename + "\".");
        // open the file, overwriting whatever was there before
        std::ofstream fout(cache_filename, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!fout)
          error("Error opening cache file \"" + cache_filename + "\" for writing.");

        for(unsigned long int ie = 0; ie < record_cache.size(); ++ie)
        {
            Bin tmp = record_cache[ie].my_bin;
            if(with_add)
              tmp.set_bin_value(record_cache[ie].my_corr);
            fout.write((char*)&tmp, sizeof(Bin));
        }
        if (!fout)
          error("Error writing to cache file \"" + cache_filename + "\".");

        fout.close();
    }

    return Succeeded::yes;
}

template<typename TargetT>
Succeeded
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::cache_listmode_file()
{
    if(!this->recompute_cache && this->cache_lm_file)
      {
        warning("Looking for existing cache files such as \""
                + this->get_cache_filename(0) + "\".\n"
                + "We will be ignoring any time frame definitions as well as num_events_to_use!");
        // find how many cache files there are
        this->num_cache_files = 0;
        while (true)
          {
            if (!FilePath::exists(this->get_cache_filename(this->num_cache_files)))
                break;
            ++this->num_cache_files;
          }
        if (!this->num_cache_files)
          error("No cache files found.");
        return Succeeded::yes; // Stop here!!!
    }

    if(this->cache_lm_file)
      {
        info("Listmode reconstruction: Creating cache...", 2);

        record_cache.clear();
        try
          {
            record_cache.reserve(this->cache_size);
          }
        catch (...)
          {
            error("Listmode: cannot allocate cache for " + std::to_string(this->cache_size) + " records. Reduce cache size.");
          }

        this->list_mode_data_sptr->reset();
        const shared_ptr<ListRecord>  record_sptr = this->list_mode_data_sptr->get_empty_record_sptr();

        const double start_time = this->frame_defs.get_start_time(this->current_frame_num);
        const double end_time = this->frame_defs.get_end_time(this->current_frame_num);
        double current_time = 0.;
        unsigned long int cached_events = 0;

        bool stop_caching = false;
        record_cache.reserve(this->cache_size);

        while(true) //keep caching across multiple files.
        {
            record_cache.clear();

            while (true) // Start for the current cache
            {
                if(this->list_mode_data_sptr->get_next_record(*record_sptr) == Succeeded::no)
                {
                    stop_caching = true;
                    break;
                }
                if (record_sptr->is_time())
                  {
                    current_time = record_sptr->time().get_time_in_secs();
                    if (this->do_time_frame && current_time >= end_time)
                      {
                        stop_caching = true;
                        break; // get out of while loop
                      }
                }
                if (current_time < start_time)
                  continue;
                if (record_sptr->is_event() && record_sptr->event().is_prompt())
                {
                    BinAndCorr tmp;
                    tmp.my_bin.set_bin_value(1.0);
                    record_sptr->event().get_bin(tmp.my_bin, *this->proj_data_info_sptr);

                    if (tmp.my_bin.get_bin_value() != 1.0f
                            ||  tmp.my_bin.segment_num() < this->proj_data_info_sptr->get_min_segment_num()
                            ||  tmp.my_bin.segment_num()  > this->proj_data_info_sptr->get_max_segment_num()
                            ||  tmp.my_bin.tangential_pos_num() < this->proj_data_info_sptr->get_min_tangential_pos_num()
                            ||  tmp.my_bin.tangential_pos_num() > this->proj_data_info_sptr->get_max_tangential_pos_num()
                            ||  tmp.my_bin.axial_pos_num() < this->proj_data_info_sptr->get_min_axial_pos_num(tmp.my_bin.segment_num())
                            ||  tmp.my_bin.axial_pos_num() > this->proj_data_info_sptr->get_max_axial_pos_num(tmp.my_bin.segment_num())
                            ||  tmp.my_bin.timing_pos_num() < this->proj_data_info_sptr->get_min_tof_pos_num()
                            ||  tmp.my_bin.timing_pos_num() > this->proj_data_info_sptr->get_max_tof_pos_num()
                            )
                    {
                        continue;
                    }
                    try
                      {
                        record_cache.push_back(tmp);
                        ++cached_events;
                      }
                    catch (...)
                      {
                        // should never get here due to `reserve` statement above, but best to check...
                        error("Listmode: running out of memory for cache. Current size: " + std::to_string(this->record_cache.size()) + " records");
                      }


                    if (record_cache.size() > 1 && record_cache.size()%500000L==0)
                      info( boost::format("Cached Prompt Events (this cache): %1% ") % record_cache.size());

                    if(this->num_events_to_use > 0)
                      if (cached_events >= static_cast<std::size_t>(this->num_events_to_use))
                        {
                            stop_caching = true;
                            break;
                        }

                    if (record_cache.size() == this->cache_size)
                      break; // cache is full. go to next cache.
                }
            }

            // add additive term to current cache
            if(this->has_add)
              {
                info( boost::format("Caching Additive corrections for : %1% events.") % record_cache.size());

#ifdef STIR_OPENMP
#pragma omp parallel
                {
#pragma omp single
                  {
                    info("Caching add background with " + std::to_string(omp_get_num_threads()) + " threads", 2);
                  }
                }
#endif

#ifdef STIR_OPENMP
#if _OPENMP <201107
    #pragma omp parallel for schedule(dynamic)
#else
    #pragma omp parallel for collapse(2) schedule(dynamic)
#endif
#endif
                for (int seg = this->additive_proj_data_sptr->get_min_segment_num();
                     seg <= this->additive_proj_data_sptr->get_max_segment_num();
                     ++seg)
                  for (int timing_pos_num = this->additive_proj_data_sptr->get_min_tof_pos_num();
                       timing_pos_num <= this->additive_proj_data_sptr->get_max_tof_pos_num();
                       ++timing_pos_num)
                  {
                    const auto segment(this->additive_proj_data_sptr->get_segment_by_view(seg, timing_pos_num));

                    for (BinAndCorr &cur_bin : record_cache)
                      {
                        if (cur_bin.my_bin.segment_num() == seg)
                          {
#ifdef STIR_OPENMP
# if _OPENMP >=201012
#  pragma omp atomic write
# else
#  pragma omp critical(PLogLikListModePMBAddSinoCaching)
# endif
#endif
                            cur_bin.my_corr = segment[cur_bin.my_bin.view_num()][cur_bin.my_bin.axial_pos_num()][cur_bin.my_bin.tangential_pos_num()];
                          }
                      }
                  }
              } // end additive correction

            if (write_listmode_cache_file(this->num_cache_files) == Succeeded::no)
              {
                error("Error writing cache file!");
              }
            ++this->num_cache_files;

            if(stop_caching)
              break;
      }
      info( boost::format("Cached Events: %1% ") % cached_events);
      return Succeeded::yes;
    }
  return Succeeded::no;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const
{
  // TODO replace with call to distributable function

    const int min_segment_num = this->proj_data_info_sptr->get_min_segment_num();
    const int max_segment_num = this->proj_data_info_sptr->get_max_segment_num();

    info(boost::format("Calculating sensitivity for subset %1%") %subset_num);

    int min_timing_pos_num = use_tofsens ? this->proj_data_info_sptr->get_min_tof_pos_num() : 0;
    int max_timing_pos_num = use_tofsens ? this->proj_data_info_sptr->get_max_tof_pos_num() : 0;
    if (min_timing_pos_num<0 || max_timing_pos_num>0)
      error("TOF code for sensitivity needs work");

    this->sens_backprojector_sptr->
      start_accumulating_in_new_target();

    // warning: has to be same as subset scheme used as in distributable_computation
#ifdef STIR_OPENMP
#if _OPENMP <201107
    #pragma omp parallel for schedule(dynamic)
#else
    #pragma omp parallel for collapse(2) schedule(dynamic)
#endif
#endif
    for (int segment_num = min_segment_num; segment_num <= max_segment_num; ++segment_num)
    {
      for (int view = this->sens_proj_data_info_sptr->get_min_view_num() + subset_num;
          view <= this->sens_proj_data_info_sptr->get_max_view_num();
          view += this->num_subsets)
      {
          const ViewSegmentNumbers view_segment_num(view, segment_num);

          if (! this->sens_backprojector_sptr->get_symmetries_used()->is_basic(view_segment_num))
            continue;
          //for (int timing_pos_num = min_timing_pos_num; timing_pos_num <= max_timing_pos_num; ++timing_pos_num)
          {
              shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_used
              (this->sens_backprojector_sptr->get_symmetries_used()->clone());

              RelatedViewgrams<float> viewgrams =
                  this->sens_proj_data_info_sptr->get_empty_related_viewgrams(
                      view_segment_num, symmetries_used, false);//, timing_pos_num);

              viewgrams.fill(1.F);
              // find efficiencies
              {
                  this->normalisation_sptr->undo(viewgrams);
              }
              // backproject
              {
                  const int min_ax_pos_num =
                      viewgrams.get_min_axial_pos_num();
                  const int max_ax_pos_num =
                      viewgrams.get_max_axial_pos_num();

                  this->sens_backprojector_sptr->
                    back_project(viewgrams,
                                 min_ax_pos_num, max_ax_pos_num);
              }
          }
      }
    }
    this->sens_backprojector_sptr->
      get_output(sensitivity);
}

template<typename TargetT>
 std::unique_ptr<ExamInfo>
 PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
 get_exam_info_uptr_for_target()  const
{
     auto exam_info_uptr = this->get_exam_info_uptr_for_target();
     if (auto norm_ptr = dynamic_cast<BinNormalisationWithCalibration const * const>(get_normalisation_sptr().get()))
     {
       exam_info_uptr->set_calibration_factor(norm_ptr->get_calibration_factor());
       // somehow tell the image that it's calibrated (do we have a way?)
     }
     else
     {
       exam_info_uptr->set_calibration_factor(1.F);
       // somehow tell the image that it's not calibrated (do we have a way?)
     }
    return exam_info_uptr;
}


template <typename TargetT> 
TargetT * 
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>:: 
construct_target_ptr() const 
{ 

 return 
   this->target_parameter_parser.create(this->get_input_data());
} 
 
template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
actual_compute_subset_gradient_without_penalty(TargetT& gradient,
                                               const TargetT &current_estimate,
                                               const int subset_num,
                                               const bool add_sensitivity)
{
    assert(subset_num>=0);
    assert(subset_num<this->num_subsets);
    if (!add_sensitivity && !this->get_use_subset_sensitivities() && this->num_subsets>1)
        error("PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin::"
              "actual_compute_subset_gradient_without_penalty(): cannot subtract subset sensitivity because "
              "use_subset_sensitivities is false. This will result in an error in the gradient computation.");

    if (this->cache_lm_file)
      {
        for (unsigned int icache = 0; icache < this->num_cache_files; ++icache)
          {
            load_listmode_cache_file(icache);
            LM_distributable_computation(this->PM_sptr,
                                         this->proj_data_info_sptr,
                                         &gradient, &current_estimate,
                                         record_cache,
                                         subset_num, this->num_subsets,
                                         this->has_add,
                                         /* accumulate = */ icache != 0);
          }
      }
    else
    {
        //  list_mode_data_sptr->set_get_position(start_time);
        // TODO implement function that will do this for a random time

      this->list_mode_data_sptr->reset();

        const double start_time = this->frame_defs.get_start_time(this->current_frame_num);
        const double end_time = this->frame_defs.get_end_time(this->current_frame_num);

        long num_used_events = 0;
        const float max_quotient = 10000.F;

        double current_time = 0.;

        // need get_bin_value(), so currently need to cast
        shared_ptr<ProjDataFromStream> add;

        if (!is_null_ptr(this->additive_proj_data_sptr))
          {
            add = std::dynamic_pointer_cast<ProjDataFromStream>(this->additive_proj_data_sptr);
            // TODO could create a ProjDataInMemory instead, but for now we give up.
            if (is_null_ptr(add))
              error("Additive projection data is in unsupported file format. You need to create an Interfile copy. sorry.");
          }

        ProjMatrixElemsForOneBin proj_matrix_row;
        gradient.fill(0);
        shared_ptr<ListRecord> record_sptr = this->list_mode_data_sptr->get_empty_record_sptr();
        ListRecord& record = *record_sptr;

        VectorWithOffset<ListModeData::SavedPosition>
                frame_start_positions(1, static_cast<int>(this->frame_defs.get_num_frames()));

        while (true)
          {

           if (this->list_mode_data_sptr->get_next_record(record) == Succeeded::no)
           {
             info("End of listmode file!", 2);
               break; //get out of while loop
           }

           if(record.is_time())
           {
               current_time = record.time().get_time_in_secs();
               if (this->do_time_frame && current_time >= end_time)
               {
                   break; // get out of while loop
               }
           }

           if (current_time < start_time)
             continue;

           if (record.is_event() && record.event().is_prompt())
           {
               Bin measured_bin;
               measured_bin.set_bin_value(1.0f);
               record.event().get_bin(measured_bin, *this->proj_data_info_sptr);

               if (measured_bin.get_bin_value() != 1.0f
                       || measured_bin.segment_num() < this->proj_data_info_sptr->get_min_segment_num()
                       || measured_bin.segment_num()  > this->proj_data_info_sptr->get_max_segment_num()
                       || measured_bin.tangential_pos_num() < this->proj_data_info_sptr->get_min_tangential_pos_num()
                       || measured_bin.tangential_pos_num() > this->proj_data_info_sptr->get_max_tangential_pos_num()
                       || measured_bin.axial_pos_num() < this->proj_data_info_sptr->get_min_axial_pos_num(measured_bin.segment_num())
                       || measured_bin.axial_pos_num() > this->proj_data_info_sptr->get_max_axial_pos_num(measured_bin.segment_num())
                       || measured_bin.timing_pos_num() < this->proj_data_info_sptr->get_min_tof_pos_num()
                       || measured_bin.timing_pos_num() > this->proj_data_info_sptr->get_max_tof_pos_num())
               {
                   continue;
               }

               measured_bin.set_bin_value(1.0f);
               // If more than 1 subsets, check if the current bin belongs to the current.
               bool in_subset = true;
               if (this->num_subsets > 1)
               {
                   Bin basic_bin = measured_bin;
                   this->PM_sptr->get_symmetries_ptr()->find_basic_bin(basic_bin);
                   in_subset = (subset_num == static_cast<int>(basic_bin.view_num() % this->num_subsets));
               }
               if (in_subset)
                 {
                   this->PM_sptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, measured_bin);
                   Bin fwd_bin;
                   fwd_bin.set_bin_value(0.0f);
                   proj_matrix_row.forward_project(fwd_bin,current_estimate);
                   // additive sinogram
                   if (!is_null_ptr(this->additive_proj_data_sptr))
                     {
                       float add_value = add->get_bin_value(measured_bin);
                       float value= fwd_bin.get_bin_value()+add_value;
                       fwd_bin.set_bin_value(value);
                     }

                   if ( measured_bin.get_bin_value() <= max_quotient *fwd_bin.get_bin_value())
                     {
                       const float measured_div_fwd = 1.0f /fwd_bin.get_bin_value();
                       measured_bin.set_bin_value(measured_div_fwd);
                       proj_matrix_row.back_project(gradient, measured_bin);
                     }
                 }

               ++num_used_events;

               if (num_used_events%200000L==0)
                   info( boost::format("Used Events: %1% ") % num_used_events);

               // if we use event-count-based processing, see if we need to stop
               if(this->num_events_to_use > 0)
                 if (num_used_events >= this->num_events_to_use)
                   break;
           }
       }
       info(boost::format("Number of used events (for all subsets): %1%") % num_used_events);

    }

    if (!add_sensitivity)
    {
      // subtract the subset sensitivity
      // compute gradient -= sub_sensitivity
      typename TargetT::full_iterator gradient_iter =
              gradient.begin_all();
      const typename TargetT::full_iterator gradient_end =
              gradient.end_all();
      typename TargetT::const_full_iterator sensitivity_iter =
              this->get_subset_sensitivity(subset_num).begin_all_const();
      while (gradient_iter != gradient_end)
      {
        *gradient_iter -= (*sensitivity_iter);
        ++gradient_iter; ++sensitivity_iter;
      }
    }
}

#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif

template class
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<DiscretisedDensity<3,float> >;


END_NAMESPACE_STIR

