/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2014, 2016, 2018, University College London
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
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjData.h"
#include "stir/listmode/ListRecord.h"
#include "stir/Viewgram.h"
#include "stir/info.h"
#include <boost/format.hpp>
#include "stir/HighResWallClockTimer.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
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
  this->additive_proj_data_sptr.reset();
  this->additive_projection_data_filename ="0"; 
  this->max_ring_difference_num_to_process =-1;
  this->PM_sptr.reset(new  ProjMatrixByBinUsingRayTracing());

  this->normalisation_sptr.reset(new TrivialBinNormalisation);
  this->do_time_frame = false;
  reduce_memory_usage = false;
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
  this->parser.add_key("max ring difference num to process", &this->max_ring_difference_num_to_process);
  this->parser.add_parsing_key("Matrix type", &this->PM_sptr); 
  this->parser.add_key("additive sinogram",&this->additive_projection_data_filename);

  this->parser.add_key("num_events_to_use",&this->num_events_to_use);
  this->parser.add_key("reduce memory usage", &reduce_memory_usage);
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

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_proj_data_info(const ProjData& arg)
{
    this->proj_data_info_sptr = arg.get_proj_data_info_sptr()->create_shared_clone();
    if(this->skip_lm_input_file)
    {
        std::cout << "Dummy LM file" << std::endl;
        this->list_mode_data_sptr.reset(new ListModeData_dummy(
                                            arg.get_exam_info_sptr(),
                                            proj_data_info_sptr));
        this->frame_defs = arg.get_exam_info_sptr()->get_time_frame_definitions();
    }
    warning("The default max ring difference is set based full Scanner geometry");
    max_ring_difference_num_to_process = this->proj_data_info_sptr->get_scanner_sptr()->
            get_num_rings() - 1;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_skip_balanced_subsets(const bool arg)
{
  skip_balanced_subsets = arg;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
set_max_ring_difference(const int arg)
{
    max_ring_difference_num_to_process = arg;
    if (max_ring_difference_num_to_process > proj_data_info_sptr->get_max_segment_num())
    {
        error("In the parameter file, the 'maximum ring difference' is larger than the number of segments"
                "in the listmode file. Abort.");
    }
    else if (max_ring_difference_num_to_process < proj_data_info_sptr->get_max_segment_num())
    {
        proj_data_info_sptr->reduce_segment_range(-max_ring_difference_num_to_process,
                                                      max_ring_difference_num_to_process);
    }
}


template<typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
actual_subsets_are_approximately_balanced(std::string& warning_message) const
{
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
            for (int segment_num = -this->max_ring_difference_num_to_process;
                 segment_num <= this->max_ring_difference_num_to_process; ++segment_num)
            {
                for (int axial_num = proj_data_info_sptr->get_min_axial_pos_num(segment_num);
                     axial_num < proj_data_info_sptr->get_max_axial_pos_num(segment_num);
                     axial_num ++)
                {
                    // For debugging.
                    //                std::cout <<segment_num << " "<<  axial_num  << std::endl;

                    for (int tang_num= proj_data_info_sptr->get_min_tangential_pos_num();
                         tang_num < proj_data_info_sptr->get_max_tangential_pos_num();
                         tang_num ++ )
                    {
                        for(int view_num = proj_data_info_sptr->get_min_view_num() + subset_num;
                            view_num <= proj_data_info_sptr->get_max_view_num();
                            view_num += this->num_subsets)
                        {
                            const Bin tmp_bin(segment_num,
                                              view_num,
                                              axial_num,
                                              tang_num, 1);

                            if (!this->PM_sptr->get_symmetries_ptr()->is_basic(tmp_bin) )
                                continue;

                            num_bins_in_subset[subset_num] +=
                                    symmetries.num_related_bins(tmp_bin);

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
                   << proj_data_info_sptr->get_num_views()
                   << "/4 (or if that's not an integer, a divisor of "
                   << proj_data_info_sptr->get_num_views()
                   << "/2 or "
                   << proj_data_info_sptr->get_num_views()
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
#ifdef STIR_MPI
    //broadcast objective_function (100=PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin)
    distributed::send_int_value(100, -1);
#endif

    // set projector to be used for the calculations
    this->PM_sptr->set_up(proj_data_info_sptr->create_shared_clone(),target_sptr);

    this->projector_pair_sptr.reset(
                new ProjectorByBinPairUsingProjMatrixByBin(this->PM_sptr));
    this->projector_pair_sptr->set_up(proj_data_info_sptr->create_shared_clone(),target_sptr);
    if (is_null_ptr(this->normalisation_sptr))
    {
        warning("Invalid normalisation object");
        return Succeeded::no;
    }

    if (this->normalisation_sptr->set_up(
                this->list_mode_data_sptr->get_exam_info_sptr(), proj_data_info_sptr->create_shared_clone()) == Succeeded::no)
        return Succeeded::no;

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
 
#if 1
  if (is_null_ptr(this->PM_sptr)) 

    { warning("You need to specify a projection matrix"); return true; } 

#else
  if(is_null_ptr(this->projector_pair_sptr->get_forward_projector_sptr()))
    {
      warning("No valid forward projector is defined"); return true;
    }

  if(is_null_ptr(this->projector_pair_sptr->get_back_projector_sptr()))
    {
      warning("No valid back projector is defined"); return true;
    }
#endif
  shared_ptr<Scanner> scanner_sptr(new Scanner(*this->list_mode_data_sptr->get_scanner_ptr()));

  if (this->max_ring_difference_num_to_process == -1)
    {
      this->max_ring_difference_num_to_process =
        scanner_sptr->get_num_rings()-1;
    }

  if (this->additive_projection_data_filename != "0")
    {
      info(boost::format("Reading additive projdata data '%1%'")
           % additive_projection_data_filename  );
      if (!reduce_memory_usage)
      {
          shared_ptr <ProjData> temp_additive_proj_data_sptr =
                  ProjData::read_from_file(this->additive_projection_data_filename);
          this->additive_proj_data_sptr.reset(new ProjDataInMemory(* temp_additive_proj_data_sptr));
      }
      else
          additive_proj_data_sptr =
                  ProjData::read_from_file(this->additive_projection_data_filename);
      this->has_add = true;
    }

   proj_data_info_sptr = this->list_mode_data_sptr->get_proj_data_info_sptr()->create_shared_clone();

   if (max_ring_difference_num_to_process > proj_data_info_sptr->get_max_segment_num())
   {
       warning("In the parameter file, the 'maximum ring difference' is larger than the number of segments"
               "in the listmode file. Abort.");
       return true;
   }
   else if (max_ring_difference_num_to_process < proj_data_info_sptr->get_max_segment_num())
   {
       proj_data_info_sptr->reduce_segment_range(-max_ring_difference_num_to_process,
                                                     max_ring_difference_num_to_process);
   }

   // Daniel: abilitate do_time_frame if there is a fdef file
      if (this->frame_defs_filename.size()!=0)
          {
            this->frame_defs = TimeFrameDefinitions(this->frame_defs_filename);
            this->do_time_frame = true;
   }

  if(!is_null_ptr(this->additive_proj_data_sptr))
      if (*(this->additive_proj_data_sptr->get_proj_data_info_sptr()) != *proj_data_info_sptr)
      {
          const ProjDataInfo& add_proj = *(this->additive_proj_data_sptr->get_proj_data_info_sptr());
          const ProjDataInfo& proj = *this->proj_data_info_sptr;
          bool ok =
                  typeid(add_proj) == typeid(proj) &&
                  *add_proj.get_scanner_ptr()== *(proj.get_scanner_ptr()) &&
                  (add_proj.get_min_view_num()==proj.get_min_view_num()) &&
                  (add_proj.get_max_view_num()==proj.get_max_view_num()) &&
                  (add_proj.get_min_tangential_pos_num() ==proj.get_min_tangential_pos_num())&&
                  (add_proj.get_max_tangential_pos_num() ==proj.get_max_tangential_pos_num()) &&
                  add_proj.get_min_segment_num() <= proj.get_min_segment_num()  &&
                  add_proj.get_max_segment_num() >= proj.get_max_segment_num();

          for (int segment_num=proj.get_min_segment_num();
               ok && segment_num<=proj.get_max_segment_num();
               ++segment_num)
          {
              ok =
                      add_proj.get_min_axial_pos_num(segment_num) <= proj.get_min_axial_pos_num(segment_num) &&
                      add_proj.get_max_axial_pos_num(segment_num) >= proj.get_max_axial_pos_num(segment_num);
          }
          if (!ok)
          {
              warning(boost::format("Incompatible additive projection data:\nAdditive projdata info:\n%s\nEmission projdata info:\n%s\n"
                                    "--- (end of incompatible projection data info)---\n")
                      % add_proj.parameter_info()
                      % proj.parameter_info());
              return true;
          }
          this->has_add = true;
      }

  if( this->normalisation_sptr->set_up(this->list_mode_data_sptr->get_exam_info_sptr(), proj_data_info_sptr)
   == Succeeded::no)
  {
warning("PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin: "
      "set-up of normalisation failed.");
return true;
    }

   return false;

}

template<typename TargetT>
Succeeded
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::cache_listmode_file()
{
    if(!this->recompute_cache && this->cache_lm_file)
    {
        info("Reading cache from disk...");
        std::string curr_dir;
        if (this->cache_path.size() > 0)
            curr_dir = this->cache_path;
        else
            curr_dir = FilePath::get_current_working_directory();
        std::string cache_filename = "my_CACHE00.bin";
        FilePath icache(cache_filename, false);
        icache.prepend_directory_name(curr_dir);

        if (icache.is_regular_file())
        {
            info( boost::format("Loading Listmode cache from disk %1%") % icache.get_as_string());
            std::ifstream fin(icache.get_as_string(), std::ios::in | std::ios::binary
                              | std::ios::ate);

            unsigned long int num_of_records = fin.tellg()/sizeof (Bin);
            record_cache.reserve(num_of_records);

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
            error("Cannot find Listmode cache on disk. Please recompute or comment out the  max cache size. Abort.");
            return Succeeded::no;
        }

        info( boost::format("Cached Events: %1% ") % record_cache.size());
        return Succeeded::yes; // Stop here!!!
    }

    if(this->cache_lm_file)
    {
        info( boost::format("Listmode reconstruction: Serializing inputs ..."));
        info( boost::format("Listmode reconstruction: Creating cache..."));

        record_cache.reserve(this->cache_size);

        this->list_mode_data_sptr->reset();
        const shared_ptr<ListRecord> & record_sptr = this->list_mode_data_sptr->get_empty_record_sptr();
        info(boost::format("Caching... "));


        while (true)
        {
            if(this->list_mode_data_sptr->get_next_record(*record_sptr) == Succeeded::no)
            {
                break;
            }

            if (record_sptr->is_event() && record_sptr->event().is_prompt())
            {
                BinAndCorr tmp;
                tmp.my_bin.set_bin_value(1.0);
                record_sptr->event().get_bin(tmp.my_bin, *proj_data_info_sptr);

                if (tmp.my_bin.get_bin_value() != 1.0f
                        ||  tmp.my_bin.segment_num() < proj_data_info_sptr->get_min_segment_num()
                        ||  tmp.my_bin.segment_num()  > proj_data_info_sptr->get_max_segment_num()
                        ||  tmp.my_bin.tangential_pos_num() < proj_data_info_sptr->get_min_tangential_pos_num()
                        ||  tmp.my_bin.tangential_pos_num() > proj_data_info_sptr->get_max_tangential_pos_num()
                        ||  tmp.my_bin.axial_pos_num() < proj_data_info_sptr->get_min_axial_pos_num(tmp.my_bin.segment_num())
                        ||  tmp.my_bin.axial_pos_num() > proj_data_info_sptr->get_max_axial_pos_num(tmp.my_bin.segment_num())
        #ifdef STIR_TOF
                        ||  tmp.timing_pos_num() < proj_data_info_sptr->get_min_tof_pos_num()
                        ||  tmp.timing_pos_num() > proj_data_info_sptr->get_max_tof_pos_num()
        #endif
                        )
                {
                    continue;
                }
                record_cache.push_back(tmp);

                if (record_cache.size() > 1 && record_cache.size()%500000L==0)
                    info( boost::format("Cached Prompt Events: %1% ") % record_cache.size());

                if(this->num_events_to_use > 0)
                    if (record_cache.size() >= this->num_events_to_use)
                        break;
            }

        }


        if(this->has_add)
        {
  #ifdef STIR_TOF
         // TODO
         error("listmode processing with caching is not yet supported for TOF");
  #else
            info( boost::format("Caching Additive corrections for : %1% events.") % record_cache.size());
            const int num_segments_in_memory = 1;

            ProjDataFromStream* add = dynamic_cast<ProjDataFromStream*>(additive_proj_data_sptr.get());
            if (is_null_ptr(add))
               error("Additive projection data is in unsupported file format for the caching. You need to create an Interfile copy. sorry.");

            int num_threads = 1;
#ifdef STIR_OPENMP
#pragma omp parallel
            {
#pragma omp single
                {

                    num_threads = omp_get_num_threads();
                    info("Caching add background with " + std::to_string(omp_get_num_threads()) + " threads");
                }
            }
#endif

#ifdef STIR_OPENMP
#pragma omp parallel for schedule(dynamic) //collapse(2)
#endif
             for (int start_segment_index = additive_proj_data_sptr->get_min_segment_num();
                  start_segment_index <= additive_proj_data_sptr->get_max_segment_num();
                  start_segment_index += num_segments_in_memory)
             {
#ifdef STIR_OPENMP
                 const int thread_num = omp_get_thread_num();
#else
                 const int thread_num = 0;
#endif

                 const int end_segment_index =
                         std::min( additive_proj_data_sptr->get_max_segment_num()+1, start_segment_index + num_segments_in_memory) - 1;

                 info( boost::format("Current start / end segments: %1% / %2%") % start_segment_index % end_segment_index);
                 VectorWithOffset<SegmentByView<float> *>
                         segments (start_segment_index, end_segment_index);

                 for (int seg=start_segment_index ; seg<=end_segment_index; seg++)
                 {
                     segments[seg] = new SegmentByView<float>(additive_proj_data_sptr->get_segment_by_view(seg));
                 }

                 for (BinAndCorr &cur_bin : record_cache)
                 {

                     if (cur_bin.my_bin.segment_num() < start_segment_index
                             ||  cur_bin.my_bin.segment_num()  > end_segment_index)
                     {
                         continue;
                     }

                     cur_bin.my_corr = (*segments[cur_bin.my_bin.segment_num()])
                             [cur_bin.my_bin.view_num()][cur_bin.my_bin.axial_pos_num()][cur_bin.my_bin.tangential_pos_num()];
                 }
             }
  #endif
        }
        info( boost::format("Cached Events: %1% ") % record_cache.size());

        if(this->recompute_cache)
        {
            info( boost::format("Storing Cached Events ... "));

            std::string curr_dir = FilePath::get_current_working_directory();
            std::string cache_filename = "my_CACHE00.bin";
            FilePath ocache(cache_filename, false);
            ocache.prepend_directory_name(curr_dir);

            bool with_add = !is_null_ptr(additive_proj_data_sptr);

  //          if (ocache.is_regular_file())
            {
                info( boost::format("Storing Listmode cache from disk %1%") % ocache.get_as_string());
                std::ofstream fin(ocache.get_as_string(), std::ios::out | std::ios::binary);

                //fout.write((char*)&student[0], student.size() * sizeof(Student));
                for(unsigned long int ie = 0; ie < record_cache.size(); ++ie)
                {
                    Bin tmp = record_cache.at(ie).my_bin;
                    if(with_add)
                      tmp.set_bin_value(record_cache.at(ie).my_corr);
                    fin.write((char*)&tmp, sizeof(Bin));
                }
                fin.close();
            }
  //          else
  //          {
  //              error("File Path for storing the cache is not writable! Abort.");
  //              return true;
  //          }


            return Succeeded::yes; // Stop here!!!
        }
    }
    return Succeeded::no;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const
{

    const int min_segment_num = proj_data_info_sptr->get_min_segment_num();
    const int max_segment_num = proj_data_info_sptr->get_max_segment_num();

    info(boost::format("Calculating sensitivity for subset %1%") %subset_num);

#ifdef STIR_TOF
    int min_timing_pos_num = use_tofsens ? this->proj_data_info_sptr->get_min_tof_pos_num() : 0;
    int max_timing_pos_num = use_tofsens ? this->proj_data_info_sptr->get_max_tof_pos_num() : 0;
#endif

    this->projector_pair_sptr->get_back_projector_sptr()->
      start_accumulating_in_new_target();

    // warning: has to be same as subset scheme used as in distributable_computation
#ifdef STIR_OPENMP
#ifdef _WIN32
    #pragma omp parallel for
#else
    #pragma omp parallel for collapse(2) schedule(dynamic)
#endif
#endif
    for (int segment_num = min_segment_num; segment_num <= max_segment_num; ++segment_num)
    {
      for (int view = proj_data_info_sptr->get_min_view_num() + subset_num;
          view <= proj_data_info_sptr->get_max_view_num();
          view += this->num_subsets)
      {
          const ViewSegmentNumbers view_segment_num(view, segment_num);

          if (! this->projector_pair_sptr->get_symmetries_used()->is_basic(view_segment_num))
            continue;
          //        this->add_view_seg_to_sensitivity(view_segment_num);
#ifdef STIR_OPENMP
          const int thread_num=omp_get_thread_num();
#else
          const int thread_num = 0;
#endif
          //for (int timing_pos_num = min_timing_pos_num; timing_pos_num <= max_timing_pos_num; ++timing_pos_num)
          {
              shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_used
              (this->projector_pair_sptr->get_symmetries_used()->clone());

              RelatedViewgrams<float> viewgrams =
                  proj_data_info_sptr->get_empty_related_viewgrams(
                      view_segment_num, symmetries_used, false);//, timing_pos_num);

              viewgrams.fill(1.F);
              // find efficiencies
              {
                  //const double start_frame = this->frame_defs.get_start_time(this->current_frame_num);
                  //const double end_frame = this->frame_defs.get_end_time(this->current_frame_num);
                  this->normalisation_sptr->undo(viewgrams);//, &start_frame, end_frame);
              }
              // backproject
              {
                  const int min_ax_pos_num =
                      viewgrams.get_min_axial_pos_num();
                  const int max_ax_pos_num =
                      viewgrams.get_max_axial_pos_num();

                  this->projector_pair_sptr->get_back_projector_sptr()->
                    back_project(viewgrams,
                                 min_ax_pos_num, max_ax_pos_num);
              }
          }
      }
    }
    this->projector_pair_sptr->get_back_projector_sptr()->
      get_output(sensitivity);
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>::
add_view_seg_to_sensitivity(const ViewSegmentNumbers& view_seg_nums) const
{
//    shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_used
//            (this->projector_pair_sptr->get_symmetries_used()->clone());

//  RelatedViewgrams<float> viewgrams =
//    proj_data_info_sptr->get_empty_related_viewgrams(view_seg_nums,symmetries_used);

//  viewgrams.fill(1.F);
//  // find efficiencies
//  {
//    const double start_frame = this->frame_defs.get_start_time(this->current_frame_num);
//    const double end_frame = this->frame_defs.get_end_time(this->current_frame_num);
//    this->normalisation_sptr->undo(viewgrams);
//  }
//  // backproject
//  {
//    const int min_ax_pos_num =
//      viewgrams.get_min_axial_pos_num();
//    const int max_ax_pos_num =
//       viewgrams.get_max_axial_pos_num();

//    this->projector_pair_sptr->get_back_projector_sptr()->
//      back_project(viewgrams,
//                   min_ax_pos_num, max_ax_pos_num);
//  }

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
    if (!add_sensitivity && !this->get_use_subset_sensitivities())
        error("PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin::"
              "actual_compute_subset_gradient_without_penalty(): cannot subtract subset sensitivity because "
              "use_subset_sensitivities is false. This will result in an error in the gradient computation.");

    if (this->cache_lm_file)
    {
        if (record_cache.size() > 0)
        {
            LM_distributable_computation(this->PM_sptr,
                                         proj_data_info_sptr,
                                         &gradient, &current_estimate,
                                         record_cache,
                                         subset_num, this->num_subsets,
                                         this->has_add);
        }
    }
    else
    {
        //go to the beginning of this frame
        //  list_mode_data_sptr->set_get_position(start_time);
        // TODO implement function that will do this for a random time
        this->list_mode_data_sptr->reset();

        const double start_time = this->frame_defs.get_start_time(this->current_frame_num);
        const double end_time = this->frame_defs.get_end_time(this->current_frame_num);

        long num_used_events = 0;
        const float max_quotient = 10000.F;

        double current_time = 0.;

        shared_ptr<ProjDataFromStream> add;

        if (!is_null_ptr(this->additive_proj_data_sptr))
            add = std::dynamic_pointer_cast<ProjDataFromStream>(additive_proj_data_sptr);

        ProjMatrixElemsForOneBin proj_matrix_row;
        gradient.fill(0);
        shared_ptr<ListRecord> record_sptr = this->list_mode_data_sptr->get_empty_record_sptr();
        ListRecord& record = *record_sptr;

        VectorWithOffset<ListModeData::SavedPosition>
                frame_start_positions(1, static_cast<int>(this->frame_defs.get_num_frames()));

        long int more_events =
                this->do_time_frame? 1 : this->num_events_to_use;

        while (more_events)
       {

           if (this->list_mode_data_sptr->get_next_record(record) == Succeeded::no)
           {
               info("End of file!");
               break; //get out of while loop
           }

           if(record.is_time() && end_time > 0.01)
           {
               current_time = record.time().get_time_in_secs();
               if (this->do_time_frame && current_time >= end_time)
                   break; // get out of while loop
               if (current_time < start_time)
                   continue;
           }

           if (record.is_event() && record.event().is_prompt())
           {
               Bin measured_bin;
               measured_bin.set_bin_value(1.0f);
               record.event().get_bin(measured_bin, *proj_data_info_sptr);

               if (measured_bin.get_bin_value() != 1.0f
                       || measured_bin.segment_num() < proj_data_info_sptr->get_min_segment_num()
                       || measured_bin.segment_num()  > proj_data_info_sptr->get_max_segment_num()
                       || measured_bin.tangential_pos_num() < proj_data_info_sptr->get_min_tangential_pos_num()
                       || measured_bin.tangential_pos_num() > proj_data_info_sptr->get_max_tangential_pos_num()
                       || measured_bin.axial_pos_num() < proj_data_info_sptr->get_min_axial_pos_num(measured_bin.segment_num())
                       || measured_bin.axial_pos_num() > proj_data_info_sptr->get_max_axial_pos_num(measured_bin.segment_num()))
               {
                   continue;
               }

               measured_bin.set_bin_value(1.0f);
               // If more than 1 subsets, check if the current bin belongs to
               // the current.
               if (this->num_subsets > 1)
               {
                   Bin basic_bin = measured_bin;
                   this->PM_sptr->get_symmetries_ptr()->find_basic_bin(basic_bin);
                   if (subset_num != static_cast<int>(basic_bin.view_num() % this->num_subsets))
                       continue;
               }
               this->PM_sptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, measured_bin);
               //in_the_range++;
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
               float  measured_div_fwd = 0.0f;

               if(!this->do_time_frame)
                   more_events -=1 ;

               num_used_events += 1;

               if (num_used_events%200000L==0)
                   info( boost::format("Stored Events: %1% ") % num_used_events);

               if ( measured_bin.get_bin_value() <= max_quotient *fwd_bin.get_bin_value())
                   measured_div_fwd = 1.0f /fwd_bin.get_bin_value();
               else
                   continue;

               measured_bin.set_bin_value(measured_div_fwd);
               proj_matrix_row.back_project(gradient, measured_bin);

           }
       }
       info(boost::format("Number of used events: %1%") % num_used_events);

    }
    info(boost::format("Finished!"));

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

