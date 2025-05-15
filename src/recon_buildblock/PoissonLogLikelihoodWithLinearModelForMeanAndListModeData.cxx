//
//
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
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
  \author Nikos Efthimiou
  \author Sanida Mustafovic

*/

#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeData.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Succeeded.h"
#include "stir/IO/read_from_file.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/is_null_ptr.h"
#include "stir/FilePath.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/format.h"

using std::vector;
using std::pair;

START_NAMESPACE_STIR

template <typename TargetT>
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::PoissonLogLikelihoodWithLinearModelForMeanAndListModeData()
{
  this->set_defaults();
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::set_defaults()
{
  base_type::set_defaults();
  this->list_mode_filename = "";
  this->frame_defs_filename = "";
  this->frame_defs = TimeFrameDefinitions();
  this->list_mode_data_sptr.reset();
  this->additive_projection_data_filename = "0";
  this->additive_proj_data_sptr.reset();
  this->has_add = false;
  this->reduce_memory_usage = false;
  this->normalisation_sptr.reset(new TrivialBinNormalisation);
  this->current_frame_num = 1;
  this->num_events_to_use = 0L;
  this->max_segment_num_to_process = -1;

  this->target_parameter_parser.set_defaults();
  cache_lm_file = false;
  recompute_cache = true;
  skip_lm_input_file = false;
  cache_path = "";
  cache_size = 0;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_key("list mode filename", &this->list_mode_filename);
  this->target_parameter_parser.add_to_keymap(this->parser);
  this->parser.add_key("time frame definition filename", &this->frame_defs_filename);
  this->parser.add_key("time frame number", &this->current_frame_num);
  this->parser.add_key("maximum absolute segment number to process", &this->max_segment_num_to_process);
  this->parser.add_key("additive sinogram", &this->additive_projection_data_filename);
  this->parser.add_key("reduce memory usage", &reduce_memory_usage);
  this->parser.add_parsing_key("Bin Normalisation type", &this->normalisation_sptr);
  this->parser.add_key("cache path", &cache_path);
  this->parser.add_key("max cache size", &cache_size);
  this->parser.add_key("recompute cache", &recompute_cache);
}

template <typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::post_processing()
{
  if (base_type::post_processing() == true)
    return true;

  if (this->list_mode_filename.length() == 0 && !this->skip_lm_input_file)
    {
      warning("You need to specify an input file\n");
      return true;
    }
  if (!this->skip_lm_input_file)
    {
      this->list_mode_data_sptr = read_from_file<ListModeData>(this->list_mode_filename);
    }

  if (this->additive_projection_data_filename != "0")
    {
      info(format("Reading additive projdata data '{}'", additive_projection_data_filename));
      this->set_additive_proj_data_sptr(ProjData::read_from_file(this->additive_projection_data_filename));
    }

  if (this->frame_defs_filename.size() != 0)
    {
      this->frame_defs = TimeFrameDefinitions(this->frame_defs_filename);
      this->do_time_frame = true;
    }
  else
    {
      this->frame_defs = TimeFrameDefinitions();
    }
  target_parameter_parser.check_values();

  this->already_set_up = false;
  return false;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::set_input_data(const shared_ptr<ExamData>& arg)
{
  this->already_set_up = false;
  try
    {
      this->list_mode_data_sptr = dynamic_pointer_cast<ListModeData>(arg);
    }
  catch (...)
    {
      error("input data doesn't seem to be listmode");
    }
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::set_max_segment_num_to_process(const int arg)
{
  this->already_set_up = this->already_set_up && (this->max_segment_num_to_process == arg);
  this->max_segment_num_to_process = arg;
}

template <typename TargetT>
int
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::get_max_segment_num_to_process() const
{
  return this->max_segment_num_to_process;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::set_recompute_cache(bool v)
{
  this->recompute_cache = v;
}

template <typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::get_recompute_cache() const
{
  return this->recompute_cache;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::set_cache_max_size(const unsigned long int arg)
{
  cache_size = arg;
}

template <typename TargetT>
unsigned long int
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::get_cache_max_size() const
{
  return cache_size;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::set_cache_path(const std::string& cache_path_v)
{
  cache_path = cache_path_v;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::set_skip_lm_input_file(const bool arg)
{
  error("set_skip_lm_input_file is not yet supported.");
  if (arg && (cache_path.length() > 0))
    {
      skip_lm_input_file = arg;

      std::cout << "PoissonLogLikelihoodWithLinearModelForMeanAndListModeData: Skipping input!" << std::endl;
      //!\todo in the future the following statements should be removed.
      {
        this->set_recompute_sensitivity(!arg);
        this->set_use_subset_sensitivities(arg);
        this->set_subsensitivity_filenames(cache_path + "sens_%d.hv");
      }
      //    info(format("Reading sensitivity from '{}'", this->get_subsensitivity_filenames()));
    }
  else
    error("set_skip_lm_input_file(): First set the cache path!");
}

template <typename TargetT>
std::string
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::get_cache_path() const
{
  if (this->cache_path.size() > 0)
    return this->cache_path;
  else
    return FilePath::get_current_working_directory();
}

template <typename TargetT>
std::string
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::get_cache_filename(unsigned int file_id) const
{
  std::string cache_filename = "my_CACHE" + std::to_string(file_id) + ".bin";
  FilePath icache(cache_filename, false);
  icache.prepend_directory_name(this->get_cache_path());
  return icache.get_as_string();
}

template <typename TargetT>
const ListModeData&
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::get_input_data() const
{
  if (is_null_ptr(this->list_mode_data_sptr))
    error("get_input_data(): no list mode data set");
  return *this->list_mode_data_sptr;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::set_additive_proj_data_sptr(const shared_ptr<ExamData>& arg)
{
  this->already_set_up = false;
  try
    {
      this->additive_proj_data_sptr = dynamic_pointer_cast<ProjData>(arg);
    }
  catch (...)
    {
      error("set_additive_proj_data_sptr: argument is wrong type. Should be projection data.");
    }
  if (!this->reduce_memory_usage && is_null_ptr(dynamic_cast<ProjDataInMemory const*>(this->additive_proj_data_sptr.get()))
      && !this->cache_lm_file)
    {
      this->additive_proj_data_sptr.reset(new ProjDataInMemory(*additive_proj_data_sptr));
    }
  this->has_add = true;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::set_normalisation_sptr(
    const shared_ptr<BinNormalisation>& arg)
{
  this->already_set_up = false;
  this->normalisation_sptr = arg;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::start_new_time_frame(const unsigned int)
{}

template <typename TargetT>
Succeeded
PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>::set_up_before_sensitivity(
    shared_ptr<const TargetT> const& target_sptr)
{
  // if ( base_type::set_up_before_sensitivity(target_sptr) != Succeeded::yes)
  //   return Succeeded::no;

  if (is_null_ptr(this->list_mode_data_sptr))
    error("No listmode data set");

  this->proj_data_info_sptr = this->list_mode_data_sptr->get_proj_data_info_sptr()->create_shared_clone();

  if (this->max_segment_num_to_process > proj_data_info_sptr->get_max_segment_num())
    {
      error("The 'maximum segment number to process' asked for is larger than the number of segments"
            "in the listmode file. Abort.");
    }
  else if (this->max_segment_num_to_process >= 0 && max_segment_num_to_process < proj_data_info_sptr->get_max_segment_num())
    {
      this->proj_data_info_sptr->reduce_segment_range(-max_segment_num_to_process, max_segment_num_to_process);
    }

  if (this->frame_defs.get_num_frames())
    {
      // check if we need to handle time frame definitions
      this->do_time_frame = (this->num_events_to_use == 0)
                            && (this->frame_defs.get_start_time(this->current_frame_num)
                                < this->frame_defs.get_end_time(this->current_frame_num));
    }
  else
    {
      // make a single frame starting from 0. End value will be ignored.
      vector<pair<double, double>> frame_times(1, pair<double, double>(0, 0));
      this->frame_defs = TimeFrameDefinitions(frame_times);
      this->do_time_frame = false;
    }

  if (!is_null_ptr(this->additive_proj_data_sptr))
    {
      if (*(this->additive_proj_data_sptr->get_proj_data_info_sptr()) != *proj_data_info_sptr)
        {
          const ProjDataInfo& add_proj = *(this->additive_proj_data_sptr->get_proj_data_info_sptr());
          const ProjDataInfo& proj = *this->proj_data_info_sptr;
          bool ok = typeid(add_proj) == typeid(proj) && *add_proj.get_scanner_ptr() == *(proj.get_scanner_ptr())
                    && (add_proj.get_min_view_num() == proj.get_min_view_num())
                    && (add_proj.get_max_view_num() == proj.get_max_view_num())
                    && (add_proj.get_min_tangential_pos_num() == proj.get_min_tangential_pos_num())
                    && (add_proj.get_max_tangential_pos_num() == proj.get_max_tangential_pos_num())
                    && (add_proj.get_min_tof_pos_num() == proj.get_min_tof_pos_num())
                    && (add_proj.get_max_tof_pos_num() == proj.get_max_tof_pos_num())
                    && add_proj.get_min_segment_num() <= proj.get_min_segment_num()
                    && add_proj.get_max_segment_num() >= proj.get_max_segment_num();

          for (int segment_num = proj.get_min_segment_num(); ok && segment_num <= proj.get_max_segment_num(); ++segment_num)
            {
              ok = add_proj.get_min_axial_pos_num(segment_num) <= proj.get_min_axial_pos_num(segment_num)
                   && add_proj.get_max_axial_pos_num(segment_num) >= proj.get_max_axial_pos_num(segment_num);
            }
          if (!ok)
            {
              error(format("Incompatible additive projection data:\nAdditive projdata info:\n{}\nEmission projdata info:\n{}\n"
                           "--- (end of incompatible projection data info)---\n",
                           add_proj.parameter_info(),
                           proj.parameter_info()));
            }
        }
    }

  if (is_null_ptr(this->normalisation_sptr))
    {
      warning("Invalid normalisation object");
      return Succeeded::no;
    }

  return Succeeded::yes;
}

#ifdef _MSC_VER
// prevent warning message on instantiation of abstract class
#  pragma warning(disable : 4661)
#endif

template class PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<DiscretisedDensity<3, float>>;

END_NAMESPACE_STIR
