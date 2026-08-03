/*
    Copyright 2025, University Medical Center Groningen
    Copyright 2025, MGH / HST A. Martinos Center for Biomedical Imaging

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for detail
*/

#include "stir/format.h"
#include "stir/info.h"
#include "stir/error.h"

#include "stir/listmode/CListModeDataPETSIRD.h"

START_NAMESPACE_STIR
/*!
  \file
  \ingroup listmode
  \brief implementation of class stir::CListModeDataPETSIRD

  \author Daniel Deidda
  \author Nikos Efthimiou
*/

CListModeDataPETSIRD::CListModeDataPETSIRD(const std::string& listmode_filename, bool use_hdf5)
    : use_hdf5(use_hdf5)
{
  CListModeDataBasedOnCoordinateMap::listmode_filename = listmode_filename;

  petsird::Header header;
  if (reset() == Succeeded::no)
    error("CListModeDataPETSIRD: Could not open/prime the reader. Abort.");

  current_lm_data_ptr->ReadHeader(header);
  petsird_info_sptr = std::make_shared<PETSIRDInfo>(header);
  auto stir_scanner_sptr = petsird_info_sptr->get_scanner_sptr();

  int tof_mash_factor = 1;
  this->set_proj_data_info_sptr(std::dynamic_pointer_cast<const ProjDataInfo>(
      ProjDataInfo::construct_proj_data_info(petsird_info_sptr->get_scanner_sptr(),
                                             1,
                                             petsird_info_sptr->get_scanner_sptr()->get_num_rings() - 1,
                                             petsird_info_sptr->get_scanner_sptr()->get_num_detectors_per_ring() / 2,
                                             petsird_info_sptr->get_scanner_sptr()->get_max_num_non_arccorrected_bins(),
                                             /* arc_correction*/ false,
                                             tof_mash_factor)
          ->create_shared_clone()));

  shared_ptr<ExamInfo> _exam_info_sptr(new ExamInfo);
  // Only PET scanners supported
  _exam_info_sptr->imaging_modality = ImagingModality::PT;
  _exam_info_sptr->originating_system = std::string("PETSIRD_defined_scanner");
  _exam_info_sptr->set_low_energy_thres(petsird_info_sptr->get_lower_energy_threshold());
  _exam_info_sptr->set_high_energy_thres(petsird_info_sptr->get_upper_energy_threshold());

  this->exam_info_sptr = _exam_info_sptr;
}

Succeeded
CListModeDataPETSIRD::open_lm_file() const
{
  // current_lm_data_ptr.reset(new petsird::hdf5::PETSIRDReader(listmode_filename));
  // if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
  //  return Succeeded::no;

  curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
  return Succeeded::yes;
}

shared_ptr<CListRecord>
CListModeDataPETSIRD::get_empty_record_sptr() const
{
  shared_ptr<CListRecord> sptr(new CListRecordPETSIRD(petsird_info_sptr, get_proj_data_info_sptr()));
  return sptr;
}

Succeeded
CListModeDataPETSIRD::get_next_record(CListRecord& record_of_general_type) const
{
  auto& record = dynamic_cast<CListRecordPETSIRD&>(record_of_general_type);
  const auto& prompt_list = curr_event_block.prompt_events.at(0).at(0); // TODO: support multiple pairs of modules.
  const auto& delayed_list = m_has_delayeds ? curr_event_block.delayed_events.at(0).at(0) : prompt_list;
  const auto& event_list = curr_is_prompt ? prompt_list : delayed_list;

  if (event_list.size() == 0 && m_has_delayeds && curr_is_prompt)
    {
      // no prompts this time block, so let's try delayeds
      curr_is_prompt = false;
      curr_event_in_event_block = 0;
      return get_next_record(record);
    }
  if (event_list.size() == 0)
    {
      // no events, so read next Time block
      if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
        {
          current_lm_data_ptr->Close();
          return Succeeded::no;
        }
      ++m_time_block_index;
      curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
      return get_next_record(record);
    }

  auto event = event_list.at(curr_event_in_event_block);

  if (record.init_from_data(event, curr_is_prompt) == Succeeded::no
      || record_of_general_type.time().set_time_in_millisecs(curr_event_block.time_interval.start) == Succeeded::no)
    {
      return Succeeded::no;
    }
  assert(current_is_prompt == record.event().is_prompt());

  ++curr_event_in_event_block;

  if (curr_event_in_event_block < event_list.size())
    {
      return Succeeded::yes;
    }

  // - Once we hit the size of the vector
  curr_event_in_event_block = 0;

  if (!m_has_delayeds || curr_is_prompt)
    {
      if (m_has_delayeds)
        {
          curr_is_prompt = false;
        }
      else
        {
          if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
            {
              current_lm_data_ptr->Close();
              return Succeeded::no;
            }
          ++m_time_block_index;
          curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
        }
    }
  else
    {
      curr_is_prompt = true;
      if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
        {
          current_lm_data_ptr->Close();
          return Succeeded::no;
        }
      ++m_time_block_index;
      curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
    }

  return Succeeded::yes;
}

ListModeData::SavedPosition
CListModeDataPETSIRD::save_get_position()
{
  PetsirdCursor c;
  c.is_prompt = curr_is_prompt;
  c.event_in_block = curr_event_in_event_block;
  c.time_block_index = m_time_block_index;

  // Cache current blocks so set_get_position is instant (recommended)
  c.time_block = this->curr_time_block;
  c.event_block = this->curr_event_block;
  c.has_cached_blocks = true;

  m_saved_positions.push_back(std::move(c));
  return static_cast<SavedPosition>(m_saved_positions.size() - 1);
}

Succeeded
CListModeDataPETSIRD::reopen_and_prime()
{
  if (use_hdf5)
    current_lm_data_ptr.reset(new petsird::hdf5::PETSIRDReader(this->listmode_filename));
  else
    current_lm_data_ptr.reset(new petsird::binary::PETSIRDReader(this->listmode_filename));

  petsird::Header header;
  current_lm_data_ptr->ReadHeader(header);
  m_has_delayeds = header.scanner.delayed_events_are_stored;

  curr_event_in_event_block = 0;
  curr_is_prompt = true;
  m_time_block_index = 0;

  return seek_to_event_block_index(0);
}

Succeeded
CListModeDataPETSIRD::seek_to_event_block_index(std::size_t target_event_block_index) const
{
  try
    {
      while (m_time_block_index <= target_event_block_index)
        {
          if (!current_lm_data_ptr->ReadTimeBlocks(curr_time_block))
            return Succeeded::no;
          ++m_time_block_index;
          if (std::holds_alternative<petsird::EventTimeBlock>(curr_time_block))
            {
              curr_event_block = std::get<petsird::EventTimeBlock>(curr_time_block);
              if (m_time_block_index > target_event_block_index)
                return Succeeded::yes;
            }
        }
      return Succeeded::yes;
    }
  catch (...)
    {
      return Succeeded::no;
    }
}

Succeeded
CListModeDataPETSIRD::set_get_position(const SavedPosition& pos)
{
  if (pos >= m_saved_positions.size())
    return Succeeded::no;
  const auto& c = m_saved_positions[pos];

  // If you cached the actual blocks, you STILL must ensure the reader state
  // will not be used incorrectly. Easiest: reopen+seek anyway (robust),
  // then overwrite curr_* with cached data.
  if (reopen_and_prime() == Succeeded::no)
    return Succeeded::no;
  if (seek_to_event_block_index(c.time_block_index) == Succeeded::no)
    return Succeeded::no;

  // restore logical cursor
  curr_is_prompt = c.is_prompt;
  curr_event_in_event_block = c.event_in_block;
  m_time_block_index = c.time_block_index;
  if (c.has_cached_blocks)
    {
      this->curr_time_block = c.time_block;
      this->curr_event_block = c.event_block;
    }
  return Succeeded::yes;
}

Succeeded
CListModeDataPETSIRD::reset()
{
  return reopen_and_prime();
}
END_NAMESPACE_STIR