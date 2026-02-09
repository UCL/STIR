/* CListModeDataPETSIRD.h

Coincidence LM Data Class for PETSIRD

     Copyright 2025, UMCG
     Copyright 2025, MGH / HST A. Martinos Center for Biomedical Imaging

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

             http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.

 */

/*!

\file
\ingroup listmode
\brief Declaration of class stir::CListModeDataPETSIRD

\author Daniel Deidda
\author Nikos Efthimiou
*/

#ifndef __stir_listmode_CListModeDataPETSIRD_H__
#define __stir_listmode_CListModeDataPETSIRD_H__

#include "stir/listmode/CListModeDataBasedOnCoordinateMap.h"
#include "stir/ProjData.h"
#include "stir/listmode/CListRecord.h"
#include "stir/shared_ptr.h"

#include "petsird_helpers.h"
#include "stir/PETSIRDInfo.h"
#include "petsird/binary/protocols.h"
#include "petsird/hdf5/protocols.h"

START_NAMESPACE_STIR

/*!
  \class CListModeDataPETSIRD
  \brief Reader for PETSIRD listmode data supporting variable geometry.
  \ingroup listmode

  \par Overview
  - Supports HDF5 and binary PETSIRD formats.
  - Infers scanner geometry:
    - Cylindrical → creates cylindrical scanner.
    - Block-based → creates block-based scanner.
    - Otherwise → creates generic scanner using crystal positions.
  - Builds a DetectorCoordinateMap when needed and stores to disk.
  \par

  Infering the scanner geometry makes a lot of assumptions about what PET is.
  In particular, it assumes:
  \li A PET scanner is made of rings of detectors
  \li The largest axis is the axial one.
  \li So far we support only a single layer. This is partly hard-coded for simplicity. (look in the code for relevant TODOs and
  comments.)
  \li Some of the hardcoded assumptions are in CListRecordPETSIRD as well.

  \note Exact PETSIRD format specification is defined in the PETSIRD project documentation. (NOTE: It looks like GATE.)
  \note Initially, I wanted to:
        - Is close to a cylindrical geometry ?
          - then yes use a cylindrical scanner that is simpler.
          - Else, is it made of blocks arranged on a cylinder.

        However, now I do the following:
        - Is close to cylindriacl geometry?
          - yes use cylindrical scanner
          - Check if blocks-on-cylinder configuration, are a good match.
            - yes use blocks-on-cylinder scanner
            - else use generic scanner and export the map to the disk.

  If listmode reconstruciton is done, the map is regenerated on-the-fly.

*/
class CListModeDataPETSIRD : public CListModeDataBasedOnCoordinateMap
{
private:
  struct PetsirdCursor
  {
    bool is_prompt = true;
    std::size_t event_in_block = 0;
    std::size_t time_block_index = 0;

    petsird::TimeBlock time_block;
    petsird::EventTimeBlock event_block;
    bool has_cached_blocks = false;
  };

public:
  /*!
    \brief Construct reader.
    \param listmode_filename Path to PETSIRD listmode file.
    \param use_hdf5 If true, use HDF5 reader; otherwise use binary reader.
  */
  CListModeDataPETSIRD(const std::string& listmode_filename, bool use_hdf5);

  virtual shared_ptr<CListRecord> get_empty_record_sptr() const override;

  Succeeded get_next_record(CListRecord& record_of_general_type) const override;

  SavedPosition save_get_position() override
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

  Succeeded reopen_and_prime()
  {
    // ensure PETSIRD state machine is satisfied
    std::cout << "1 nikos" << std::endl; 
    if (current_lm_data_ptr)
      {
        try
          {
            current_lm_data_ptr->Close();
          }
        catch (...)
          {}
      }
    // current_lm_data_ptr.reset();
      std::cout << "2 nikos" << std::endl; 
    if (use_hdf5)
      current_lm_data_ptr.reset(new petsird::hdf5::PETSIRDReader(this->listmode_filename));
    else
      current_lm_data_ptr.reset(new petsird::binary::PETSIRDReader(this->listmode_filename));

      petsird::Header header;
      current_lm_data_ptr->ReadHeader(header);
    // m_eof_reached = false;
    curr_event_in_event_block = 0;
    curr_is_prompt = true;
    m_time_block_index = 0;
    std::cout << "3 nikos" << std::endl;
    // read until first EventTimeBlock
    while (true)
      {
        std::cout << "4 nikos" << std::endl; 
        if (!current_lm_data_ptr->ReadTimeBlocks(this->curr_time_block))
          {
            std::cout << "5 nikos" << std::endl; 
            // m_eof_reached = true;
            current_lm_data_ptr->Close();
            return Succeeded::no;
          }
          std::cout << "55 nikos" << std::endl; 
        if (std::holds_alternative<petsird::EventTimeBlock>(this->curr_time_block))
          {
            std::cout << "6 nikos" << std::endl; 
            this->curr_event_block = std::get<petsird::EventTimeBlock>(this->curr_time_block);
            return Succeeded::yes;
          }
      }
  }

  Succeeded seek_to_event_block_index(std::size_t target_event_block_index) const
  {
    // assumes we are primed at event_block_index = 0
    std::size_t idx = 0;
    while (idx < target_event_block_index)
      {
        // read next until EventTimeBlock
        while (true)
          {
            if (!current_lm_data_ptr->ReadTimeBlocks(this->curr_time_block))
              {
                // m_eof_reached = true;
                current_lm_data_ptr->Close();
                return Succeeded::no;
              }
            if (std::holds_alternative<petsird::EventTimeBlock>(this->curr_time_block))
              break;
          }
        this->curr_event_block = std::get<petsird::EventTimeBlock>(this->curr_time_block);
        ++idx;
      }
    return Succeeded::yes;
  }

  Succeeded set_get_position(const SavedPosition& pos) override
  {
    if (pos >= m_saved_positions.size())
      return Succeeded::no;
std::cout << "---1 nikos" << std::endl; 
    const auto& c = m_saved_positions[pos];

    // If you cached the actual blocks, you STILL must ensure the reader state
    // will not be used incorrectly. Easiest: reopen+seek anyway (robust),
    // then overwrite curr_* with cached data.
    if (reopen_and_prime() == Succeeded::no)
      return Succeeded::no;
std::cout << "---2 nikos" << std::endl; 
    if (seek_to_event_block_index(c.time_block_index) == Succeeded::no)
      return Succeeded::no;

    // restore logical cursor
    curr_is_prompt = c.is_prompt;
    curr_event_in_event_block = c.event_in_block;
    m_time_block_index = c.time_block_index;
    std::cout << "---3 nikos" << std::endl; 
    if (c.has_cached_blocks)
      {
        this->curr_time_block = c.time_block;
        this->curr_event_block = c.event_block;
      }
std::cout << "---4 nikos" << std::endl; 
    return Succeeded::yes;
  }

  virtual bool has_delayeds() const override { return m_has_delayeds; }

  Succeeded reset() override
  {
    std::cout << "RESETING CListModeDataPETSIRD" << std::endl;
    // if (current_lm_data_ptr)
    //   {
    //     try
    //       {
    //         current_lm_data_ptr->Close();
    //       }
    //     catch (...)
    //       {
    //         // If Close throws, treat as failure (or swallow if you must)
    //         return Succeeded::no;
    //       }
    //   }

    if (use_hdf5)
      current_lm_data_ptr.reset(new petsird::hdf5::PETSIRDReader(this->listmode_filename));
    else
      current_lm_data_ptr.reset(new petsird::binary::PETSIRDReader(this->listmode_filename));

    // current_lm_data_ptr->ReadHeader(header);

    curr_event_in_event_block = 0;
    curr_is_prompt = true;
    m_time_block_index = 0;

    // if (!current_lm_data_ptr->ReadTimeBlocks(this->curr_time_block))
    //   return Succeeded::no;

    try
      {
        while (true)
          {
            std::cout << "Reading TimeBlock index " << m_time_block_index << std::endl;
            if (!current_lm_data_ptr->ReadTimeBlocks(this->curr_time_block))
              return Succeeded::no;

            ++m_time_block_index;

            if (std::holds_alternative<petsird::EventTimeBlock>(this->curr_time_block))
              {
                this->curr_event_block = std::get<petsird::EventTimeBlock>(this->curr_time_block);
                break;
              }
          }
      }
    catch (...)
      {
        return Succeeded::no;
      }

    return Succeeded::yes;
  }

protected:
  virtual Succeeded open_lm_file() const override;

  shared_ptr<petsird::PETSIRDReaderBase> current_lm_data_ptr;

private:
  //! Whether to use the HDF5 reader.
  const bool use_hdf5;

  mutable unsigned long int curr_event_in_event_block = 0;

  mutable petsird::TimeBlock curr_time_block;

  mutable petsird::EventTimeBlock curr_event_block;
  //! Current event prompt flag.
  mutable bool curr_is_prompt = true;
  //! Whether delayed events are present.
  mutable bool m_has_delayeds;

  shared_ptr<const PETSIRDInfo> petsird_info_sptr;

  mutable PetsirdCursor m_saved_cursor;
  mutable std::size_t m_time_block_index = 0;
  // saved states indexed by SavedPosition handle
  mutable std::vector<PetsirdCursor> m_saved_positions;
};

END_NAMESPACE_STIR
#endif // CLISTMODEDATAPETSIRD_H
