/*

     Copyright 2025, 2026, University Medical Center Groningen
     Copyright 2025, MGH / HST A. Martinos Center for Biomedical Imaging

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details.

 */

#ifndef __stir_listmode_CListModeDataPETSIRD_H__
#define __stir_listmode_CListModeDataPETSIRD_H__

#include "stir/listmode/CListModeDataBasedOnCoordinateMap.h"
#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListRecordPETSIRD.h"
#include "stir/Succeeded.h"
#include "petsird_helpers.h"
#include "stir/PETSIRDInfo.h"
#include "petsird/binary/protocols.h"
#include "petsird/hdf5/protocols.h"

START_NAMESPACE_STIR

/*!
  \class CListModeDataPETSIRD
  \brief Reader for PETSIRD listmode data supporting variable geometry.
  \ingroup listmode
  \author Nikos Efthimiou

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

  \note Exact PETSIRD format specification is defined in the PETSIRD project documentation.
  \note Initially, I wanted to:
        - Is close to a cylindrical geometry ?
          - then yes use a cylindrical scanner that is simpler.
          - Else, is it made of blocks arranged on a cylinder.

        However, now I do the following:
        - Is close to cylindrical geometry?
          - yes use cylindrical scanner
          - Check if blocks-on-cylinder configuration, are a good match.
            - yes use blocks-on-cylinder scanner
            - else use generic scanner and export the map to the disk.

  If listmode reconstruction is done, the map is regenerated on-the-fly.

*/
class CListModeDataPETSIRD : public CListModeDataBasedOnCoordinateMap
{
private:
  //! Snapshot of the current PETSIRD list-mode reader position.
  /*!
    Stores enough state to resume reading from a previously saved position
    in the PETSIRD stream. This includes the current prompt/delayed stream,
    the event index within the currently cached event block, the time-block
    index, and optionally cached block contents.
  */
  struct PetsirdCursor
  {
    //! Whether the cursor points to the prompt-event stream.
    /*!
      True for prompt events, false for delayed events.
    */
    bool is_prompt = true;
    //! Index of the next event within the cached event block.
    std::size_t event_in_block = 0;
    //! Index of the time block associated with this cursor.
    std::size_t time_block_index = 0;
    //! Cached PETSIRD time block at this cursor position.
    petsird::TimeBlock time_block;
    //! Cached PETSIRD event-time block at this cursor position.
    petsird::EventTimeBlock event_block;
    //! Whether time_block and event_block contain valid cached data.
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

  SavedPosition save_get_position() override;

  Succeeded reopen_and_prime();

  Succeeded seek_to_event_block_index(std::size_t target_event_block_index) const;

  Succeeded set_get_position(const SavedPosition& pos) override;

  virtual bool has_delayeds() const override { return m_has_delayeds; }

  Succeeded reset() override;

protected:
  virtual Succeeded open_lm_file() const override;

  shared_ptr<petsird::PETSIRDReaderBase> current_lm_data_ptr;

private:
  //! Whether to use the HDF5-based PETSIRD reader.
  const bool use_hdf5;
  //! Index of the current event within the currently loaded event block.
  mutable unsigned long int curr_event_in_event_block = 0;
  //! Currently loaded PETSIRD time block.
  mutable petsird::TimeBlock curr_time_block;
  //! Currently loaded PETSIRD event-time block.
  mutable petsird::EventTimeBlock curr_event_block;
  //! Prompt/delayed classification of the current event.
  //! True if the current event is a prompt event, false if it is delayed.
  mutable bool curr_is_prompt = true;
  //! Whether the PETSIRD data contains delayed events.
  mutable bool m_has_delayeds;
  //! Shared PETSIRD scanner and acquisition information.
  shared_ptr<const PETSIRDInfo> petsird_info_sptr;
  //! Cursor used to restore the most recently saved reader position
  mutable PetsirdCursor m_saved_cursor;
  //! Index of the current time block in the PETSIRD stream
  mutable std::size_t m_time_block_index = 0;
  //! Saved reader positions indexed by SavedPosition handles.
  /*!
    Each entry stores a cursor state that can later be restored, allowing
    random access or rollback to previously saved positions in the PETSIRD
    list-mode stream.
  */
  mutable std::vector<PetsirdCursor> m_saved_positions;
};

END_NAMESPACE_STIR
#endif // CLISTMODEDATAPETSIRD_H
