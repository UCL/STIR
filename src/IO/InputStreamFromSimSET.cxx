/*
    Copyright (C) 2019, University of Hull
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
#include "stir/IO/InputStreamFromSimSET.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"

extern "C"
{
//#include <LbFile.h>
//#include <PhgHdr.h>
#include <PhoHFile.h>
#include <print.header.h>
}

START_NAMESPACE_STIR

const char* const InputStreamFromSimSET::registered_name = "SimSET_History_File";

InputStreamFromSimSET::InputStreamFromSimSET()
{
  set_defaults();
}

InputStreamFromSimSET::~InputStreamFromSimSET()
{}

unsigned long int
InputStreamFromSimSET::get_total_number_of_events() const
{
  return static_cast<unsigned long int>(phgrdhstHdrParams.H.NumDecays);
}

Succeeded
InputStreamFromSimSET::reset()
{
  fseek(historyFile, headerHk.headerSize, SEEK_SET);
  blueScatters = 0;
  pinkScatters = 0;
  bluePhotons.clear();
  pinkPhotons.clear();
  buffer.clear();
  have_pending_decay = false;
  reached_eof = false;
  return Succeeded::yes;
}

InputStreamFromSimSET::SavedPosition
InputStreamFromSimSET::save_get_position()
{
  std::streampos pos;
  pos = ftell(historyFile);
  saved_get_positions.push_back(pos);
  return saved_get_positions.size() - 1;
}

Succeeded
InputStreamFromSimSET::set_get_position(const InputStreamFromSimSET::SavedPosition& pos)
{
  assert(pos < saved_get_positions.size());

  if (saved_get_positions[pos] == std::streampos(-1))
    fseek(historyFile, 0, SEEK_END); // go to eof
  else
    fseek(historyFile, saved_get_positions[pos], SEEK_SET);

  bluePhotons.clear();
  pinkPhotons.clear();
  buffer.clear();
  have_pending_decay = false;
  reached_eof = false;

  return Succeeded::yes;
}

std::vector<std::streampos>
InputStreamFromSimSET::get_saved_get_positions() const
{
  return saved_get_positions;
}

void
InputStreamFromSimSET::set_saved_get_positions(const std::vector<std::streampos>& poss)
{
  saved_get_positions = poss;
}

std::string
InputStreamFromSimSET::method_info() const
{
  std::ostringstream s;
  s << this->registered_name;
  return s.str();
}

void
InputStreamFromSimSET::set_defaults()
{
  startFileIndex = 0;
  curFileIndex = 0;
  numPhotonsProcessed = 0;
  numDecaysProcessed = 0;
  numPhotons = 0;
  have_pending_decay = false;
  reached_eof = false;
}

Succeeded
InputStreamFromSimSET::set_up(const std::string historyFileName,
                              PHG_BinParamsTy const& _binParams,
                              const float lowEnWin,
                              const float highEnWin)
{
  strcpy(phgrdhstHistName, historyFileName.c_str());
  binParams = &_binParams;

  /* the maximum scatter range is ignored when
  scatterRandomParam == 3,5,8 or 10 */
  ignoreMaxScatters = ((binParams->scatterRandomParam == 3) || (binParams->scatterRandomParam == 5)
                       || (binParams->scatterRandomParam == 8) || (binParams->scatterRandomParam == 10));

  /* for scatter-random parameters 4,5,9,10 the
  range min is for the sum of blue and pink scatters,
  and thus cannot be applied to individual photons */
  ignoreMinScatters = ((binParams->scatterRandomParam == 4) || (binParams->scatterRandomParam == 5)
                       || (binParams->scatterRandomParam == 9) || (binParams->scatterRandomParam == 10));

  if (set_up_standard_hist_file() == Succeeded::no)
    {
      return set_up_custom_hist_file();
    }

  low_energy_threshold = lowEnWin;

  high_energy_threshold = highEnWin;

  return Succeeded::yes;
}

Succeeded
InputStreamFromSimSET::set_up_standard_hist_file()
{
  PHG_DetectedPhoton detectedPhoton; /* The detected photon */

  Boolean isPHGList; /* this is PHG history file */
  Boolean isColList; /* this is collimator history file */
  Boolean isDetList; /* this is detector history file */

  /* Open history file file */
  if ((historyFile = LbFlFileOpen(phgrdhstHistName, "rb")) == nullptr)
    {
      return Succeeded::no;
    }

  /* Read in the header and verify it is the right type of file */
  if (PhgHdrGtParams(historyFile, &phgrdhstHdrParams, &headerHk) == false)
    {
      return Succeeded::no;
    }

  /* Verify old collimator/detector list mode files are not being used for SPECT/DHCI:
   photons had insufficient information for further processing--no detector angle was saved - NE: skipped */

  /* Set flags for type of list mode file being processed */
  if ((phgrdhstHdrParams.H.HdrKind == PhoHFileEn_DET) || (phgrdhstHdrParams.H.HdrKind == PhoHFileEn_DET2625)
      || (phgrdhstHdrParams.H.HdrKind == PhoHFileEn_DETOLD))
    {
      isPHGList = false;
      isColList = false;
      isDetList = true;
    }
  else
    {
      error("InputStreamFromSimSET: File specified as PHG history file is not valid.");
    }

  //    PhoHFileEventType eventType = PhoHFileReadEvent(historyFile, &curDecay, &detectedPhoton);
  //    EventTy	locEventType;

  //    /* Convert to the local event type */
  //    switch ( eventType ) {
  //        case PhoHFileNullEvent:
  //            locEventType = Null;
  //            break;

  //        case PhoHFileDecayEvent:
  //            locEventType = Decay;
  //            break;

  //        case PhoHFilePhotonEvent:
  //            locEventType = Photon;
  //            break;

  //        default:
  //            locEventType = Null;
  //            break;
  //    }
  ////    firstDecay = curDecay;
  ////    rewind(historyFile);
  //    fseek(historyFile, headerHk.headerSize, SEEK_SET);

  ////    numDecaysProcessed++;

  //    if (locEventType != Decay)
  //    {
  //        error("InputStreamFromSimSET: Expected first event to be decay, and it wasn't.");
  //    }

  //    display(&phgrdhstHdrParams);

  return Succeeded::yes;
}

Succeeded
InputStreamFromSimSET::set_up_custom_hist_file()
{

  //    Boolean	isPHGList; /* this is PHG history file */
  //    Boolean	isColList; /* this is collimator history file */
  //    Boolean	isDetList; /* this is detector history file */

  //    /* Open history file file */
  //    if ((historyFile = LbFlFileOpen(phgrdhstHistName, "rb")) == nullptr)
  //    {
  //        return Succeeded::no;
  //    }

  //    /* Read in the header and verify it is the right type of file */
  //    if (PhgHdrGtParams(historyFile, &phgrdhstHdrParams, &headerHk) == false)
  //    {
  //        return Succeeded::no;
  //    }

  //    /* Set flags for type of list mode file being processed */
  //    if ( 	(phgrdhstHdrParams.H.HdrKind == PhoHFileEn_PHG) ||
  //        (phgrdhstHdrParams.H.HdrKind == PhoHFileEn_PHG2625) ||
  //        (phgrdhstHdrParams.H.HdrKind == PhoHFileEn_PHGOLD) )
  //    {
  //        isPHGList = true;
  //        isColList = false;
  //        isDetList = false;
  //    }
  //    else
  //    {
  //        error("InputStreamFromSimSET: File specified as PHG history file is not valid.");
  //    }

  //    /* Verify that requested file is of the right type */
  //    {
  //        if ( PHGRDHST_IsUsePHGHistory() && (!isPHGList) )
  //        {
  //            error("InputStreamFromSimSET: File specified as PHG history file is not valid.");
  //        }

  //        if ( PHGRDHST_IsUseColHistory() && (!isColList) )
  //        {
  //            error("InputStreamFromSimSET: File specified as Collimator history file is not valid.");
  //        }

  //        if ( PHGRDHST_IsUseDetHistory() && (!isDetList) )
  //        {
  //            error("InputStreamFromSimSET: File specified as Detector history file is not valid.");
  //        }
  //    }

  //    /* Setup header hook */
  //    {
  //        histHk.doCustom = true;

  //        /* Do custom parameter initialization  */
  //        if (PhoHFileGetRunTimeParams(phgrdhstHistParamsName, &(histHk.customParams)) == false)
  //        {
  //            error("InputStreamFromSimSET: Unable to get custom parameters for history file.");
  //        }

  //        strcpy(histHk.customParamsName, phgrdhstHistParamsName);

  //        histHk.bluesReceived = 0;
  //        histHk.bluesAccepted = 0;
  //        histHk.pinksReceived = 0;
  //        histHk.pinksAccepted = 0;
  //        histHk.histFile = historyFile;
  //    }

  return Succeeded::no;
}

Succeeded
InputStreamFromSimSET::get_next_record(CListRecordSimSET& record)
{
  for (;;)
    {
      while (!buffer.empty())
        {
          PHG_DetectedPhoton blue = buffer.front().first;
          PHG_DetectedPhoton pink = buffer.front().second;
          buffer.erase(buffer.begin());

          if ((binParams->numE1Bins > 0) && (blue.energy < binParams->minE || blue.energy > binParams->maxE))
            continue;
          if ((binParams->numE2Bins > 0) && (pink.energy < binParams->minE || pink.energy > binParams->maxE))
            continue;

          const float coincidence_weight = decay_weight * blue.photon_weight * pink.photon_weight;

          blue.location.z_position = (blue.location.z_position - static_cast<float>(binParams->minZ)) * 10.F - 4.1026F/2;
          blue.location.y_position *= 10.F;
          blue.location.x_position *= 10.F;
          pink.location.z_position = (pink.location.z_position - static_cast<float>(binParams->minZ)) * 10.F - 4.1026F/2;
          pink.location.y_position *= 10.F;
          pink.location.x_position *= 10.F;

          // SimSET stores times in seconds; STIR expects the difference in ps.
          const float tof_difference = 1.E12F * (pink.time_since_creation - blue.time_since_creation);

          // A rejected geometrical pair is not end-of-file. Keep looking.
          if (record.init_from_data(pink, blue, coincidence_weight, tof_difference) == Succeeded::yes)
            return Succeeded::yes;
        }

      if (reached_eof)
        return Succeeded::no;

      PHG_DetectedPhoton detected_photon{};
      EventTy current_event;

      if (have_pending_decay)
        {
          current_event = Decay;
          have_pending_decay = false;
        }
      else
        {
          current_event = readEvent(historyFile, &curDecay, &detected_photon);
        }

      // Be robust against an unexpected photon at a restored file position.
      while (current_event != Decay && current_event != PhoHFileNullEvent)
        current_event = readEvent(historyFile, &curDecay, &detected_photon);

      if (current_event == PhoHFileNullEvent)
        {
          reached_eof = true;
          continue;
        }

      const PHG_Decay decay_being_processed = curDecay;
      bluePhotons.clear();
      pinkPhotons.clear();

      PHG_Decay next_decay{};
      current_event = readEvent(historyFile, &next_decay, &detected_photon);
      while (current_event == Photon)
        {
          if (LbFgIsSet((detected_photon.flags & 3), PHGFg_PhotonBlue))
            bluePhotons.push_back(detected_photon);
          else
            pinkPhotons.push_back(detected_photon);

          current_event = readEvent(historyFile, &next_decay, &detected_photon);
        }

      if (current_event == Decay)
        {
          curDecay = next_decay;
          have_pending_decay = true;
        }
      else
        {
          reached_eof = true;
        }

      if (!(PHG_IsPETCoincidencesOnly() || PHG_IsPETCoincPlusSingles()))
        error("InputStreamFromSimSET: only PET coincidence history is supported.");

      if ((decay_being_processed.decayType == PhgEn_PETRandom) && !binParams->acceptRandoms)
        continue;

      decay_weight = decay_being_processed.startWeight;

      // Preserve every blue-pink combination as values. This avoids dangling
      // pointers when the photon vectors are cleared for the next decay.
      for (const auto& blue : bluePhotons)
        for (const auto& pink : pinkPhotons)
          buffer.emplace_back(blue, pink);
    }
}

END_NAMESPACE_STIR
