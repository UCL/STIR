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

  PHG_Decay nextDecay;

  //! The first detected photon
  PHG_DetectedPhoton cur_detectedPhoton;

  if (buffer.size() > 0)
    {
      int i = 0;
      bool found = false;
      float tofDifference;
      for (; i < buffer.size(); i++)
        {
          if ((binParams->numE1Bins > 0) && (buffer.at(i).first->energy < binParams->minE))
            continue;

          if ((binParams->numE1Bins > 0) && (buffer.at(i).first->energy > binParams->maxE))
            continue;

          if ((binParams->numE2Bins > 0) && (buffer.at(i).second->energy < binParams->minE))
            continue;

          if ((binParams->numE2Bins > 0) && (buffer.at(i).second->energy > binParams->maxE))
            continue;

          found = true;
          break;
        }

      if (found)
        {
          PHG_DetectedPhoton blue = *buffer.at(i).first;
          PHG_DetectedPhoton pink = *buffer.at(i).second;

          buffer.erase(buffer.begin() + i);

          float coincidenceWeight = (decay_weight * blue.photon_weight * pink.photon_weight);

          blue.location.z_position -= static_cast<float>(binParams->minZ);
          pink.location.z_position -= static_cast<float>(binParams->minZ);

          // STIR uses mm.
          blue.location.z_position *= 10.f;
          blue.location.y_position *= 10.f;
          blue.location.x_position *= 10.f;

          pink.location.z_position *= 10.f;
          pink.location.y_position *= 10.f;
          pink.location.x_position *= 10.f;

          //            std::cout << buffer.size() << std::endl;

          /* tofDifference is in nanoseconds, hence '1E9*' */
          tofDifference = 1.0E12 * (pink.time_since_creation - blue.time_since_creation);

          /* make sure that positive TOF is always oriented in the same direction,
                                  +TOF equating to +x. */
          //            if (blue.location.x_position < pink.location.x_position) {
          //                tofDifference = -tofDifference;
          //            } else if ( (blue.location.x_position == pink.location.x_position)
          //                        &&  (blue.location.y_position < pink.location.y_position) ) {
          //                tofDifference = -tofDifference;
          //            }

          if (tofDifference < 0.0)
            {
              tofDifference *= -1;
              return record.init_from_data(pink, blue, coincidenceWeight, tofDifference);
            }
          return record.init_from_data(blue, pink, coincidenceWeight, tofDifference);
        }
    }

  {

    EventTy eventType = readEvent(historyFile, &curDecay, &cur_detectedPhoton);

    while (eventType == Decay)
      {
        /* Clear decay variables */
        bluePhotons.clear();
        pinkPhotons.clear();
        buffer.clear();
        buffer_size = 0;

        eventType = readEvent(historyFile, &nextDecay, &cur_detectedPhoton);

        if (eventType == PhoHFileNullEvent)
          return Succeeded::no;

        decay_weight = nextDecay.startWeight;

        while (eventType == Photon)
          {
            /* See if it is blue or pink */
            if (LbFgIsSet((cur_detectedPhoton.flags & 3), PHGFg_PhotonBlue))
              {
                bluePhotons.push_back(cur_detectedPhoton);
              }
            else
              {
                pinkPhotons.push_back(cur_detectedPhoton);
              }
            eventType = readEvent(historyFile, &nextDecay, &cur_detectedPhoton);

            if (eventType == PhoHFileNullEvent)
              return Succeeded::no;
          }

        /* Update current decay */
        curDecay = nextDecay;
        //-> break
        if (PHG_IsPETCoincidencesOnly() || PHG_IsPETCoincPlusSingles())
          {
            if ((bluePhotons.size() == 1) && (pinkPhotons.size() == 1))
              {
                /* Reject the coincidence if it is random and
                  we are rejecting randoms */
                if ((curDecay.decayType == PhgEn_PETRandom) && (!binParams->acceptRandoms))
                  continue;

                if ((binParams->numE1Bins > 0) && (bluePhotons.at(0).energy < binParams->minE))
                  continue;

                if ((binParams->numE1Bins > 0) && (bluePhotons.at(0).energy > binParams->maxE))
                  continue;

                if ((binParams->numE2Bins > 0) && (pinkPhotons.at(0).energy < binParams->minE))
                  continue;

                if ((binParams->numE2Bins > 0) && (pinkPhotons.at(0).energy > binParams->maxE))
                  continue;

                float coincidenceWeight = (decay_weight * bluePhotons.at(0).photon_weight * pinkPhotons.at(0).photon_weight);

                bluePhotons.at(0).location.z_position -= static_cast<float>(binParams->minZ);
                pinkPhotons.at(0).location.z_position -= static_cast<float>(binParams->minZ);

                // STIR uses mm.
                bluePhotons.at(0).location.z_position *= 10.f;
                bluePhotons.at(0).location.y_position *= 10.f;
                bluePhotons.at(0).location.x_position *= 10.f;

                pinkPhotons.at(0).location.z_position *= 10.f;
                pinkPhotons.at(0).location.y_position *= 10.f;
                pinkPhotons.at(0).location.x_position *= 10.f;

                /* tofDifference is in nanoseconds, hence '1E9*' */
                float tofDifference = 1.0E12 * (pinkPhotons.at(0).time_since_creation - bluePhotons.at(0).time_since_creation);

                /* make sure that positive TOF is always oriented in the same direction,
                    +TOF equating to +x. */
                //                    if (bluePhotons.at(0).location.x_position < pinkPhotons.at(0).location.x_position) {
                //                        tofDifference = -tofDifference;
                //                    } else if ( (bluePhotons.at(0).location.x_position == pinkPhotons.at(0).location.x_position)
                //                                &&  (bluePhotons.at(0).location.y_position <
                //                                pinkPhotons.at(0).location.y_position) ) {
                //                        tofDifference = -tofDifference;
                //                    }

                if (tofDifference < 0.0)
                  {
                    tofDifference *= -1;
                    return record.init_from_data(pinkPhotons.at(0), bluePhotons.at(0), coincidenceWeight, tofDifference);
                    //                       std::cout <<tofDifference << std::endl;
                  }

                return record.init_from_data(bluePhotons.at(0), pinkPhotons.at(0), coincidenceWeight, tofDifference);
              }
            else if (bluePhotons.size() == 0 || pinkPhotons.size() == 0)
              {
                // warning("InputStreamFromSimSET: Single, not supported. continue... ");
                continue;
              }
            else if (bluePhotons.size() > 1 || pinkPhotons.size() > 1)
              {
                //                    warning("Multiples");
                // Create buffer
                for (int i = 0; i < bluePhotons.size(); ++i)
                  {
                    for (int j = 0; j < pinkPhotons.size(); ++j)
                      {
                        buffer.push_back({ &bluePhotons.at(i), &pinkPhotons.at(j) });
                        buffer_size++;
                      }
                  }

                buffer_size--;

                //                    /* Reject the coincidence if it is random and
                //                      we are rejecting randoms */
                //                    if ( (curDecay.decayType == PhgEn_PETRandom) &&
                //                            (!binParams->acceptRandoms) )
                //                        continue;

                int i = 0;
                bool found = false;
                for (; i < buffer.size(); ++i)
                  {
                    if ((binParams->numE1Bins > 0) && (buffer.at(i).first->energy < binParams->minE))
                      continue;

                    if ((binParams->numE1Bins > 0) && (buffer.at(i).first->energy > binParams->maxE))
                      continue;

                    if ((binParams->numE2Bins > 0) && (buffer.at(i).second->energy < binParams->minE))
                      continue;

                    if ((binParams->numE2Bins > 0) && (buffer.at(i).second->energy > binParams->maxE))
                      continue;

                    found = true;
                    break;
                  }

                if (found)
                  {
                    PHG_DetectedPhoton blue = *buffer.at(i).first;
                    PHG_DetectedPhoton pink = *buffer.at(i).second;

                    buffer.erase(buffer.begin() + i);

                    float coincidenceWeight = (decay_weight * blue.photon_weight * pink.photon_weight);

                    blue.location.z_position -= static_cast<float>(binParams->minZ);
                    pink.location.z_position -= static_cast<float>(binParams->minZ);

                    // STIR uses mm.
                    blue.location.z_position *= 10.f;
                    blue.location.y_position *= 10.f;
                    blue.location.x_position *= 10.f;

                    pink.location.z_position *= 10.f;
                    pink.location.y_position *= 10.f;
                    pink.location.x_position *= 10.f;

                    /* tofDifference is in nanoseconds, hence '1E9*' */
                    float tofDifference = 1.0E12 * (pink.time_since_creation - blue.time_since_creation);

                    /* make sure that positive TOF is always oriented in the same direction,
                    +TOF equating to +x. */
                    //                        if (blue.location.x_position < pink.location.x_position) {
                    //                            tofDifference = -tofDifference;
                    //                        } else if ( (blue.location.x_position == pink.location.x_position)
                    //                                    &&  (blue.location.y_position < pink.location.y_position) ) {
                    //                            tofDifference = -tofDifference;
                    //                        }

                    // if (tofDifference < 0.0)
                    //  {
                    //    tofDifference *= -1;
                    //  return record.init_from_data(pink, blue, coincidenceWeight, tofDifference);
                    //                             std::cout << "Negative" << std::endl;
                    //}

                    return record.init_from_data(blue, pink, coincidenceWeight, tofDifference);
                  }
              }
            else
              {
                error("Why am I here??");
              }
          }
        else
          {
            error("Not supported.");
          }
      }
  }
}

END_NAMESPACE_STIR
