/*
 *  Copyright (C) 2019 University of Hull
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
/*!
  \file
  \ingroup IO SimSET
  \brief Implementation of class stir::InputStreamFromSimSETFile

  \author Nikos Efthimiou
*/

#include "stir/IO/InputStreamFromSimSET.h"

extern "C" {
#include <PhoHFile.h>
}

START_NAMESPACE_STIR

unsigned long int
InputStreamFromSimSET::
get_total_number_of_events() const
{
    unsigned long int nikos =static_cast<unsigned long int>(phgrdhstHdrParams.H.NumDecays);
    return static_cast<unsigned long int>(phgrdhstHdrParams.H.NumDecays);
}

Succeeded
InputStreamFromSimSET::
reset()
{
    fseek(historyFile, headerHk.headerSize, SEEK_SET);
    blueScatters = 0;
    pinkScatters = 0;
    return Succeeded::yes;
}

InputStreamFromSimSET::SavedPosition
InputStreamFromSimSET::
save_get_position()
{
    std::streampos pos;
    pos = ftell(historyFile);
    saved_get_positions.push_back(pos);
    return saved_get_positions.size()-1;
}

Succeeded
InputStreamFromSimSET::
set_get_position(const InputStreamFromSimSET::SavedPosition& pos)
{
    assert(pos < saved_get_positions.size());

    if (saved_get_positions[pos] == std::streampos(-1))
      fseek (historyFile , 0 , SEEK_END ); // go to eof
    else
      fseek(historyFile, saved_get_positions[pos], SEEK_SET);

    return Succeeded::yes;
}

std::vector<std::streampos>
InputStreamFromSimSET::
get_saved_get_positions() const
{
    return saved_get_positions;
}

void
InputStreamFromSimSET::
set_saved_get_positions(const std::vector<std::streampos> &poss)
{
    saved_get_positions = poss;
}

Succeeded
InputStreamFromSimSET::
get_next_record(CListRecordSimSET& record)
{

    while(true)
    {

        EventTy	retEventType = Null;
        PhoHFileEventType	eventType = PhoHFileNullEvent;	/* The event type we read */

        while(true)
        {

            // We do not process decay events, so skip them
            do
            {
                eventType = PhoHFileReadEvent(historyFile, &curDecay, &cur_detectedPhotonBlue);

                if (eventType == PhoHFilePhotonEvent)
                    retEventType = Photon;

                if (eventType == PhoHFileDecayEvent)
                    retEventType = Decay;

                if (eventType == PhoHFileNullEvent)
                    return Succeeded::no;

            }while (retEventType == Decay);

            /* See if it is blue or pink */
            if (!LbFgIsSet(cur_detectedPhotonBlue.flags, PHGFg_PhotonBlue))
                continue;

            eventType = PhoHFileReadEvent(historyFile, &curDecay, &cur_detectedPhotonPink);
            if (LbFgIsSet(cur_detectedPhotonPink.flags, PHGFg_PhotonBlue))
                continue;

            if (cur_detectedPhotonBlue.energy < low_energy_threshold ||
                    cur_detectedPhotonBlue.energy > high_energy_threshold ||
                    cur_detectedPhotonPink.energy < low_energy_threshold ||
                    cur_detectedPhotonPink.energy > high_energy_threshold )
                continue;

            break;
        }

        /* Reject the coincidence if it is random and
          we are rejecting randoms */
        if ( (curDecay.decayType == PhgEn_PETRandom) &&
             (!binParams->acceptRandoms) )
            continue;

        /* Get scatter count */
//        blueScatters += cur_detectedPhotonBlue.flags>>2;

        /* See if this photon fails the acceptance criteria */
        /* NOTE: It seems like failing a blue should go to the
                        next blue, not the next pink. However, because we let
                        the user get their hands on the photon pair we will
                        always call the user routine with every blue and
                        every pink
                    */
        {
//            if ( (blueScatters < binParams->minS) &&
//                 (!ignoreMinScatters) )
//                continue;

            /* Max scatters is ignored for some scatterRandomParam, this means
                            take all photons from max and above, and put them into
                            the top bin
                        */
//            if	((blueScatters > binParams->maxS) &&
//                 (!ignoreMaxScatters))
//                continue;

            if ((binParams->numE1Bins > 0) && (cur_detectedPhotonBlue.energy < static_cast<float>(binParams->minE)))
                continue;

            if	((binParams->numE1Bins > 0) && (cur_detectedPhotonBlue.energy > static_cast<float>(binParams->maxE)))
                continue;

            if ((binParams->numZBins > 0) && (cur_detectedPhotonBlue.location.z_position <= static_cast<float>(binParams->minZ)))
                continue;

            if ((binParams->numZBins > 0) && (cur_detectedPhotonBlue.location.z_position >= static_cast<float>(binParams->maxZ)))
                continue;
        }


        /* Get scatter count */
//        pinkScatters += cur_detectedPhotonPink.flags>>2;

        /* See if this photon fails the acceptance criteria */
        {
//            if ( (pinkScatters < binParams->minS) &&
//                 (!ignoreMinScatters) )
//                continue;

            /* Max scatters is ignored for some scatterRandomParam, this means
                                        take all photons from max and above, and put them into
                                        the top bin
                                    */
//            if	( (pinkScatters > binParams->maxS) &&
//                  (!ignoreMaxScatters) )
//                continue;

            if ((binParams->numE2Bins > 0) && (cur_detectedPhotonPink.energy < static_cast<float>(binParams->minE)))
                continue;

            if	((binParams->numE2Bins > 0) && (cur_detectedPhotonPink.energy > static_cast<float>(binParams->maxE)))
                continue;

            if ((binParams->numZBins > 0) && (cur_detectedPhotonPink.location.z_position <= static_cast<float>(binParams->minZ)))
                continue;

            if ((binParams->numZBins > 0) && (cur_detectedPhotonPink.location.z_position >= static_cast<float>(binParams->maxZ)))
                continue;
        }
        break;
    }

    cur_detectedPhotonBlue.location.z_position -= static_cast<float>(binParams->minZ);
    cur_detectedPhotonPink.location.z_position -= static_cast<float>(binParams->minZ);

    // STIR uses mm.
    cur_detectedPhotonBlue.location.z_position *= 10.f;
    cur_detectedPhotonBlue.location.y_position *= 10.f;
    cur_detectedPhotonBlue.location.x_position *= 10.f;

    cur_detectedPhotonPink.location.z_position *= 10.f;
    cur_detectedPhotonPink.location.y_position *= 10.f;
    cur_detectedPhotonPink.location.x_position *= 10.f;

    return record.init_from_data(cur_detectedPhotonBlue,
                                 cur_detectedPhotonPink);
}

END_NAMESPACE_STIR
