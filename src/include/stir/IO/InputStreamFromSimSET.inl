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
static int ddd = 0;
Succeeded
InputStreamFromSimSET::
get_next_record(CListRecordSimSET& record)
{

    PHG_Decay nextDecay;

    //! The first detected photon
    PHG_DetectedPhoton	cur_detectedPhoton;

    if(buffer.size() > 0)
    {
        int i = 0;
        bool found = false;
        for (; i < buffer.size(); i++)
        {
            if ((binParams->numE1Bins > 0) && (buffer.at(i).first->energy < binParams->minE))
                continue;

            if	((binParams->numE1Bins > 0) && (buffer.at(i).first->energy > binParams->maxE))
                continue;

            if ((binParams->numE2Bins > 0) && (buffer.at(i).second->energy < binParams->minE))
                continue;

            if	((binParams->numE2Bins > 0) && (buffer.at(i).second->energy > binParams->maxE))
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


            std::cout << buffer.size() << std::endl;

            return record.init_from_data(blue, pink, coincidenceWeight);
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

            while ( eventType == Photon)
            {
                /* See if it is blue or pink */
                if (LbFgIsSet( (cur_detectedPhoton.flags & 3), PHGFg_PhotonBlue)) {
                    bluePhotons.push_back(cur_detectedPhoton);
                }
                else {
                    pinkPhotons.push_back(cur_detectedPhoton);
                }
                eventType = readEvent(historyFile, &nextDecay, &cur_detectedPhoton);

                if (eventType == PhoHFileNullEvent)
                    return Succeeded::no;
            }

            /* Update current decay */
            curDecay = nextDecay;
            //-> break
            if (PHG_IsPETCoincidencesOnly())
            {
                if ((bluePhotons.size() == 1) &&
                        (pinkPhotons.size() == 1))
                {
                    /* Reject the coincidence if it is random and
                      we are rejecting randoms */
                    if ( (curDecay.decayType == PhgEn_PETRandom) &&
                            (!binParams->acceptRandoms) )
                        continue;

                    if ((binParams->numE1Bins > 0) && (bluePhotons.at(0).energy < binParams->minE))
                        continue;

                    if	((binParams->numE1Bins > 0) && (bluePhotons.at(0).energy > binParams->maxE))
                        continue;

                    if ((binParams->numE2Bins > 0) && (pinkPhotons.at(0).energy < binParams->minE))
                        continue;

                    if	((binParams->numE2Bins > 0) && (pinkPhotons.at(0).energy > binParams->maxE))
                        continue;

                    float coincidenceWeight = (decay_weight * bluePhotons.at(0).photon_weight *
                                               pinkPhotons.at(0).photon_weight);

                    bluePhotons.at(0).location.z_position -= static_cast<float>(binParams->minZ);
                    pinkPhotons.at(0).location.z_position -= static_cast<float>(binParams->minZ);

                    // STIR uses mm.
                    bluePhotons.at(0).location.z_position *= 10.f;
                    bluePhotons.at(0).location.y_position *= 10.f;
                    bluePhotons.at(0).location.x_position *= 10.f;

                    pinkPhotons.at(0).location.z_position *= 10.f;
                    pinkPhotons.at(0).location.y_position *= 10.f;
                    pinkPhotons.at(0).location.x_position *= 10.f;

                    return record.init_from_data(bluePhotons.at(0),
                                                     pinkPhotons.at(0),
                                                     coincidenceWeight);

                }
                else if (bluePhotons.size() == 0 || pinkPhotons.size() == 0)
                {
//                    warning("Single");
                    continue;
                }
                else if (bluePhotons.size() > 1 || pinkPhotons.size() > 1)
                {
//                    warning("Multiples");
                    // Create buffer
                    for (int i = 0; i < bluePhotons.size(); ++i)
                    {
                        for(int j = 0; j< pinkPhotons.size(); ++j)
                        {
                            buffer.push_back({&bluePhotons.at(i), &pinkPhotons.at(j)});
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

                        if	((binParams->numE1Bins > 0) && (buffer.at(i).first->energy > binParams->maxE))
                            continue;

                        if ((binParams->numE2Bins > 0) && (buffer.at(i).second->energy < binParams->minE))
                            continue;

                        if	((binParams->numE2Bins > 0) && (buffer.at(i).second->energy > binParams->maxE))
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

                        return record.init_from_data(blue, pink, coincidenceWeight);
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
