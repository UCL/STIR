/*
 *  Copyright (C) 2015, 2016 University of Leeds
    Copyright (C) 2016, UCL
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
  \ingroup IO
  \brief Implementation of class stir::InputStreamFromROOTFile

  \author Nikos Efthimiou
  \author Harry Tsoumpas
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
    return static_cast<unsigned long int>(numPhotons);
}

Succeeded
InputStreamFromSimSET::
reset()
{
    curFileIndex = startFileIndex;
    return Succeeded::yes;
}

InputStreamFromSimSET::SavedPosition
InputStreamFromSimSET::
save_get_position()
{
    assert(curFileIndex <= numPhotons);
    saved_get_positions.push_back(curFileIndex);
    return saved_get_positions.size()-1;
}

Succeeded
InputStreamFromSimSET::
set_get_position(const InputStreamFromSimSET::SavedPosition& pos)
{
    curFileIndex = pos;
    return Succeeded::yes;
}

std::vector<LbUsFourByte>
InputStreamFromSimSET::
get_saved_get_positions() const
{
    return saved_get_positions;
}

void
InputStreamFromSimSET::
set_saved_get_positions(const std::vector<LbUsFourByte> &poss)
{
    saved_get_positions = poss;
}

Succeeded
InputStreamFromSimSET::
get_next_record(CListRecordSimSET& record)
{

    LbUsFourByte		numBluePhotons;				/* Number of blue photons for this decay */
    LbUsFourByte		numPinkPhotons;				/* Number of pink photons for this decay */

    LbUsFourByte		numPhotonsProcessed;		/* Number of photons processed */
    LbUsFourByte		numDecaysProcessed;			/* Number of decays processed */
    PHG_Decay		   	decay;						/* The decay */

    PHG_DetectedPhoton	detectedPhoton;				/* The detected photon */
    PHG_TrackingPhoton	trackingPhoton;				/* The tracking photon */
    PHG_TrackingPhoton	*bluePhotons = 0;			/* Blue photons for current decay */
    PHG_TrackingPhoton	*pinkPhotons = 0;			/* Pink photon for current decay*/
    double 				angle_norm;					/* for normalizing photon direction */
    Boolean				isOldPhotons1;				/* is this a very old history file--must be
                                                     read using oldReadEvent */
    Boolean				isOldPhotons2;				/* is this a moderately old history file--must be
                                                     read using oldReadEvent */
    Boolean				isOldDecays;				/* is this an old history file--must be read using oldReadEvent */

    EventTy	retEventType = Null;
    PhoHFileEventType	eventType = PhoHFileNullEvent;	/* The event type we read */

    while (retEventType == Decay)
    {
        numDecaysProcessed++;

        eventType = PhoHFileReadEvent(historyFile, &nextDecay, &detectedPhoton);

        if (eventType == PhoHFilePhotonEvent)
            retEventType = Photon;

        if (eventType == PhoHFileNullEvent)
            error("");


        /* Update current decay */
        decay = nextDecay;

    }

    int nikos = 0;
    //    return
    //            record.init_from_data(ring1, ring2,
    //                                  crystal1, crystal2,
    //                                  time1, time2,
    //                                  eventID1, eventID2);
}

END_NAMESPACE_STIR
