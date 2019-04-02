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

extern "C" {
//#include <LbFile.h>
//#include <PhgHdr.h>
//#include <PhoHFile.h>
#include <print.header.h>
}


START_NAMESPACE_STIR

const char * const
InputStreamFromSimSET::registered_name =
        "SimSET_History_File";

InputStreamFromSimSET::
InputStreamFromSimSET()
{
    set_defaults();
}


InputStreamFromSimSET::
~InputStreamFromSimSET()
{
}

std::string
InputStreamFromSimSET::
method_info() const
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
InputStreamFromSimSET::
set_up(const std::string historyFileName,
       PHG_BinParamsTy * _binParams)
{
    strcpy(phgrdhstHistName, historyFileName.c_str());
    binParams = _binParams;

    /* the maximum scatter range is ignored when
    scatterRandomParam == 3,5,8 or 10 */
    ignoreMaxScatters = ( (binParams->scatterRandomParam == 3) ||
                (binParams->scatterRandomParam == 5) ||
                (binParams->scatterRandomParam == 8) ||
                (binParams->scatterRandomParam == 10) );

    /* for scatter-random parameters 4,5,9,10 the
    range min is for the sum of blue and pink scatters,
    and thus cannot be applied to individual photons */
    ignoreMinScatters = ( (binParams->scatterRandomParam == 4) ||
                (binParams->scatterRandomParam == 5) ||
                (binParams->scatterRandomParam == 9) ||
                (binParams->scatterRandomParam == 10) );

    if (set_up_standard_hist_file() == Succeeded::no)
    {
        return set_up_custom_hist_file();
    }

    return Succeeded::yes;
}


Succeeded
InputStreamFromSimSET::set_up_standard_hist_file()
{
    PHG_DetectedPhoton	detectedPhoton;				/* The detected photon */

    Boolean	isPHGList; /* this is PHG history file */
    Boolean	isColList; /* this is collimator history file */
    Boolean	isDetList; /* this is detector history file */

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
    if ( 	(phgrdhstHdrParams.H.HdrKind == PhoHFileEn_PHG) ||
            (phgrdhstHdrParams.H.HdrKind == PhoHFileEn_PHG2625) ||
            (phgrdhstHdrParams.H.HdrKind == PhoHFileEn_PHGOLD) )
    {
        isPHGList = true;
        isColList = false;
        isDetList = false;
    }
    else
    {
        error("InputStreamFromSimSET: File specified as PHG history file is not valid.");
    }

    PhoHFileEventType eventType = PhoHFileReadEvent(historyFile, &curDecay, &detectedPhoton);
    EventTy	locEventType;

    /* Convert to the local event type */
    switch ( eventType ) {
        case PhoHFileNullEvent:
            locEventType = Null;
            break;

        case PhoHFileDecayEvent:
            locEventType = Decay;
            break;

        case PhoHFilePhotonEvent:
            locEventType = Photon;
            break;

        default:
            locEventType = Null;
            break;
    }
//    firstDecay = curDecay;
//    rewind(historyFile);
    fseek(historyFile, headerHk.headerSize, SEEK_SET);

//    numDecaysProcessed++;

    if (locEventType != Decay)
    {
        error("InputStreamFromSimSET: Expected first event to be decay, and it wasn't.");
    }

    display(&phgrdhstHdrParams);

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

END_NAMESPACE_STIR
