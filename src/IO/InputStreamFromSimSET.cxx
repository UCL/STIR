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
#include <print.header.h>
#include <LbFile.h>
#include <PhgHdr.h>
#include <PhoHFile.h>
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
    // Try to close history file if opened
    // -Keep in mind that this is not were the actual reading will
    // take place, but in the InputStreamFromSimSET.
    // Here we just need the header.
    if (historyFile != 0)
        fclose(historyFile);
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
}

Succeeded
InputStreamFromSimSET::
set_up(const std::string & _history_params_filename)
{

    history_filename = _history_params_filename;

    if (set_up_standard_hist_file() == Succeeded::no)
    {
        return set_up_custom_hist_file();
    }

    return Succeeded::yes;
}


Succeeded
InputStreamFromSimSET::set_up_standard_hist_file()
{
    PHG_Decay		   	decay;						/* The decay */
    PHG_DetectedPhoton	detectedPhoton;				/* The detected photon */

    Boolean	isPHGList; /* this is PHG history file */
    Boolean	isColList; /* this is collimator history file */
    Boolean	isDetList; /* this is detector history file */


    char *history_cstr = new char[history_filename.length() + 1];
    strcpy(history_cstr, history_filename.c_str());

    /* Open history file file */
    if ((historyFile = LbFlFileOpen(history_cstr, "rb")) == nullptr)
    {
        delete [] history_cstr;
        return Succeeded::no;
    }

    /* Read in the header and verify it is the right type of file */
    if (PhgHdrGtParams(historyFile, phgrdhstHdrParams.get(), headerHk.get()) == false)
    {
        delete [] history_cstr;
        return Succeeded::no;
    }

    /* Verify old collimator/detector list mode files are not being used for SPECT/DHCI:
     photons had insufficient information for further processing--no detector angle was saved - NE: skipped */

    /* Set flags for type of list mode file being processed */
    if ( 	(phgrdhstHdrParams->H.HdrKind == PhoHFileEn_PHG) ||
            (phgrdhstHdrParams->H.HdrKind == PhoHFileEn_PHG2625) ||
            (phgrdhstHdrParams->H.HdrKind == PhoHFileEn_PHGOLD) )
    {

        isPHGList = true;
        isColList = false;
        isDetList = false;
    }
    else
    {
        error("InputStreamFromSimSET: File specified as PHG history file is not valid.");
    }

    PhoHFileEventType eventType = PhoHFileReadEvent(historyFile, &decay, &detectedPhoton);
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

    if (locEventType != Decay)
    {
        error("InputStreamFromSimSET: Expected first event to be decay, and it wasn't.");
    }

    delete [] history_cstr;
    return Succeeded::yes;
}

Succeeded
InputStreamFromSimSET::set_up_custom_hist_file()
{

    return Succeeded::yes;
}

END_NAMESPACE_STIR


//phgrdhstHdrParams.reset(new PhoHFileHdrTy);
//headerHk.reset(new LbHdrHkTy);

//// Let's read the header in the proper structure.
//char *history_cstr = new char[history_filename.length() + 1];
//strcpy(history_cstr, history_filename.c_str());

///* Open data file file */
//if ((historyFile = LbFlFileOpen(history_cstr, "rb")) == nullptr)
//{
//    warning("CListModeDataSimSET: Unable to open history file.");
//    return Succeeded::no;
//}

///* Read in the header and verify it is the right type of file */
//if (PhgHdrGtParams(historyFile, phgrdhstHdrParams.get(), headerHk.get()) == false)
//{
//    warning("CListModeDataSimSET: Unable to read the header. Wrong type or version?.");
//    return Succeeded::no;
//}

///* Display the header */
//info("History file Header: ");
//display(phgrdhstHdrParams.get());


//delete [] history_cstr;
