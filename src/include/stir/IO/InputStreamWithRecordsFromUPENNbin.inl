/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamWithRecordsFromUPENNbin

  \author Nikos Efthimiou
*/
/*
 *  Copyright (C) 2020-2022 University of Pennsylvania
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/


#include "liboption.h"


START_NAMESPACE_STIR

Succeeded
InputStreamWithRecordsFromUPENNbin::
create_output_file(std::string ofilename)
{
    olistCodec = new list::EventCodec( eventFormat );

    if ( !ofilename.empty() )
    {
        const std::ios_base::openmode writeonly = std::ios_base::out
                | std::ios_base::binary;

        if ( !outputListFile.open( ofilename.c_str(), writeonly ) )
        {
            error("Cannot create file " + ofilename);
        }
        outputList = &outputListFile;
    }

    if ( !list::encodeHeader( *outputList, listHeader ) )
    {
       error("cannot write header to output list");
    }
    out = new list::OutputBuffer(*outputList, eventSize );
    has_output = true;

    if(keep_delayed == 1 && has_output)
    {
        error("You cannot keep delayed events and pass output.");
    }
    return Succeeded::yes;
}

END_NAMESPACE_STIR
