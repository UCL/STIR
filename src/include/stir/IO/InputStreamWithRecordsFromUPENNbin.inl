/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamWithRecordsFromUPENNbin

  \author Nikos Efthimiou
*/
/*
    Copyright (C) 2020-2022 University of Pennsylvania
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


#include "/autofs/space/celer_001/users/nikos/src/UPENN/penn/include/liboption.h"


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
            std::cerr << "error: cannot create file " << ofilename << '\n';
            std::exit( EXIT_FAILURE );
        }
        outputList = &outputListFile;
    }

    if ( !list::encodeHeader( *outputList, listHeader ) )
    {
        std::cerr << "error: cannot write header to output list\n";
        std::exit( EXIT_FAILURE );
    }
    out = new list::OutputBuffer(*outputList, eventSize );
    has_output = true;

    if(keep_delayed == 2 && has_output)
    {
        error("You cannot keep delayed events and pass output.");
    }
    return Succeeded::yes;
}

END_NAMESPACE_STIR
