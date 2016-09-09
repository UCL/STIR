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
*/

#include "stir/IO/InputStreamFromROOTFile.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

long long int
InputStreamFromROOTFile::
get_total_number_of_events()
{
    return nentries;
}

Succeeded
InputStreamFromROOTFile::
reset()
{
    current_position = starting_stream_position;
    return Succeeded::yes;
}

InputStreamFromROOTFile::SavedPosition
InputStreamFromROOTFile::
save_get_position()
{
    assert(current_position <= nentries);

    if(current_position < nentries)
        saved_get_positions.push_back(current_position);
    else
        saved_get_positions.push_back(-1);
    return saved_get_positions.size()-1;
}

Succeeded
InputStreamFromROOTFile::
set_get_position(const InputStreamFromROOTFile::SavedPosition& pos)
{
    if (current_position == nentries)
        return Succeeded::no;

    assert(pos < saved_get_positions.size());
    if (saved_get_positions[pos] == -1)
        current_position = nentries; // go to eof
    else
        current_position = saved_get_positions[pos];

    return Succeeded::yes;
}

std::vector<long long int>
InputStreamFromROOTFile::
get_saved_get_positions() const
{
    return saved_get_positions;
}

void
InputStreamFromROOTFile::
set_saved_get_positions(const std::vector<long long int>& poss)
{
    saved_get_positions = poss;
}

END_NAMESPACE_STIR
