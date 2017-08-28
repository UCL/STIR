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

#include "stir/IO/InputStreamFromROOTFile.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

unsigned long int
InputStreamFromROOTFile::
get_total_number_of_events() const
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
    saved_get_positions.push_back(current_position);
    return saved_get_positions.size()-1;
}

Succeeded
InputStreamFromROOTFile::
set_get_position(const InputStreamFromROOTFile::SavedPosition& pos)
{
    assert(pos < saved_get_positions.size());
    if (saved_get_positions[pos] > nentries)
        current_position = nentries; // go to eof
    else
        current_position = saved_get_positions[pos];

    return Succeeded::yes;
}

std::vector<unsigned long int>
InputStreamFromROOTFile::
get_saved_get_positions() const
{
    return saved_get_positions;
}

void
InputStreamFromROOTFile::
set_saved_get_positions(const std::vector<unsigned long> &poss)
{
    saved_get_positions = poss;
}

float
InputStreamFromROOTFile::
get_low_energy_thres() const
{
    return low_energy_window;
}

float
InputStreamFromROOTFile::
get_up_energy_thres() const
{
    return up_energy_window;
}

std::string
InputStreamFromROOTFile::
get_ROOT_filename() const
{
    return this->filename;
}

END_NAMESPACE_STIR
