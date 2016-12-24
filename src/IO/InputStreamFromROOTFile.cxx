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

#include "stir/IO/InputStreamFromROOTFile.h"

START_NAMESPACE_STIR

InputStreamFromROOTFile::
InputStreamFromROOTFile()
{
    starting_stream_position = 0;
    reset();
}


InputStreamFromROOTFile::
InputStreamFromROOTFile(std::string filename,
                        std::string chain_name,
                        bool exclude_scattered, bool exclude_randoms,
                        float low_energy_window, float up_energy_window,
                        int offset_dets)
    : filename(filename), chain_name(chain_name),
      exclude_scattered(exclude_scattered), exclude_randoms(exclude_randoms),
      low_energy_window(low_energy_window), up_energy_window(up_energy_window), offset_dets(offset_dets)
{
    starting_stream_position = 0;
    reset();
}

void
InputStreamFromROOTFile::set_defaults()
{}

void
InputStreamFromROOTFile::initialise_keymap()
{
    this->parser.add_key("name of data file", &this->filename);
    this->parser.add_key("Singles readout depth", &this->singles_readout_depth);
    this->parser.add_key("name of input TChain", &this->chain_name);
    this->parser.add_key("exclude scattered events", &this->exclude_scattered);
    this->parser.add_key("exclude random events", &this->exclude_randoms);
    this->parser.add_key("offset (num of detectors)", &this->offset_dets);
    this->parser.add_key("low energy window (keV)", &this->low_energy_window);
    this->parser.add_key("upper energy window (keV)", &this->up_energy_window);
}

bool
InputStreamFromROOTFile::post_processing()
{
    stream_ptr = new TChain(this->chain_name.c_str());
    stream_ptr->Add(this->filename.c_str());

    stream_ptr->SetBranchAddress("time1", &time1);
    stream_ptr->SetBranchAddress("time2", &time2);

    stream_ptr->SetBranchAddress("eventID1",&event1);
    stream_ptr->SetBranchAddress("eventID2",&event2);
    stream_ptr->SetBranchAddress("energy1", &energy1);
    stream_ptr->SetBranchAddress("energy2", &energy2);
    stream_ptr->SetBranchAddress("comptonPhantom1", &comptonphantom1);
    stream_ptr->SetBranchAddress("comptonPhantom2", &comptonphantom2);

    least_significant_clock_bit = 5.0e+11 / least_significant_clock_bit;
    return false;
}

END_NAMESPACE_STIR
