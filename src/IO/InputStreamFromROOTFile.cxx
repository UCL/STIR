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
#include "stir/IO/FileSignature.h"
#include "stir/error.h"
#include "stir/FilePath.h"

START_NAMESPACE_STIR

InputStreamFromROOTFile::
InputStreamFromROOTFile()
{
    set_defaults();
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
    set_defaults();
    reset();
}

void
InputStreamFromROOTFile::set_defaults()
{
    starting_stream_position = 0;
    singles_readout_depth = -1;
    exclude_scattered = false;
    exclude_randoms = false;
    low_energy_window = 0.f;
    up_energy_window = 1000.f;
    read_optional_root_fields=false;
}

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
    this->parser.add_key("read optional ROOT fields", &this->read_optional_root_fields);
}

bool
InputStreamFromROOTFile::post_processing()
{
    return false;
}

Succeeded
InputStreamFromROOTFile::set_up(const std::string & header_path)
{

    FilePath f(filename,false);
    f.prepend_directory_name(header_path);

    const std::string fullfilename = f.get_as_string();
    // Read the 4 bytes to check whether this is a ROOT file.
    FileSignature signature(fullfilename.c_str());
    if ( strncmp(signature.get_signature(), "root", 4) != 0)
    {
        error("InputStreamFromROOTFile: File '%s' is not a ROOT file! (first 4 bytes should say 'root')",
              filename.c_str());
    }

    stream_ptr = new TChain(this->chain_name.c_str());
    stream_ptr->Add(fullfilename.c_str());

    stream_ptr->SetBranchAddress("time1", &time1);
    stream_ptr->SetBranchAddress("time2", &time2);
    stream_ptr->SetBranchAddress("axialPos",&axialPos);
    stream_ptr->SetBranchAddress("eventID1",&eventID1);
    stream_ptr->SetBranchAddress("eventID2",&eventID2);
    stream_ptr->SetBranchAddress("energy1", &energy1);
    stream_ptr->SetBranchAddress("energy2", &energy2);
    stream_ptr->SetBranchAddress("comptonPhantom1", &comptonphantom1);
    stream_ptr->SetBranchAddress("comptonPhantom2", &comptonphantom2);

    if (read_optional_root_fields)
    {
        stream_ptr->SetBranchAddress("globalPosX1",&globalPosX1);
        stream_ptr->SetBranchAddress("globalPosX2",&globalPosX2);
        stream_ptr->SetBranchAddress("globalPosY1",&globalPosY1);
        stream_ptr->SetBranchAddress("globalPosY2",&globalPosY2);
        stream_ptr->SetBranchAddress("globalPosZ1",&globalPosZ1);
        stream_ptr->SetBranchAddress("globalPosZ2",&globalPosZ2);
        stream_ptr->SetBranchAddress("rotationAngle",&rotation_angle);
        stream_ptr->SetBranchAddress("runID",&runID);
        stream_ptr->SetBranchAddress("sinogramS",&sinogramS);
        stream_ptr->SetBranchAddress("sinogramTheta",&sinogramTheta);
        stream_ptr->SetBranchAddress("sourceID1",&sourceID1);
        stream_ptr->SetBranchAddress("sourceID2",&sourceID2);
        stream_ptr->SetBranchAddress("sourcePosX1",&sourcePosX1);
        stream_ptr->SetBranchAddress("sourcePosX2",&sourcePosX2);
        stream_ptr->SetBranchAddress("sourcePosY1",&sourcePosY1);
        stream_ptr->SetBranchAddress("sourcePosY2",&sourcePosY2);
        stream_ptr->SetBranchAddress("sourcePosZ1",&sourcePosZ1);
        stream_ptr->SetBranchAddress("sourcePosZ2",&sourcePosZ2);
    }

    return Succeeded::yes;
}

END_NAMESPACE_STIR
