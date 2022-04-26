/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamFromROOTFile

  \author Nikos Efthimiou
  \author Kris Thielemans
  \author Robert Twyman
*/
/*
 *  Copyright (C) 2015, 2016 University of Leeds
    Copyright (C) 2016, 2020, 2021 UCL
    Copyright (C) 2018 University of Hull
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/IO/InputStreamFromROOTFile.h"
#include "stir/IO/FileSignature.h"
#include "stir/error.h"
#include "stir/FilePath.h"
#include "stir/info.h"

#include <TROOT.h>
#include <TSystem.h>
#include <TChain.h>
#include <type_traits>

START_NAMESPACE_STIR

InputStreamFromROOTFile::
InputStreamFromROOTFile()
{
    set_defaults();
    reset();
}

#if 0 // disabled as unused and incorrect
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
    error("This constructor is incorrect"); //TODO set_defaults() will override the above
    reset();
}
#endif

void
InputStreamFromROOTFile::set_defaults()
{
    starting_stream_position = 0;
    singles_readout_depth = -1;
    exclude_nonrandom = false;
    exclude_scattered = false;
    exclude_unscattered = false;
    exclude_randoms = false;
    check_energy_window_information = true;
    low_energy_window = 0.f;
    up_energy_window = 1000.f;
    read_optional_root_fields=false;
    crystal_repeater_x = -1;
    crystal_repeater_y = -1;
    crystal_repeater_z = -1;
    num_virtual_axial_crystals_per_block = 0;
    num_virtual_transaxial_crystals_per_block = 0;
}

void
InputStreamFromROOTFile::initialise_keymap()
{
    this->parser.add_key("name of data file", &this->filename);
    this->parser.add_key("Singles readout depth", &this->singles_readout_depth);
    this->parser.add_key("name of input TChain", &this->chain_name);
    this->parser.add_key("exclude non-random events", &this->exclude_nonrandom);
    this->parser.add_key("exclude scattered events", &this->exclude_scattered);
    this->parser.add_key("exclude unscattered events", &this->exclude_unscattered);
    this->parser.add_key("exclude random events", &this->exclude_randoms);
    this->parser.add_key("check energy window information", &this->check_energy_window_information);
#ifdef STIR_ROOT_ROTATION_AS_V4
    this->parser.add_key("offset (num of detectors)", &this->offset_dets);
#endif
    this->parser.add_key("low energy window (keV)", &this->low_energy_window);
    this->parser.add_key("upper energy window (keV)", &this->up_energy_window);
    this->parser.add_key("read optional ROOT fields", &this->read_optional_root_fields);

    this->parser.add_key("number of crystals X", &this->crystal_repeater_x);
    this->parser.add_key("number of crystals Y", &this->crystal_repeater_y);
    this->parser.add_key("number of crystals Z", &this->crystal_repeater_z);
}

bool
InputStreamFromROOTFile::post_processing()
{
    return false;
}

Succeeded
InputStreamFromROOTFile::set_up(const std::string & header_path)
{
    // check types, really should be compile time assert
    if (!std::is_same<Int_t, std::int32_t>::value ||
        !std::is_same<Float_t, float>::value ||
        !std::is_same<Double_t, double>::value)
      error("Internal error: ROOT types are not what we think they are.");

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
    // Turn off all branches
    stream_ptr->SetBranchStatus("*",0);

    // Branches are turned back on by SetBranchAddress()
    stream_ptr->SetBranchAddress("time1", &time1, &br_time1);
    stream_ptr->SetBranchAddress("time2", &time2, &br_time2);
    stream_ptr->SetBranchAddress("eventID1",&eventID1, &br_eventID1);
    stream_ptr->SetBranchAddress("eventID2",&eventID2, &br_eventID2);
    stream_ptr->SetBranchAddress("energy1", &energy1, &br_energy1);
    stream_ptr->SetBranchAddress("energy2", &energy2, &br_energy2);
    stream_ptr->SetBranchAddress("comptonPhantom1", &comptonphantom1, &br_comptonPhantom1);
    stream_ptr->SetBranchAddress("comptonPhantom2", &comptonphantom2, &br_comptonPhantom2);

    if (read_optional_root_fields)
    {
        stream_ptr->SetBranchAddress("axialPos",&axialPos, &br_axialPos);
        stream_ptr->SetBranchAddress("globalPosX1",&globalPosX1, &br_globalPosX1);
        stream_ptr->SetBranchAddress("globalPosX2",&globalPosX2, &br_globalPosX2);
        stream_ptr->SetBranchAddress("globalPosY1",&globalPosY1, &br_globalPosY1);
        stream_ptr->SetBranchAddress("globalPosY2",&globalPosY2, &br_globalPosY2);
        stream_ptr->SetBranchAddress("globalPosZ1",&globalPosZ1, &br_globalPosZ1);
        stream_ptr->SetBranchAddress("globalPosZ2",&globalPosZ2, &br_globalPosZ2);
        stream_ptr->SetBranchAddress("rotationAngle",&rotation_angle, &br_rotation_angle);
        stream_ptr->SetBranchAddress("runID",&runID, &br_runID);
        stream_ptr->SetBranchAddress("sinogramS",&sinogramS, &br_sinogramS);
        stream_ptr->SetBranchAddress("sinogramTheta",&sinogramTheta, &br_sinogramTheta);
        stream_ptr->SetBranchAddress("sourceID1",&sourceID1, &br_sourceID1);
        stream_ptr->SetBranchAddress("sourceID2",&sourceID2, &br_sourceID2);
        stream_ptr->SetBranchAddress("sourcePosX1",&sourcePosX1, &br_sourcePosX1);
        stream_ptr->SetBranchAddress("sourcePosX2",&sourcePosX2, &br_sourcePosX2);
        stream_ptr->SetBranchAddress("sourcePosY1",&sourcePosY1, &br_sourcePosY1);
        stream_ptr->SetBranchAddress("sourcePosY2",&sourcePosY2, &br_sourcePosY2);
        stream_ptr->SetBranchAddress("sourcePosZ1",&sourcePosZ1, &br_sourcePosZ1);
        stream_ptr->SetBranchAddress("sourcePosZ2",&sourcePosZ2, &br_sourcePosZ2);
    }


    {
      // Ensure that two conflicting exclusions are not applied
      if (this->exclude_nonrandom && this->exclude_randoms)
        error("InputStreamFromROOTFile: Both the exclusion of true and random events has been set. Therefore, "
              "no data will be processed.");
      if (this->exclude_scattered && this->exclude_unscattered)
        error("InputStreamFromROOTFile: Both the exclusion of scattered and unscattered events has been set. Therefore, "
              "no data will be processed.");

      // Show which event types will be unlisted based upon the exclusion criteria
      bool trues = true;
      bool randoms = true;
      bool scattered = true;
      bool scattered_randoms = true;
      if (this->exclude_nonrandom || this->exclude_unscattered)
        trues = false;
      if ( this->exclude_randoms || this->exclude_unscattered )
        randoms= false;
      if ( this->exclude_scattered || this->exclude_nonrandom )
        scattered = false;
      if ( this->exclude_scattered || this->exclude_randoms )
        scattered_randoms = false;

      std::string status = "InputStreamFromROOTFile: Processing data with the following event type inclusions:"
                           "\n  Unscattered from same eventID:        " + std::to_string(trues) +
                           "\n  Unscattered from different eventIDs:  " + std::to_string(randoms) +
                           "\n  Scattered from same eventID:          " + std::to_string(scattered) +
                           "\n  Scattered different eventIDs:         " + std::to_string(scattered_randoms);
      info(status, 2);
    }

    return Succeeded::yes;
}

bool
InputStreamFromROOTFile::check_brentry_randoms_scatter_energy_conditions(Long64_t brentry)
{
  if (brentry < 0)
    return false;

  // Scattered/unscattered event exclusion condition.
  if (this->exclude_scattered || this->exclude_unscattered) {
    GetEntryCheck(br_comptonPhantom1->GetEntry(brentry));
    GetEntryCheck(br_comptonPhantom2->GetEntry(brentry));

    // Check if either event has been Compton scattered and should be excluded
    if (this->exclude_scattered && (this->comptonphantom1 > 0 || this->comptonphantom2 > 0))
      return false;
    if (this->exclude_unscattered && (this->comptonphantom1 == 0 && this->comptonphantom2 == 0))
      return false;
  }

  // Trues/Random event exclusion condition.
  if (this->exclude_randoms || this->exclude_nonrandom) {
    GetEntryCheck(br_eventID1->GetEntry(brentry));
    GetEntryCheck(br_eventID2->GetEntry(brentry));

    // exclude randoms
    if (this->exclude_randoms && this->eventID1 != this->eventID2)
      return false;

    // exclude trues
    if (this->exclude_nonrandom && this->eventID1 == this->eventID2)
      return false;
  }

  // Energy condition.
  if (this->check_energy_window_information){
    GetEntryCheck(br_energy1->GetEntry(brentry));
    GetEntryCheck(br_energy2->GetEntry(brentry));

    // Check both energy values are within window
    if (this->get_energy1_in_keV() < this->low_energy_window ||
        this->get_energy1_in_keV() > this->up_energy_window ||
        this->get_energy2_in_keV() < this->low_energy_window ||
        this->get_energy2_in_keV() > this->up_energy_window)
    {
      return false;
    }
  }
  return true;
}


void
InputStreamFromROOTFile::set_crystal_repeater_x(int val)
{
    crystal_repeater_x = val;
}

void
InputStreamFromROOTFile::set_crystal_repeater_y(int val)
{
    crystal_repeater_y = val;
}

void
InputStreamFromROOTFile::set_crystal_repeater_z(int val)
{
    crystal_repeater_z = val;
}

void
InputStreamFromROOTFile::set_num_virtual_axial_crystals_per_block(int val)
{
  num_virtual_axial_crystals_per_block = val;
}

void
InputStreamFromROOTFile::set_num_virtual_transaxial_crystals_per_block(int val)
{
  num_virtual_transaxial_crystals_per_block = val;
}

END_NAMESPACE_STIR
