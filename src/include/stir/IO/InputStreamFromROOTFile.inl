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

#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/shared_ptr.h"
#include "boost/shared_array.hpp"
#include <fstream>

#include <TROOT.h>
#include <TSystem.h>
#include <TChain.h>
#include <TH2D.h>
#include <TDirectory.h>
#include <TList.h>
#include <TChainElement.h>
#include <TTree.h>
#include <TFile.h>
#include <TVersionCheck.h>

START_NAMESPACE_STIR

template <class RecordT>
        InputStreamFromROOTFile<RecordT>::
        InputStreamFromROOTFile(const std::string& filename,
                                const std::string& chain_name,
                                int crystal_repeater_x, int crystal_repeater_y, int crystal_repeater_z,
                                int submodule_repeater_x, int submodule_repeater_y, int submodule_repeater_z,
                                int module_repeater_x, int module_repeater_y, int module_repeater_z,
                                int rsector_repeater,
                                bool exclude_scattered, bool exclude_randoms,
                                float low_energy_window, float up_energy_window,
                                int offset_dets)
    : filename(filename), chain_name(chain_name),
      crystal_repeater_x(crystal_repeater_x), crystal_repeater_y(crystal_repeater_y), crystal_repeater_z(crystal_repeater_z),
      submodule_repeater_x(submodule_repeater_x), submodule_repeater_y(submodule_repeater_y), submodule_repeater_z(submodule_repeater_z),
      module_repeater_x(module_repeater_x), module_repeater_y(module_repeater_y), module_repeater_z(module_repeater_z),
      rsector_repeater(rsector_repeater), exclude_scattered(exclude_scattered), exclude_randoms(exclude_randoms),
      low_energy_window(low_energy_window), up_energy_window(up_energy_window), offset_dets(offset_dets)
{

    initialize_root_read_out();
    starting_stream_position = 0;

    reset();

    half_block = module_repeater_y * submodule_repeater_y * crystal_repeater_y / 2.f  - 1;
    if (half_block < 0 )
        half_block = 0;
}

template <class RecordT>
Succeeded
InputStreamFromROOTFile<RecordT>::
initialize_root_read_out()
{

    stream_ptr = new TChain(this->chain_name.c_str());
    stream_ptr->Add(this->filename.c_str());

    stream_ptr->SetBranchAddress("crystalID1",&crystalID1);
    stream_ptr->SetBranchAddress("crystalID2",&crystalID2);
    stream_ptr->SetBranchAddress("submoduleID1",&submoduleID1);
    stream_ptr->SetBranchAddress("submoduleID2",&submoduleID2);
    stream_ptr->SetBranchAddress("moduleID1",&moduleID1);
    stream_ptr->SetBranchAddress("moduleID2",&moduleID2);
    stream_ptr->SetBranchAddress("rsectorID1",&rsectorID1);
    stream_ptr->SetBranchAddress("rsectorID2",&rsectorID2);

    //    stream_ptr->SetBranchAddress("globalPosX1",&globalPosX1);
    //    stream_ptr->SetBranchAddress("globalPosX2",&globalPosX2);
    //    stream_ptr->SetBranchAddress("globalPosY1",&globalPosY1);
    //    stream_ptr->SetBranchAddress("globalPosY2",&globalPosY2);
    //    stream_ptr->SetBranchAddress("globalPosZ1",&globalPosZ1);
    //    stream_ptr->SetBranchAddress("globalPosZ2",&globalPosZ2);
    //    stream_ptr->SetBranchAddress("sourcePosX1",&sourcePosX1);
    //    stream_ptr->SetBranchAddress("sourcePosX2",&sourcePosX2);
    //    stream_ptr->SetBranchAddress("sourcePosZ2",&sourcePosZ2);
    //    stream_ptr->SetBranchAddress("sourcePosZ1",&sourcePosZ1);
    //    stream_ptr->SetBranchAddress("sourcePosY1",&sourcePosY1);
    //    stream_ptr->SetBranchAddress("sourcePosY2",&sourcePosY2);
    stream_ptr->SetBranchAddress("time1", &time1);
    stream_ptr->SetBranchAddress("time2", &time2);

    stream_ptr->SetBranchAddress("eventID1",&eventID1);
    stream_ptr->SetBranchAddress("eventID2",&eventID2);
    //    stream_ptr->SetBranchAddress("rotationAngle",&this->rotationAngle);
    stream_ptr->SetBranchAddress("energy1", &energy1);
    stream_ptr->SetBranchAddress("energy2", &energy2);
    //    stream_ptr->SetBranchAddress("sourceID1", &this->sourceid1);
    //    stream_ptr->SetBranchAddress("sourceID2", &this->sourceid2);
    stream_ptr->SetBranchAddress("comptonPhantom1", &comptonphantom1);
    stream_ptr->SetBranchAddress("comptonPhantom2", &comptonphantom2);

    nentries = static_cast<long long int>(stream_ptr->GetEntries());
    if (nentries == 0)
        error("The total number of entries in the ROOT file is zero. Abort.");
}


template <class RecordT>
Succeeded
InputStreamFromROOTFile<RecordT>::
get_next_record(RecordT& record)
{

    while(true)
    {
        if (current_position == nentries)
            return Succeeded::no;


        if (stream_ptr->GetEntry(current_position) == 0 )
            return Succeeded::no;

        current_position ++ ;

        if ( (comptonphantom1 > 0 && comptonphantom2>0) && exclude_scattered )
            continue;
        else if ( (eventID1 != eventID2) && exclude_randoms )
            continue;
        else if (energy1 < low_energy_window ||
                 energy1 > up_energy_window ||
                 energy2 < low_energy_window ||
                 energy2 > up_energy_window)
            continue;
        else
            break;
    }

    int ring1 = static_cast<int>(crystalID1/crystal_repeater_y)
            + static_cast<int>(submoduleID1/submodule_repeater_y)*crystal_repeater_z
            + static_cast<int>(moduleID1/module_repeater_y)*submodule_repeater_z*crystal_repeater_z;

    int ring2 = static_cast<int>(crystalID2/crystal_repeater_y)
            + static_cast<int>(submoduleID2/submodule_repeater_y)*crystal_repeater_z
            + static_cast<int>(moduleID2/module_repeater_y)*submodule_repeater_z*crystal_repeater_z;

    int crystal1 = rsectorID1  * module_repeater_y * submodule_repeater_y * crystal_repeater_y
            + (moduleID1%module_repeater_y) * submodule_repeater_y * crystal_repeater_y
            + (submoduleID1%submodule_repeater_y) * crystal_repeater_y
            + (crystalID1%crystal_repeater_y);

    int crystal2 = rsectorID2 * module_repeater_y * submodule_repeater_y * crystal_repeater_y
            + (moduleID2%module_repeater_y) * submodule_repeater_y * crystal_repeater_y
            + (submoduleID2% submodule_repeater_y) * crystal_repeater_y
            + (crystalID2%crystal_repeater_y);

    // GATE counts crystal ID =0 the most negative. Therefore
    // ID = 0 should be negative, in Rsector 0 and the mid crystal ID be 0 .
    crystal1 -= half_block;
    crystal2 -= half_block;

    // Add offset
    crystal1 += offset_dets;
    crystal2 += offset_dets;

    return
            record.init_from_data(ring1, ring2,
                                  crystal1, crystal2,
                                  time1, time2,
                                  eventID1, eventID2);
}

template <class RecordT>
long long int
InputStreamFromROOTFile<RecordT>::
get_total_number_of_events()
{
    return nentries;
}

template <class RecordT>
Succeeded
InputStreamFromROOTFile<RecordT>::
reset()
{
    current_position = starting_stream_position;
    return Succeeded::yes;
}


template <class RecordT>
typename InputStreamFromROOTFile<RecordT>::SavedPosition
InputStreamFromROOTFile<RecordT>::
save_get_position()
{
    assert(current_position <= nentries);

    if(current_position < nentries)
        saved_get_positions.push_back(current_position);
    else
        saved_get_positions.push_back(-1);
    return saved_get_positions.size()-1;
}

template <class RecordT>
Succeeded
InputStreamFromROOTFile<RecordT>::
set_get_position(const typename InputStreamFromROOTFile<RecordT>::SavedPosition& pos)
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

template <class RecordT>
std::vector<long long int>
InputStreamFromROOTFile<RecordT>::
get_saved_get_positions() const
{
    return saved_get_positions;
}

template <class RecordT>
void
InputStreamFromROOTFile<RecordT>::
set_saved_get_positions(const std::vector<long long int>& poss)
{
        saved_get_positions = poss;
}

END_NAMESPACE_STIR
