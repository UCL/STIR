/*
    Copyright (C) 2003-2012 Hammersmith Imanet Ltd
    Copyright (C) 2016 University College London
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
  \ingroup listmode
  \brief Implementation of class stir::CListModeDataROOT

  \author Nikos Efthimiou
*/

#include "stir/listmode/CListModeDataROOT.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include <boost/format.hpp>
#include <fstream>
#include <sstream>

START_NAMESPACE_STIR

CListModeDataROOT::
CListModeDataROOT(const std::string& listmode_filename)
    : listmode_filename(listmode_filename)
{
    this->parser.add_start_key("ROOT header");
    this->parser.add_stop_key("End ROOT header");

    this->parser.add_key("name of data file", &this->input_data_filename);

    // Scanner related & Physical dimentions.
    this->parser.add_key("originating system", &this->originating_system);
    this->parser.add_key("Number of rings", &this->num_rings);
    this->parser.add_key("Number of detectors per ring", &this->num_detectors_per_ring);
    this->parser.add_key("Inner ring diameter (cm)", &this->inner_ring_diameter);
    this->parser.add_key("Average depth of interaction (cm)", &this->average_depth_of_interaction);
    this->parser.add_key("Distance between rings (cm)", &this->ring_spacing);
    this->parser.add_key("Default bin size (cm)", &this->bin_size);
    this->parser.add_key("Maximum number of non-arc-corrected bins", &this->max_num_non_arccorrected_bins);
    // end Scanner and physical dimentions.

    // ROOT related
    this->parser.add_key("Singles readout depth", &this->singles_readout_depth);
    this->parser.add_key("number of Rsectors", &this->number_of_rsectors);
    this->parser.add_key("number of modules X", &this->number_of_modules_x);
    this->parser.add_key("number of modules Y", &this->number_of_modules_y);
    this->parser.add_key("number of modules Z", &this->number_of_modules_z);

    this->parser.add_key("number of submodules X", &this->number_of_submodules_x);
    this->parser.add_key("number of submodules Y", &this->number_of_submodules_y);
    this->parser.add_key("number of submodules Z", &this->number_of_submodules_z);

    this->parser.add_key("number of crystals X", &this->number_of_crystals_x);
    this->parser.add_key("number of crystals Y", &this->number_of_crystals_y);
    this->parser.add_key("number of crystals Z", &this->number_of_crystals_z);

    this->parser.add_key("name of input TChain", &this->name_of_input_tchain);
    this->parser.add_key("exclude scattered events", &this->exclude_scattered);
    this->parser.add_key("exclude random events", &this->exclude_randoms);
    this->parser.add_key("offset (num of detectors)", &this->offset_dets);
    // end ROOT related

    // Acquisition related
    this->parser.add_key("low energy window (keV)", &this->low_energy_window);
    this->parser.add_key("upper energy window (keV)", &this->up_energy_window);
    // end acquisirion related

    this->parser.parse(listmode_filename.c_str(), false /* no warnings about unrecognised keywords */);
    // ExamInfo initialisation
    this->exam_info_sptr.reset(new ExamInfo);

    // Only PET scanners supported
    this->exam_info_sptr->imaging_modality = ImagingModality::PT;
    this->exam_info_sptr->originating_system = this->originating_system;

    if (this->originating_system != "User_defined_scanner")
    {
        this->scanner_sptr.reset(Scanner::get_scanner_from_name(this->originating_system));
        if (this->scanner_sptr->get_type() == Scanner::Unknown_scanner)
        {
            error(boost::format("Unknown value for originating_system keyword: '%s. Abort.") % originating_system );
        }
    }
    else
    {
        info(boost::format("Trying to figure out the scanner geometry from the information"
                           "given in the ROOT header file."));

        int axial_crystals_per_singles = 0;
        int trans_crystals_per_singles = 0;

        if (this->singles_readout_depth == 1) // One PMT per Rsector
        {
            axial_crystals_per_singles = this->number_of_crystals_z *
                    this->number_of_modules_z *
                    this->number_of_submodules_z;
            trans_crystals_per_singles = this->number_of_modules_y *
                    this->number_of_submodules_y *
                    this->number_of_crystals_y;
        }
        else if (this->singles_readout_depth == 2) // One PMT per module
        {
            axial_crystals_per_singles = this->number_of_crystals_z *
                    this->number_of_submodules_z;
            trans_crystals_per_singles = this->number_of_submodules_y *
                    this->number_of_crystals_y;
        }
        else if (this->singles_readout_depth == 3) // One PMT per submodule
        {
            axial_crystals_per_singles = this->number_of_crystals_z;
            trans_crystals_per_singles = this->number_of_crystals_y;
        }
        else if (this->singles_readout_depth == 4) // One PMT per crystal
        {
            axial_crystals_per_singles = 1;
            trans_crystals_per_singles = 1;
        }
        else
            error(boost::format("Singles readout depth (%1%) is invalid") % this->singles_readout_depth);


        this->scanner_sptr.reset(new Scanner(Scanner::User_defined_scanner,
                                             std::string ("ROOT_defined_scanner"),
                                             /* num dets per ring */
                                             int(  this->number_of_rsectors * this->number_of_modules_y *
                                                   this->number_of_submodules_y * this->number_of_crystals_y),
                                             /* num of rings */
                                             int( this->number_of_crystals_z * this->number_of_modules_z *
                                                  this->number_of_submodules_z),
                                             /* number of non arccor bins */
                                             this->max_num_non_arccorrected_bins,
                                             /* number of maximum arccor bins */
                                             this->max_num_non_arccorrected_bins,
                                             /* inner ring radius */
                                             this->inner_ring_diameter/0.2f,
                                             /* doi */
                                             this->average_depth_of_interaction * 10.f,
                                             /* ring spacing */
                                             this->ring_spacing * 10.f,
                                             this->bin_size * 10.f,
                                             /* offset*/
                                             this->offset_dets,
                                             /*num_axial_blocks_per_bucket_v */
                                             this->number_of_modules_z,
                                             /*num_transaxial_blocks_per_bucket_v*/
                                             this->number_of_modules_y,
                                             /*num_axial_crystals_per_block_v*/
                                             int(this->number_of_crystals_z * this->number_of_submodules_z),
                                             /*num_transaxial_crystals_per_block_v*/
                                             int(this->number_of_crystals_y * this->number_of_submodules_y),
                                             /*num_axial_crystals_per_singles_unit_v*/
                                             axial_crystals_per_singles,
                                             /*num_transaxial_crystals_per_singles_unit_v*/
                                             trans_crystals_per_singles,
                                             /*num_detector_layers_v*/
                                             1));
    }

    if (this->open_lm_file() == Succeeded::no)
        error("CListModeDataROOT: error opening the first listmode file for filename %s\n",
              listmode_filename.c_str());
}

std::string
CListModeDataROOT::
get_name() const
{
    return listmode_filename;
}


shared_ptr <CListRecord>
CListModeDataROOT::
get_empty_record_sptr() const
{
    shared_ptr<CListRecord> sptr(new CListRecordROOT(this->scanner_sptr));
    return sptr;
}

Succeeded
CListModeDataROOT::
open_lm_file()
{
    info(boost::format("CListModeDataROOT: opening ROOT file %s") % this->input_data_filename);

    // Read the 4 bytes to check whether this is a ROOT file, indeed.
    // I could rewrite it in a more sofisticated way ...
    std::stringstream ss;
    char mem[4]="\0";
    std::string sig= "root";

    std::ifstream t;
    t.open(this->input_data_filename.c_str(),  std::ios::in |std::ios::binary);

    if (t.is_open())
    {

        t.seekg(0, std::ios::beg);
        t.read(mem,4);
        ss << mem;

        if ( !sig.compare(ss.str()) )
        {
            warning("CListModeDataROOT: File '%s is not a ROOT file!!'", this->input_data_filename.c_str());
            return Succeeded::no;
        }

        current_lm_data_ptr.reset(
                    new InputStreamFromROOTFile(this->input_data_filename,
                                                this->name_of_input_tchain,
                                                this->number_of_crystals_x, this->number_of_crystals_y, this->number_of_crystals_z,
                                                this->number_of_submodules_x, this->number_of_submodules_y, this->number_of_submodules_z,
                                                this->number_of_modules_x, this->number_of_modules_y, this->number_of_modules_z,
                                                this->number_of_rsectors,
                                                this->exclude_scattered, this->exclude_randoms,
                                                static_cast<float>(this->low_energy_window*0.001f),
                                                static_cast<float>(this->up_energy_window*0.001f),
                                                this->offset_dets));
        t.close();
        return Succeeded::yes;

    }
    else
    {
        warning("CListModeDataROOT: cannot open file '%s'", this->input_data_filename.c_str());
        return Succeeded::no;
    }
}



Succeeded
CListModeDataROOT::
get_next_record(CListRecord& record_of_general_type) const
{
    CListRecordROOT& record = dynamic_cast<CListRecordROOT&>(record_of_general_type);
    return current_lm_data_ptr->get_next_record(record);
}

Succeeded
CListModeDataROOT::
reset()
{
    return current_lm_data_ptr->reset();
}

long long int
CListModeDataROOT::
get_total_number_of_events() const
{
    return current_lm_data_ptr->get_total_number_of_events();
}

CListModeData::SavedPosition
CListModeDataROOT::
save_get_position()
{
    return static_cast<SavedPosition>(current_lm_data_ptr->save_get_position());
}

Succeeded
CListModeDataROOT::
set_get_position(const CListModeDataROOT::SavedPosition& pos)
{
    return current_lm_data_ptr->set_get_position(pos);
}


END_NAMESPACE_STIR
