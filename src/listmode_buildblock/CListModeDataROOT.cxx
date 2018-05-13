/*
    Copyright (C) 2015, 2016 University of Leeds
    Copyright (C) 2016, 2017 University College London
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
  \author Harry Tsoumpas
  \author Kris Thielemans
*/

#include "stir/listmode/CListRecordROOT.h"
#include "stir/listmode/CListModeDataROOT.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include <boost/format.hpp>

START_NAMESPACE_STIR

CListModeDataROOT::
CListModeDataROOT(const std::string& hroot_filename)
    : hroot_filename(hroot_filename)
{
    this->parser.add_start_key("ROOT header");
    this->parser.add_stop_key("End ROOT header");

    // Scanner related & Physical dimensions.
    this->parser.add_key("originating system", &this->originating_system);
    this->parser.add_key("Number of rings", &this->num_rings);
    this->parser.add_key("Number of detectors per ring", &this->num_detectors_per_ring);
    this->parser.add_key("Inner ring diameter (cm)", &this->inner_ring_diameter);
    this->parser.add_key("Average depth of interaction (cm)", &this->average_depth_of_interaction);
    this->parser.add_key("Distance between rings (cm)", &this->ring_spacing);
    this->parser.add_key("Default bin size (cm)", &this->bin_size);
    this->parser.add_key("Maximum number of non-arc-corrected bins", &this->max_num_non_arccorrected_bins);
    // end Scanner and physical dimensions.

    // ROOT related
    this->parser.add_parsing_key("GATE scanner type", &this->root_file_sptr);
    if(!this->parser.parse(hroot_filename.c_str()))
        error("CListModeDataROOT: error parsing '%s'", hroot_filename.c_str());

//    this->root_file_sptr->set_up();
    // ExamInfo initialisation
    this->exam_info_sptr.reset(new ExamInfo);

    // Only PET scanners supported
    this->exam_info_sptr->imaging_modality = ImagingModality::PT;
    this->exam_info_sptr->originating_system = this->originating_system;
    this->exam_info_sptr->set_low_energy_thres(this->root_file_sptr->get_low_energy_thres());
    this->exam_info_sptr->set_high_energy_thres(this->root_file_sptr->get_up_energy_thres());

    if (this->originating_system != "User_defined_scanner")
    {
        this->scanner_sptr.reset(Scanner::get_scanner_from_name(this->originating_system));
        if (this->scanner_sptr->get_type() == Scanner::Unknown_scanner)
        {
            error(boost::format("Unknown value for originating_system keyword: '%s. Abort.") % originating_system );
        }
        warning("I've set the scanner from STIR settings and ignored values in the hroot header.");
        // TODO at least check consistency
    }
    else
    {
        info("Trying to figure out the scanner geometry from the information "
             "given in the ROOT header file.");

        this->scanner_sptr.reset(new Scanner(Scanner::User_defined_scanner,
                                             std::string ("ROOT_defined_scanner"),
                                             /* num dets per ring */
                                             this->root_file_sptr->get_num_rings(),
                                             /* num of rings */
                                             this->root_file_sptr->get_num_dets_per_ring(),
                                             /* number of non arccor bins */
                                             this->max_num_non_arccorrected_bins,
                                             /* number of maximum arccor bins */
                                             this->max_num_non_arccorrected_bins,
                                             /* inner ring radius */
                                             this->inner_ring_diameter/0.2f,
                                             /* doi */ this->average_depth_of_interaction * 10.f,
                                             /* ring spacing */
                                             this->ring_spacing * 10.f,
                                             this->bin_size * 10.f,
                                             /* offset*/ 0,
                                             /*num_axial_blocks_per_bucket_v */
                                             this->root_file_sptr->get_num_axial_blocks_per_bucket_v(),
                                             /*num_transaxial_blocks_per_bucket_v*/
                                             this->root_file_sptr->get_num_transaxial_blocks_per_bucket_v(),
                                             /*num_axial_crystals_per_block_v*/
                                             this->root_file_sptr->get_num_axial_crystals_per_block_v(),
                                             /*num_transaxial_crystals_per_block_v*/
                                             this->root_file_sptr->get_num_transaxial_crystals_per_block_v(),
                                             /*num_axial_crystals_per_singles_unit_v*/
                                             this->root_file_sptr->get_num_axial_crystals_per_singles_unit(),
                                             /*num_transaxial_crystals_per_singles_unit_v*/
                                             this->root_file_sptr->get_num_trans_crystals_per_singles_unit(),
                                             /*num_detector_layers_v*/ 1 ));
    }

    if (this->open_lm_file() == Succeeded::no)
        error("CListModeDataROOT: error opening ROOT file for filename '%s'",
              hroot_filename.c_str());
}

std::string
CListModeDataROOT::
get_name() const
{
    return hroot_filename;
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
    info(boost::format("CListModeDataROOT: used ROOT file %s") %
         this->root_file_sptr->get_ROOT_filename());
    // we actually don't do anything here. It's all done by the parsing above.

//        root_file_sptr.reset(
//                    new InputStreamFromROOTFile(this->input_data_filename,
//                                                this->name_of_input_tchain,
//                                                this->number_of_crystals_x, this->number_of_crystals_y, this->number_of_crystals_z,
//                                                this->number_of_submodules_x, this->number_of_submodules_y, this->number_of_submodules_z,
//                                                this->number_of_modules_x, this->number_of_modules_y, this->number_of_modules_z,
//                                                this->number_of_rsectors,
//                                                this->exclude_scattered, this->exclude_randoms,
//                                                static_cast<float>(this->low_energy_window*0.001f),
//                                                static_cast<float>(this->up_energy_window*0.001f),
//                                                this->offset_dets));

    return Succeeded::yes;
}



Succeeded
CListModeDataROOT::
get_next_record(CListRecord& record_of_general_type) const
{
    CListRecordROOT& record = dynamic_cast<CListRecordROOT&>(record_of_general_type);
    return root_file_sptr->get_next_record(record);
}

Succeeded
CListModeDataROOT::
reset()
{
    return root_file_sptr->reset();
}

unsigned long CListModeDataROOT::get_total_number_of_events() const
{
    return root_file_sptr->get_total_number_of_events();
}

CListModeData::SavedPosition
CListModeDataROOT::
save_get_position()
{
    return static_cast<SavedPosition>(root_file_sptr->save_get_position());
}

Succeeded
CListModeDataROOT::
set_get_position(const CListModeDataROOT::SavedPosition& pos)
{
    return root_file_sptr->set_get_position(pos);
}


END_NAMESPACE_STIR
