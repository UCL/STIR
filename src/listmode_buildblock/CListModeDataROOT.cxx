/*
    Copyright (C) 2015, 2016 University of Leeds
    Copyright (C) 2016 University College London
    Copyright (C) 2016, University of Hull
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
*/

#include "stir/listmode/CListModeDataROOT.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include <boost/format.hpp>
#include <fstream>
#include <cstring>

START_NAMESPACE_STIR

CListModeDataROOT::
CListModeDataROOT(const std::string& listmode_filename)
    : listmode_filename(listmode_filename)
{
    this->parser.add_start_key("ROOT header");
    this->parser.add_stop_key("End ROOT header");

    // N.E.: Compression on ROOT listmode data is commented out until further testing is done.
    //    axial_compression = -1;
    //    maximum_ring_difference = -1;
    //    number_of_projections = -1;
    //    number_of_views = -1;
    //    number_of_segments = -1;

    max_num_timing_bins = -1;
    size_timing_bin = -1.f;
    timing_resolution = -1.f;
    tof_mash_factor = -1;

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

    // Acquisition related
    // N.E.: Compression on ROOT listmode data has been commented out until further testing is done.
    //    this->parser.add_key("%axial_compression", &axial_compression);
    //    this->parser.add_key("%maximum_ring_difference", &maximum_ring_difference);
    //    this->parser.add_key("%number_of_projections", &number_of_projections);
    //    this->parser.add_key("%number_of_views", &number_of_views);
    //    this->parser.add_key("%number_of_segments", &number_of_segments);

    this->parser.add_key("number of TOF time bins", &this->max_num_timing_bins);
    this->parser.add_key("Size of timing bin (in picoseconds)", &this->size_timing_bin);
    this->parser.add_key("Timing resolution (in picoseconds)", &this->timing_resolution);

    this->parser.add_key("%TOF mashing factor", &this->tof_mash_factor);
    //

    // ROOT related
    this->parser.add_parsing_key("GATE scanner type", &this->current_lm_data_ptr);
    this->parser.parse(listmode_filename.c_str(), false /* no warnings about unrecognised keywords */);

    // ExamInfo initialisation
    this->exam_info_sptr.reset(new ExamInfo);

    // Only PET scanners supported
    this->exam_info_sptr->imaging_modality = ImagingModality::PT;
    this->exam_info_sptr->originating_system = this->originating_system;

    // When merge with master ... set the energy windows.

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

        this->scanner_sptr.reset(new Scanner(Scanner::User_defined_scanner,
                                             std::string ("ROOT_defined_scanner"),
                                             /* num dets per ring */
                                             this->current_lm_data_ptr->get_num_rings(),
                                             /* num of rings */
                                             this->current_lm_data_ptr->get_num_dets_per_ring(),
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
                                             this->current_lm_data_ptr->get_num_axial_blocks_per_bucket_v(),
                                             /*num_transaxial_blocks_per_bucket_v*/
                                             this->current_lm_data_ptr->get_num_transaxial_blocks_per_bucket_v(),
                                             /*num_axial_crystals_per_block_v*/
                                             this->current_lm_data_ptr->get_num_axial_crystals_per_block_v(),
                                             /*num_transaxial_crystals_per_block_v*/
                                             this->current_lm_data_ptr->get_num_transaxial_crystals_per_block_v(),
                                             /*num_axial_crystals_per_singles_unit_v*/
                                             this->current_lm_data_ptr->get_num_axial_crystals_per_singles_unit(),
                                             /*num_transaxial_crystals_per_singles_unit_v*/
                                             this->current_lm_data_ptr->get_num_trans_crystals_per_singles_unit(),
                                             /*num_detector_layers_v*/ 1,
                                             max_num_timing_bins,
                                             size_timing_bin,
                                             timing_resolution));
    }

    if (this->open_lm_file() == Succeeded::no)
        error("CListModeDataROOT: error opening the first listmode file for filename %s\n",
              listmode_filename.c_str());

    // N.E.: Compression on ROOT listmode data has been commented out until further testing is done.
    //    this->proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(this->scanner_sptr,
    //                                  std::max(axial_compression, 1),
    //                                  std::max(maximum_ring_difference, num_rings-1),
    //                                  std::max(number_of_views, num_detectors_per_ring/2),
    //                                  std::max(number_of_projections, max_num_non_arccorrected_bins),
    //                                  /* arc_correction*/false));

    this->proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(this->scanner_sptr,
                                  1,
                                  num_rings-1,
                                  num_detectors_per_ring/2,
                                  max_num_non_arccorrected_bins,
                                  /* arc_correction*/false));
    if (tof_mash_factor > 0)
        this->proj_data_info_sptr->set_tof_mash_factor(tof_mash_factor);
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
    info(boost::format("CListModeDataROOT: opening ROOT file %s") %
         this->current_lm_data_ptr->get_ROOT_filename());

    // Read the 4 bytes to check whether this is a ROOT file
    char mem[5]="\0";
    char sig[5]="root";

    std::ifstream t;
    t.open(this->current_lm_data_ptr->get_ROOT_filename().c_str(),  std::ios::in);

    if (t.is_open())
    {

        t.seekg(0, std::ios::beg);
        t.read(mem,4);

        if (strncmp(sig, mem, 4))
        {
            warning("CListModeDataROOT: File '%s is not a ROOT file!!'",
                    this->current_lm_data_ptr->get_ROOT_filename().c_str());
            return Succeeded::no;
        }

        t.close();
        return Succeeded::yes;

    }
    else
    {
        warning("CListModeDataROOT: cannot open file '%s'",
                this->current_lm_data_ptr->get_ROOT_filename().c_str());
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

unsigned long CListModeDataROOT::get_total_number_of_events() const
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

shared_ptr<ProjDataInfo>
CListModeDataROOT::
get_proj_data_info_sptr() const
{
    assert(!is_null_ptr(proj_data_info_sptr));
    return proj_data_info_sptr;
}


END_NAMESPACE_STIR
