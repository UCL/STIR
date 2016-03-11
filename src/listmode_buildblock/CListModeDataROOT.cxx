/*
    Copyright (C) 2003-2012 Hammersmith Imanet Ltd
    Copyright (C) 2013-2014 University College London
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
//#include "stir/listmode/CListRecordROOT.h"
#include "stir/ExamInfo.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include <boost/format.hpp>
#include <iostream>
#include <fstream>

START_NAMESPACE_STIR

//!
//! \brief CListModeDataROOT::CListModeDataROOT
//! \param listmode_filename
//! \todo We should be able to import strings, in order to give the option
//! to import from different coincidence chains.
//!
CListModeDataROOT::
CListModeDataROOT(const std::string& listmode_filename)
    : listmode_filename(listmode_filename)
{
    this->interfile_parser.add_key("%axial_compression", &this->axial_compression);
    this->interfile_parser.add_key("%maximum_ring_difference", &this->maximum_ring_difference);
    this->interfile_parser.add_key("%number_of_projections", &this->number_of_projections);
    this->interfile_parser.add_key("%number_of_views", &this->number_of_views);
    this->interfile_parser.add_key("%number_of_segments", &this->number_of_segments);

    this->interfile_parser.add_key("number of Rsectors", &this->number_of_rsectors);
    this->interfile_parser.add_key("number of modules X", &this->number_of_modules_x);
    this->interfile_parser.add_key("number of modules Y", &this->number_of_modules_y);
    this->interfile_parser.add_key("number of modules Z", &this->number_of_modules_z);

    this->interfile_parser.add_key("number of submodules X", &this->number_of_submodules_x);
    this->interfile_parser.add_key("number of submodules Y", &this->number_of_submodules_y);
    this->interfile_parser.add_key("number of submodules Z", &this->number_of_submodules_z);

    this->interfile_parser.add_key("number of crystals X", &this->number_of_crystals_x);
    this->interfile_parser.add_key("number of crystals Y", &this->number_of_crystals_y);
    this->interfile_parser.add_key("number of crystals Z", &this->number_of_crystals_z);

    //    this->interfile_parser.add_key("%name_of_input_tchain", &this->name_of_input_tchain);
    // TODO add overload to KeyParser such that we can read std::vector<int>
    // However, we would need the value of this keyword only for verification purposes, so we don't read it for now.
    // this->interfile_parser.add_key("segment_table", &this->segment_table);

#if 0
    // at the moment, we fix the header to confirm to "STIR interfile" as opposed to
    // "Siemens interfile", so don't enable the following.
    // It doesn't work properly anyway, as the "image duration (sec)" keyword
    // isn't "vectored" in Siemens headers (i.e. it doesn't have "[1]" appended).
    // As stir::KeyParser currently doesn't know if a keyword is vectored or not,
    // it causes memory overwrites if you use the wrong one.

    // We need to set num_time_frames to 1 as the Siemens header doesn't have the num_time_frames keyword
    {
        const int num_time_frames=1;
        this->interfile_parser.num_time_frames=1;
        this->interfile_parser.image_scaling_factors.resize(num_time_frames);
        for (int i=0; i<num_time_frames; i++)
            this->interfile_parser.image_scaling_factors[i].resize(1, 1.);
        this->interfile_parser.data_offset.resize(num_time_frames, 0UL);
        this->interfile_parser.image_relative_start_times.resize(num_time_frames, 0.);
        this->interfile_parser.image_durations.resize(num_time_frames, 0.);
    }
#endif

    this->interfile_parser.parse(listmode_filename.c_str(), false /* no warnings about unrecognised keywords */);

    this->exam_info_sptr.reset(new ExamInfo(*interfile_parser.get_exam_info_ptr()));

    const std::string originating_system(this->interfile_parser.get_exam_info_ptr()->originating_system);
    this->scanner_sptr.reset(Scanner::get_scanner_from_name(originating_system));
    if (this->scanner_sptr->get_type() == Scanner::Unknown_scanner)
        error(boost::format("Unknown value for originating_system keyword: '%s") % originating_system );


    this->proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(this->scanner_sptr,
                                                                  this->axial_compression,
                                                                  this->maximum_ring_difference,
                                                                  this->number_of_views,
                                                                  this->number_of_projections,
                                                                  /* arc_correction*/false));

    if (this->open_lm_file() == Succeeded::no)
        error("CListModeDataROOT: error opening the first listmode file for filename %s\n",
              listmode_filename.c_str());

    if (this->check_scanner_consistency() == Succeeded::no)
        error("It appears as the information on GATE's repeaters in not concistent to the scanner's template");
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
    shared_ptr<CListRecord> sptr(new CListRecordT(this->proj_data_info_sptr));
    return sptr;
}

//!
//! \brief CListModeDataROOT::check_scanner_consistency
//! \return
//! \details Quite simple functions in order to avoid index_out_of_range errors, later.
//! \author Nikos Efthimiou
//!
Succeeded
CListModeDataROOT::
check_scanner_consistency()
{

    // Check that the repeaters would give the correct number of rings
    int  number_of_rings = 0;

    if (number_of_crystals_z > 1)
        number_of_rings += number_of_crystals_z;

    if (number_of_submodules_z > 1 )
        number_of_rings *= number_of_submodules_z;

    if (number_of_modules_z > 1 )
        number_of_rings *= number_of_modules_z;

    if (number_of_rings != proj_data_info_sptr->get_scanner_ptr()->get_num_rings())
        return Succeeded::no;

    // Check that the repeaters would give the correct number of detectors
    // per ring.
    int   number_of_detectors = 0;

    if (number_of_crystals_y > 1)
        number_of_detectors += number_of_crystals_y;

    if (number_of_submodules_y > 1 )
        number_of_detectors *= number_of_submodules_y;

    if (number_of_modules_y > 1 )
        number_of_detectors *= number_of_modules_y;

    if (number_of_rsectors > 1 )
        number_of_detectors *= number_of_rsectors;

    if (number_of_detectors !=
            proj_data_info_sptr->get_scanner_ptr()->get_num_detectors_per_ring())
        return Succeeded::no;

    return Succeeded::yes;
}

//!
//! \brief CListModeDataROOT::open_lm_file
//! \return
//! \author Nikos Efthimiou
//! \details Opens a root file
//! \todo The check of the filetype could be better.
Succeeded
CListModeDataROOT::
open_lm_file()
{
    std::string filename = interfile_parser.data_file_name;
    info(boost::format("CListModeDataROOT: opening ROOT file %s") % filename);

    // Read the 4 bytes to check whether this is a ROOT file, indeed.
    // I could rewrite it in a more sofisticated way ...

    char mem[4]="\0";
    char sig[] = "root";

    std::ifstream t;
    t.open(filename.c_str(),  std::ios::in |std::ios::binary);

    if (t.is_open())
    {

        t.seekg(0, std::ios::beg);
        t.read(mem,4);

        if ( strcmp(mem, sig) != 0)
        {
            warning("CListModeDataROOT: File '%s is not a ROOT file!!'", filename.c_str());
            return Succeeded::no;
        }

        current_lm_data_ptr.reset(
                    new InputStreamFromROOTFile<CListRecordT, bool>(filename,
                                                                    number_of_crystals_x, number_of_crystals_y, number_of_crystals_z,
                                                                    number_of_submodules_x, number_of_submodules_y, number_of_submodules_z,
                                                                    number_of_modules_x, number_of_modules_y, number_of_modules_z,
                                                                    number_of_rsectors));
        t.close();
        return Succeeded::yes;

    }
    else
    {
        warning("CListModeDataROOT: cannot open file '%s'", filename.c_str());
        return Succeeded::no;
    }
}



Succeeded
CListModeDataROOT::
get_next_record(CListRecord& record_of_general_type) const
{
    CListRecordT& record = static_cast<CListRecordT&>(record_of_general_type);
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
    return
            current_lm_data_ptr->set_get_position(pos);
}


END_NAMESPACE_STIR
