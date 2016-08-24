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
    // end Scanner and physical dimentions.

    // ROOT related
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


    this->scanner_sptr.reset(Scanner::get_scanner_from_name(this->originating_system));
    if (this->scanner_sptr->get_type() == Scanner::Unknown_scanner)
    {
        error(boost::format("Unknown value for originating_system keyword: '%s ."
                              "Trying to figure out from rest input parameters") % originating_system );
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
    shared_ptr<CListRecord> sptr(new CListRecordT(this->scanner_sptr));
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
                    new InputStreamFromROOTFile<CListRecordT>(this->input_data_filename,
                                                                    this->name_of_input_tchain,
                                                                    this->number_of_crystals_x, this->number_of_crystals_y, this->number_of_crystals_z,
                                                                    this->number_of_submodules_x, this->number_of_submodules_y, this->number_of_submodules_z,
                                                                    this->number_of_modules_x, this->number_of_modules_y, this->number_of_modules_z,
                                                                    this->number_of_rsectors,
                                                                    this->exclude_scattered, this->exclude_randoms,
                                                                    this->low_energy_window*0.001f, this->up_energy_window*0.001f,
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
    return current_lm_data_ptr->set_get_position(pos);
}


END_NAMESPACE_STIR
