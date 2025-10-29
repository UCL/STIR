/*
    Copyright (C) 2020-2022 University of Pennsylvania
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Implementation of class stir::CListModeDataPENN

  \author Nikos Efthimiou
*/

#include "stir/listmode/CListModeDataPENN.h"
#include "stir/listmode/CListRecordPENN.h"
#include "stir/FilePath.h"
#include "stir/error.h"
#include "stir/format.h"

START_NAMESPACE_STIR

CListModeDataPENN::CListModeDataPENN(const std::string& listmode_filename)
    : listmode_filename(listmode_filename)
{

  KeyParser parser;

  std::string originating_system;

  parser.add_start_key("PENN header");
  parser.add_stop_key("End PENN header");
  parser.add_key("originating system", &originating_system);
  parser.add_parsing_key("FileType", &this->lm_data_sptr);

  if (!parser.parse(listmode_filename.c_str()))
    error("listmode_filename: error parsing '%s'", listmode_filename.c_str());

  FilePath f(listmode_filename);

  if (lm_data_sptr->set_up() == Succeeded::no)
    error("CListModeDataPENN: Unable to set_up() from the input Header file (.hpenn).");

  shared_ptr<ExamInfo> _exam_info_sptr(new ExamInfo);

  //    Only PET scanners supported
  _exam_info_sptr->imaging_modality = ImagingModality::PT;
  _exam_info_sptr->originating_system = originating_system;
  this->exam_info_sptr = _exam_info_sptr;

  shared_ptr<Scanner> this_scanner_sptr(Scanner::get_scanner_from_name(originating_system));
  if (this_scanner_sptr->get_type() == Scanner::Unknown_scanner)
    error(format("CListModeDataPENN: Unknown value for originating_system keyword: '{}", originating_system));

  proj_data_info_sptr = ProjDataInfo::construct_proj_data_info(this_scanner_sptr,
                                                               1,
                                                               this_scanner_sptr->get_num_rings() - 1,
                                                               this_scanner_sptr->get_num_detectors_per_ring() / 2,
                                                               this_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                                               /* arc_correction*/ false
#ifdef STIR_TOF
                                                               ,
                                                               1
#endif
  );
}

std::string
CListModeDataPENN::get_name() const
{
  return listmode_filename;
}

shared_ptr<CListRecord>
CListModeDataPENN::get_empty_record_sptr() const
{
  shared_ptr<CListRecord> sptr(new CListRecordT(this->get_proj_data_info_sptr()->get_scanner_sptr()));
  return sptr;
}

Succeeded
CListModeDataPENN::get_next_record(CListRecord& record_of_general_type) const
{
  CListRecordT& record = static_cast<CListRecordT&>(record_of_general_type);
  return lm_data_sptr->get_next_record(record);
}

Succeeded
CListModeDataPENN::reset()
{
  return lm_data_sptr->reset();
}

CListModeData::SavedPosition
CListModeDataPENN::save_get_position()
{
  return static_cast<SavedPosition>(lm_data_sptr->save_get_position());
}

Succeeded
CListModeDataPENN::set_get_position(const CListModeDataPENN::SavedPosition& pos)
{
  return lm_data_sptr->set_get_position(pos);
}

void
CListModeDataPENN::set_event(const bool& is_delay,
                             const short int& _dt,
                             const unsigned short int& _xa,
                             const unsigned short int& _xb,
                             const unsigned short int& _za,
                             const unsigned short int& _zb,
                             const unsigned short int& _ea,
                             const unsigned short int& _eb)
{
  lm_data_sptr->set_new_record(is_delay, _dt, _xa, _xb, _za, _zb, _ea, _eb);
}

END_NAMESPACE_STIR
