/*
    Copyright (C) 2020-2022 University of Pennsylvania
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Declaration of class stir::CListModeDataPENN

  \author Nikos Efthimiou
*/

#ifndef __stir_listmode_CListModeDataPENN_H__
#define __stir_listmode_CListModeDataPENN_H__

#include "stir/listmode/CListModeData.h"
#include "stir/listmode/CListRecordPENN.h"
#include "stir/IO/InputStreamWithRecordsFromUPENN.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR
//! Base class for listmode data for PENNPET Explorer scanner
class CListModeDataPENN : public CListModeData
{
public:
    //! Construct fron the filename of the Interfile header
    CListModeDataPENN(const std::string& listmode_filename_prefix);

    virtual std::string
    get_name() const;

    virtual
    shared_ptr <CListRecord> get_empty_record_sptr() const;

    virtual
    Succeeded get_next_record(CListRecord& record) const;

    virtual
    Succeeded reset();

    virtual
    SavedPosition save_get_position();

    virtual
    Succeeded set_get_position(const SavedPosition&);

    //! The safest way to get the total number of events is to count them.
    virtual inline
    unsigned long int get_total_number_of_events() const
    {
        shared_ptr<CListRecord> sptr(new CListRecordT(this->get_proj_data_info_sptr()->get_scanner_sptr()));
        CListRecordPENN& record = static_cast<CListRecordPENN&>(*sptr);

        return lm_data_sptr->get_total_number_of_events(record);
    }

    virtual bool has_delayeds() const { return true; }

//    inline const PET::ListFileHeader* get_file_header() const
//    {
//        return current_lm_data_ptr->get_file_header();
//    }

    inline void set_output_filename(const std::string ofname)
    {
        lm_data_sptr->create_output_file(ofname);
    }

    inline void set_event()
    {
        lm_data_sptr->set_current_record();
    }

    void set_event(const bool& is_delay,
                          const short int& _dt,
                          const unsigned short int& _xa, const unsigned short int& _xb,
                          const unsigned short int& _za, const unsigned short int& _zb,
                          const unsigned short int& _ea, const unsigned short int& _eb);

private:
    typedef CListRecordPENN CListRecordT;
    std::string listmode_filename;
    shared_ptr<InputStreamWithRecordsFromUPENN> lm_data_sptr;

    std::string filename;
};

END_NAMESPACE_STIR

#endif
