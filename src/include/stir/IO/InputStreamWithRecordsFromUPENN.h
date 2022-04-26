//
//
/*!
  \file
  \ingroup IO
  \brief Declaration of class stir::InputStreamWithRecordsFromUPENN

  \author Nikos Efthimiou

*/
/*
    Copyright (C) 2020-2022 University of Pennsylvania
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
#ifndef __stir_IO_InputStreamWithRecordsFromUPENN_H__
#define __stir_IO_InputStreamWithRecordsFromUPENN_H__

#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "stir/RegisteredObject.h"
#include "stir/listmode/CListRecordPENN.h"

START_NAMESPACE_STIR

class InputStreamWithRecordsFromUPENN: public RegisteredObject< InputStreamWithRecordsFromUPENN >
{
public:
    typedef std::vector<long long int>::size_type SavedPosition;

    InputStreamWithRecordsFromUPENN();

    unsigned long int get_total_number_of_events(CListRecordPENN& record);

    virtual inline Succeeded
    create_output_file(const std::string ofilename) = 0;

    virtual ~InputStreamWithRecordsFromUPENN() {}
    //! Must be called before calling for the first event.
    virtual Succeeded set_up();

    virtual
    Succeeded get_next_record(CListRecordPENN& record) = 0;

    //! go back to starting position
    virtual Succeeded reset();

    virtual SavedPosition save_get_position() = 0;

    virtual
    Succeeded set_get_position(const SavedPosition&) = 0;

    inline
    std::vector<std::streampos> get_saved_get_positions() const;

    inline
    void set_saved_get_positions(const std::vector<std::streampos>& );

    inline
    std::streambuf& get_stream(){/*return &this->inputList;*/}

//    virtual const PET::ListFileHeader* get_file_header() const = 0;

    virtual void set_current_record() = 0;


    virtual void set_new_record(const bool& d,
                               const short int& _dt,
                               const unsigned short int& _xa, const unsigned short int& _xb,
                               const unsigned short int& _za, const unsigned short int& _zb,
                               const unsigned short int& _ea, const unsigned short int& _eb) = 0;

protected:
    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();

    std::string filename;
    std::streampos starting_stream_position;
    std::vector<std::streampos> saved_get_positions;

    //    uint8_t current_record[8];
    const uint8_t* current_record;
    // 0: all
    // 1: prompts
    // 2: delay
//    int keep_type;

    int eventSize = 0;

    int low_energy_window = 0;
    int up_energy_window = 1000;

    int timeout = 0;
    //! Total number of events
    long unsigned int N;
    bool has_output = false;
    bool abrupt = false;
    //! Minimum energy channel
    int minE_chan;
    //! Maximum energy channel
    int maxE_chan;
    //! In some processes it might be better to filter prompts and delayeds
    //! early.
    bool keep_prompt;
    //! In some processes it might be better to filter prompts and delayeds
    //! early.
    bool keep_delayed;
};

END_NAMESPACE_STIR

#include "stir/IO/InputStreamWithRecordsFromUPENN.inl"

#endif
