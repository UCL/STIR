//
//
/*!
  \file
  \ingroup IO
  \brief Declaration of class stir::InputStreamWithRecordsFromUPENNbin

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
#ifndef __stir_IO_InputStreamWithRecordsFromUPENNbin_H__
#define __stir_IO_InputStreamWithRecordsFromUPENNbin_H__

#include "stir/IO/InputStreamWithRecordsFromUPENN.h"
#include "stir/RegisteredParsingObject.h"

#include "/autofs/space/celer_001/users/nikos/src/UPENN/penn/include/liblist.h"
#include "/autofs/space/celer_001/users/nikos/src/UPENN/penn/include/libmhdr.h"

START_NAMESPACE_STIR

class InputStreamWithRecordsFromUPENNbin : public
        RegisteredParsingObject< InputStreamWithRecordsFromUPENNbin ,
        InputStreamWithRecordsFromUPENN,
        InputStreamWithRecordsFromUPENN >
{
private:
    typedef RegisteredParsingObject< InputStreamWithRecordsFromUPENNbin,
    InputStreamWithRecordsFromUPENN,
    InputStreamWithRecordsFromUPENN > base_type;

public:

    static const char * const registered_name;

    InputStreamWithRecordsFromUPENNbin();

    virtual ~InputStreamWithRecordsFromUPENNbin() {
        delete in;
        delete eventCodec;
    }

    virtual inline
    Succeeded create_output_file(const std::string ofilename);
    //! Must be called before calling for the first event.
    virtual Succeeded set_up();
    //! gives method information
    virtual std::string method_info() const;

    virtual Succeeded get_next_record(CListRecordPENN& record);
    //! go back to starting position
    virtual Succeeded reset();

    virtual
    SavedPosition save_get_position();

    virtual
    Succeeded set_get_position(const SavedPosition&);

    inline
    std::streambuf& get_stream(){/*return &this->inputList;*/}

    const PET::ListFileHeader* get_file_header() const;

    virtual void set_current_record();

    virtual void set_new_record(const bool& d,
                                const short int& _dt,
                                const unsigned short int& _xa, const unsigned short int& _xb,
                                const unsigned short int& _za, const unsigned short int& _zb,
                                const unsigned short int& _ea, const unsigned short int& _eb);
protected:

    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();


private:

    void set_record(const uint8_t* e);

    list::EventFormat eventFormat;
    list::EventCodec* eventCodec = nullptr;
    list::InputBuffer *in = nullptr;
    PET::ListFileHeader listHeader;
    std::filebuf inputListFile;
    std::streambuf *inputList;// = std::cin.rdbuf();
    //  shared_ptr<list::EventFormat> eventFormat;
    int pos;
    list::EventCodec* olistCodec = nullptr;
    std::filebuf outputListFile;
    std::streambuf* outputList;
    list::OutputBuffer* out = nullptr;
    long int abrupt_counter = 0;
    std::streampos starting_stream_position;
    std::vector<std::streampos> saved_get_positions;

    //    uint8_t current_record[8];
    const uint8_t* current_record;

    int eventSize = 0;
    int timeout = 0;

    bool has_output = false;
    bool abrupt = false;
};

END_NAMESPACE_STIR

#include "stir/IO/InputStreamWithRecordsFromUPENNbin.inl"

#endif
