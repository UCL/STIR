/*!
  \file
  \ingroup IO
  \brief Declaration of class stir::InputStreamWithRecordsFromUPENNtxt

  \author Nikos Efthimiou

*/
/*
    Copyright (C) 2020-2022 University of Pennsylvania
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_IO_InputStreamWithRecordsFromUPENNtxt_H__
#define __stir_IO_InputStreamWithRecordsFromUPENNtxt_H__

#include "stir/RegisteredParsingObject.h"
#include "stir/IO/InputStreamWithRecordsFromUPENN.h"

START_NAMESPACE_STIR

/*!
  \brief Class for reading listmode files in text format from the PENNPet Explorer scanner.

  \ingroup IO

  \todo write functions

  */

class InputStreamWithRecordsFromUPENNtxt : public
        RegisteredParsingObject< InputStreamWithRecordsFromUPENNtxt ,
        InputStreamWithRecordsFromUPENN,
        InputStreamWithRecordsFromUPENN >
{
private:
    typedef RegisteredParsingObject< InputStreamWithRecordsFromUPENNtxt,
    InputStreamWithRecordsFromUPENN,
    InputStreamWithRecordsFromUPENN > base_type;

public:

    static const char * const registered_name;

    InputStreamWithRecordsFromUPENNtxt();

    virtual ~InputStreamWithRecordsFromUPENNtxt() {}

    virtual inline
    Succeeded create_output_file(const std::string ofilename);
    //! Must be called before calling for the first event.
    virtual Succeeded set_up();
    //! gives method information
    virtual std::string method_info() const;

    virtual Succeeded get_next_record(CListRecordPENN& record);

    //! go back to starting position
    virtual
    Succeeded reset();

    //! save current "get" position in an internal array
    /*! \return an "index" into the array that allows you to go back.
      \see set_get_position
  */
    virtual
    SavedPosition save_get_position();

    //! set current "get" position to previously saved value
    virtual
    Succeeded set_get_position(const SavedPosition&);

    inline
    std::istream& get_stream(){return *this->stream_ptr;}

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
    shared_ptr<std::istream> stream_ptr = nullptr;

    std::streampos starting_stream_position;

    std::vector<std::streampos> saved_get_positions;

    shared_ptr<std::string> line;
};

END_NAMESPACE_STIR

#include "stir/IO/InputStreamWithRecordsFromUPENNtxt.inl"

#endif
