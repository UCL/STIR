//
//
/*!
  \file
  \ingroup IO
  \brief Declaration of class stir::InputStreamWithRecordsFromUPENNtxt

  \author Nikos Efthimiou

*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
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

#ifndef __stir_IO_InputStreamWithRecordsFromUPENNtxt_H__
#define __stir_IO_InputStreamWithRecordsFromUPENNtxt_H__

#include "stir/RegisteredParsingObject.h"
#include "stir/IO/InputStreamWithRecordsFromUPENN.h"

START_NAMESPACE_STIR

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
    shared_ptr<std::istream> stream_ptr;
    const std::string filename;
    std::streampos starting_stream_position;
    std::vector<std::streampos> saved_get_positions;

    shared_ptr<std::string> line;
};

END_NAMESPACE_STIR

#include "stir/IO/InputStreamWithRecordsFromUPENNtxt.inl"

#endif


/*
 *   inline
  unsigned long int get_total_number_events(CListRecordPENN& record)
  {
    reset();
//    return current_lm_data_ptr->get_next_record(record);
    unsigned long int counter = 0;

    while(true)
    {
        if(get_next_record(record) == Succeeded::no)
            break;
        if (counter > 1 && counter%1000000L==0)
//            info( boost::format("Counting records: %1% ") % counter);
            std::cout << "\r" << counter << " events counted "<<std::flush;
        counter++;
    }

    reset();
    return counter;
  }
  */
