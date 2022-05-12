/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamWithRecordsFromUPENNbin

  \author Nikos Efthimiou
*/
/*
 *  Copyright (C) 2020-2022 University of Pennsylvania
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/IO/InputStreamWithRecordsFromUPENNbin.h"

START_NAMESPACE_STIR

const char * const
InputStreamWithRecordsFromUPENNbin::registered_name =
        "UPENN_binary_listmode";

InputStreamWithRecordsFromUPENNbin::
InputStreamWithRecordsFromUPENNbin():
    base_type()
{
    set_defaults();
}

Succeeded
InputStreamWithRecordsFromUPENNbin::
reset()
{
    inputListFile.pubseekpos(pos);
    inputList->pubsync();

    abrupt_counter = N;
    if(!is_null_ptr(in))
    {
        delete in;
        in = new list::InputBuffer( *inputList, eventSize );
    }
    return Succeeded::yes;
}

Succeeded
InputStreamWithRecordsFromUPENNbin::
get_next_record(CListRecordPENN& record)
{
    int dt = 0;
    int xa = 0;
    int xb = 0;
    int za = 0;
    int zb = 0;
    int ea = 0;
    int eb = 0;
    bool is_delay = false;
    bool found = false;

#ifdef STIR_OPENMP
#pragma omp critical(LISTMODEIO)
#endif
    while (in->next())
    {
        list::Event event( *eventCodec, in->data() );
        if(abrupt_stop)
        {
            abrupt_counter--;
            if (abrupt_counter < 0)
            {
                found = false;
                break;
            }
        }

        if(!event.isData() || event.isControl())
        {
            timeout = 0;
            continue;
        }
        else if(event.ea()<low_energy_window || event.ea() > up_energy_window ||
                event.eb()<low_energy_window || event.eb() > up_energy_window)
        {
            timeout = 0;
            continue;
        }
        else if(keep_prompt == true && event.isPrompt())
        {
            timeout = 0;
            dt = event.dt();
            xa = event.xa();
            xb = event.xb();
            za = event.za();
            zb = event.zb();
            ea = event.ea();
            eb = event.eb();
            is_delay = false;
            if (xa != xb)
            {

                found = true;
                break;
            }else {
                continue;
            }
        }
        else if (keep_delayed == true && event.isDelay())
        {
            timeout = 0;

            dt = event.dt();
            xa = event.xa();
            xb = event.xb();
            za = event.za();
            zb = event.zb();
            ea = event.ea();
            eb = event.eb();
            is_delay = true;
            if (xa != xb)
            {
                found = true;
                break;
            }else {
                continue;
            }
        }
        else {
            continue;
        }
    }

    if(found)
    {
        if(abrupt_stop)
            if(abrupt_counter < 0)
                return Succeeded::no;
        return
                record.init_from_data(is_delay,
                                      dt,
                                      xa, xb,
                                      za, zb,
                                      ea, eb
                                      );
    }
    else {
        return Succeeded::no;
    }

}

std::string
InputStreamWithRecordsFromUPENNbin::
method_info() const
{
    std::ostringstream s;
    s << this->registered_name;
    return s.str();
}

void
InputStreamWithRecordsFromUPENNbin::set_record(const uint8_t* e)
{
#ifdef STIR_OPENMP
#pragma omp critical(LISTMODEIO)
#endif
    {
        out->put(e);
    }
}

void
InputStreamWithRecordsFromUPENNbin::set_current_record()
{
    set_record(current_record);
}

void
InputStreamWithRecordsFromUPENNbin::set_new_record(const bool& d,
                                                   const short int& _dt,
                                                   const unsigned short int& _xa, const unsigned short int& _xb,
                                                   const unsigned short int& _za, const unsigned short int& _zb,
                                                   const unsigned short int& _ea, const unsigned short int& _eb)
{
    //transformations
    list::EventCodec out_event(eventFormat);

    int la = 0, lb = 0;
    std::uint8_t e[8];
    out_event.init(e, d,
                   static_cast<int>(_dt),
                   la, lb,
                   static_cast<int>(_xa), static_cast<int>(_xb),
                   static_cast<int>(_za), static_cast<int>(_zb),
                   static_cast<int>(_ea), static_cast<int>(_eb));

    set_record(e);
    delete e;
}

Succeeded
InputStreamWithRecordsFromUPENNbin::
set_up()
{
    if ( !this->filename.empty() )
    {
        if ( !inputListFile.open( filename.c_str(), std::ios_base::in | std::ios_base::binary ) )
        {
            error("cannot open file " + filename);
        }

        inputList = &inputListFile;
    }

    if ( !list::decodeHeader( *inputList, &listHeader )
         || !list::isCoincListHeader( listHeader ) )
    {
        error("cannot read valid header from input list");
    }

    eventFormat = list::eventFormat( listHeader );
    eventCodec = new list::EventCodec( eventFormat );
    eventSize = list::eventSize( eventFormat );
    in = new list::InputBuffer( *inputList, eventSize );
    pos = inputListFile.pubseekoff(0, std::ios_base::cur);

    return Succeeded::yes;
}

const PET::ListFileHeader*
InputStreamWithRecordsFromUPENNbin::get_file_header() const
{
    return &listHeader;
}


void
InputStreamWithRecordsFromUPENNbin::set_defaults()
{
    base_type::set_defaults();
    starting_stream_position = 0;
}

typename InputStreamWithRecordsFromUPENNbin::SavedPosition
InputStreamWithRecordsFromUPENNbin::
save_get_position()
{
    int pos = inputListFile.pubseekoff(0, std::ios_base::cur);
    saved_get_positions.push_back(pos);
    return saved_get_positions.size()-1;
}


Succeeded
InputStreamWithRecordsFromUPENNbin::
set_get_position(const typename InputStreamWithRecordsFromUPENNbin::SavedPosition& pos)
{
    assert(pos < saved_get_positions.size());
    inputListFile.pubseekoff(saved_get_positions[pos], std::ios_base::beg);

    return Succeeded::yes;
}

void
InputStreamWithRecordsFromUPENNbin::initialise_keymap()
{
    base_type::initialise_keymap();
    this->parser.add_start_key("UPENN_binary_listmode Parameters");
    this->parser.add_stop_key("End UPENN_binary_listmode Parameters");
}

bool
InputStreamWithRecordsFromUPENNbin::post_processing()
{
    if (base_type::post_processing())
        return true;
    return false;
}

END_NAMESPACE_STIR
