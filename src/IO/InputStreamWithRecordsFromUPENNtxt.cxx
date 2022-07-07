/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamWithRecordsFromUPENNtxt

  \author Nikos Efthimiou
*/
/*
 *  Copyright (C) 2020-2022 University of Pennsylvania
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/IO/InputStreamWithRecordsFromUPENNtxt.h"

START_NAMESPACE_STIR

const char * const
InputStreamWithRecordsFromUPENNtxt::registered_name =
        "UPENN_txt_listmode";

InputStreamWithRecordsFromUPENNtxt::
InputStreamWithRecordsFromUPENNtxt():
    base_type()
{
    error("The text variant of the listmode files has several pending TODOs. Don't use right now.");
    set_defaults();
}

Succeeded
InputStreamWithRecordsFromUPENNtxt::
reset()
{
    if (is_null_ptr(stream_ptr))
      return Succeeded::no;

    // Strangely enough, once you read past EOF, even seekg(0) doesn't reset the eof flag
    if (stream_ptr->eof())
      stream_ptr->clear();
    stream_ptr->seekg(starting_stream_position, std::ios::beg);
    if (stream_ptr->bad())
      return Succeeded::no;
    else
      return Succeeded::yes;
}

Succeeded
InputStreamWithRecordsFromUPENNtxt::
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
    while(true)
    {
        std::getline(*stream_ptr, *line);

        if (stream_ptr->eof())
        {
            found = false;
            break;
        }
        else if (stream_ptr->bad())
        {
            warning("InputStreamWithRecordsFromUPENNtxt: Error after reading from list mode stream in get_next_record");
            found = false;
            break;
        }
        else if (keep_prompt == true && line->at(0)=='p')
        {
            is_delay = false;
            found = true;
            break;
        }
        else if (keep_delayed == true && line->at(0)=='d')
        {
            is_delay = true;
            found = true;
            break;
        }
    }

    if(!found)
        return Succeeded::no;

    std::istringstream iss(*line);
    std::vector<std::string> results((std::istream_iterator<std::string>(iss)),
                                     std::istream_iterator<std::string>());

    dt = stoi(results[1]);

    xa = stoi(results[4]);
    xb = stoi(results[5]);

    za = stoi(results[6]);
    zb = stoi(results[7]);

    ea = stoi(results[8]);
    eb = stoi(results[9]);

    return
      record.init_from_data(is_delay,
                            dt,
                            xa, xb,
                            za, zb,
                            ea, eb);
}

void
InputStreamWithRecordsFromUPENNtxt::set_current_record()
{
    //set_record(current_record);
    error("InputStreamWithRecordsFromUPENNtxt::set_current_record not implemented, yet.");
}

void
InputStreamWithRecordsFromUPENNtxt::set_new_record(const bool& d,
                                                   const short int& _dt,
                                                   const unsigned short int& _xa, const unsigned short int& _xb,
                                                   const unsigned short int& _za, const unsigned short int& _zb,
                                                   const unsigned short int& _ea, const unsigned short int& _eb)
{
 // Create a new line and replace the old one
    error("InputStreamWithRecordsFromUPENNtxt::set_new_record not implemented, yet.");
}

Succeeded
InputStreamWithRecordsFromUPENNtxt::
set_up()
{

    return Succeeded::yes;
}

std::string
InputStreamWithRecordsFromUPENNtxt::
method_info() const
{
    return this->registered_name;
}

void
InputStreamWithRecordsFromUPENNtxt::set_defaults()
{
    starting_stream_position = 0;
}

void
InputStreamWithRecordsFromUPENNtxt::initialise_keymap()
{
    base_type::initialise_keymap();
    this->parser.add_start_key("UPENN_text_listmode Parameters");
    this->parser.add_stop_key("End UPENN_text_listmode Parameters");
}

bool
InputStreamWithRecordsFromUPENNtxt::post_processing()
{
    return false;
}

typename InputStreamWithRecordsFromUPENNtxt::SavedPosition
InputStreamWithRecordsFromUPENNtxt::
save_get_position()
{
  assert(!is_null_ptr(stream_ptr));
  // TODO should somehow check if tellg() worked and return an error if it didn't
  std::streampos pos;
  if (!stream_ptr->eof())
    {
      pos = stream_ptr->tellg();
      if (!stream_ptr->good())
    error("InputStreamWithRecords<RecordT, OptionsT>::save_get_position\n"
          "Error after getting position in file");
    }
  else
    {
      // use -1 to signify eof
      // (this is probably the behaviour of tellg anyway, but this way we're sure).
      pos = std::streampos(-1);
    }
  saved_get_positions.push_back(pos);
  return saved_get_positions.size()-1;
}

Succeeded
InputStreamWithRecordsFromUPENNtxt::
set_get_position(const typename InputStreamWithRecordsFromUPENNtxt::SavedPosition& pos)
{
  if (is_null_ptr(stream_ptr))
    return Succeeded::no;

  assert(pos < saved_get_positions.size());
  stream_ptr->clear();
  if (saved_get_positions[pos] == std::streampos(-1))
    stream_ptr->seekg(0, std::ios::end); // go to eof
  else
    stream_ptr->seekg(saved_get_positions[pos]);

  if (!stream_ptr->good())
    return Succeeded::no;
  else
    return Succeeded::yes;
}

END_NAMESPACE_STIR
