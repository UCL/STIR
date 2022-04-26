/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamWithRecordsFromUPENN

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

//#ifdef STIR_OPENMP
//#pragma omp critical(LISTMODEIO)
//#endif
    while(true)
    {
        std::getline(*stream_ptr, *line);

        if (stream_ptr->eof())
            return Succeeded::no;
        else if (stream_ptr->bad())
        {
            warning("InputStreamWithRecordsFromUPENNtxt: Error after reading from list mode stream in get_next_record");
            return Succeeded::no;
        }
        else if (keep_prompt == true && line->at(0)=='p')
        {
            is_delay = false;
            break;
        }
        else if (keep_delayed == true && line->at(0)=='d')
        {
            is_delay = true;
            break;
        }
    }

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
}

void
InputStreamWithRecordsFromUPENNtxt::set_new_record(const bool& d,
                                                   const short int& _dt,
                                                   const unsigned short int& _xa, const unsigned short int& _xb,
                                                   const unsigned short int& _za, const unsigned short int& _zb,
                                                   const unsigned short int& _ea, const unsigned short int& _eb)
{
 // Create a new line and replace the old one
}

Succeeded
InputStreamWithRecordsFromUPENNtxt::
set_up()
{

/*    if ( !this->filename.empty() )
    {
        if ( !inputListFile.open( filename.c_str(), std::ios_base::in | std::ios_base::binary ) )
        {
            std::cerr << "error: cannot open file " << filename.c_str() << '\n';
            std::exit( EXIT_FAILURE );
        }

        inputList = &inputListFile;
    }

    if ( !list::decodeHeader( *inputList, &listHeader )
         || !list::isCoincListHeader( listHeader ) )
    {
        std::cerr << "error: cannot read valid header from input list\n";
        std::exit( EXIT_FAILURE );
    }

    eventFormat = list::eventFormat( listHeader );
    eventCodec = new list::EventCodec( eventFormat );
    eventSize = list::eventSize( eventFormat );
    in = new list::InputBuffer( *inputList, eventSize );
    pos = inputListFile.pubseekoff(0, std::ios_base::cur);

    return Succeeded::yes;*/
}

std::string
InputStreamWithRecordsFromUPENNtxt::
method_info() const
{
    std::ostringstream s;
    s << this->registered_name;
    return s.str();
}

void
InputStreamWithRecordsFromUPENNtxt::set_defaults()
{
    starting_stream_position = 0;
}

void
InputStreamWithRecordsFromUPENNtxt::initialise_keymap()
{

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


//if (is_null_ptr(stream_ptr))
//  return Succeeded::no;

//while(true)
//{
//    std::getline(*stream_ptr, *line);

//    if (stream_ptr->eof())
//        return Succeeded::no;
//    else if (stream_ptr->bad())
//    {
//        warning("InputStreamWithRecordsFromUPENNtxt: Error after reading from list mode stream in get_next_record");
//        return Succeeded::no;
//    }
//    else if (keep_type == 1 && line->at(0)=='p')
//        break;
//    else if (keep_type ==2 && line->at(0)=='d')
//        break;
//    else if (keep_type == 0)
//        break;

//}
////  // rely on file caching by the C++ library or the OS
////  assert(this->size_of_record_signature <= this->max_size_of_record);
////  boost::shared_array<char> data_sptr(new char[this->max_size_of_record]);
////  char * data_ptr = data_sptr.get();
////  stream_ptr->read(data_ptr, this->size_of_record_signature);
////  if (stream_ptr->gcount()<static_cast<std::streamsize>(this->size_of_record_signature))
////    return Succeeded::no;
////  const std::size_t size_of_record = record.size_of_record_at_ptr(data_ptr, this->size_of_record_signature,options);
////  assert(size_of_record <= this->max_size_of_record);
////  if (size_of_record > this->size_of_record_signature)
////    stream_ptr->read(data_ptr + this->size_of_record_signature,
////                     size_of_record - this->size_of_record_signature);

//return
//  record.init_from_data_ptr(line.get());
