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

#include "stir/IO/InputStreamWithRecordsFromUPENN.h"
#include "stir/IO/FileSignature.h"
#include "stir/error.h"
#include "stir/FilePath.h"
#include "stir/info.h"

START_NAMESPACE_STIR

InputStreamWithRecordsFromUPENN::InputStreamWithRecordsFromUPENN()
{
  set_defaults();
}

unsigned long int
InputStreamWithRecordsFromUPENN::get_total_number_of_events(CListRecordPENN& record)
{

  this->save_get_position();
  this->reset();
  unsigned long int counter = 0;

  while (true)
    {
      if (this->get_next_record(record) == Succeeded::no)
        break;
      if (counter > 1 && counter % 1000000L == 0)
        //            info( format("Counting records: {} ", counter));
        std::cout << "\r" << counter << " events counted " << std::flush;
      counter++;
    }

  this->reset();
  this->set_get_position(get_saved_get_positions().back());
  return counter;
}

Succeeded
InputStreamWithRecordsFromUPENN::set_up()
{
  return Succeeded::yes;
}

void
InputStreamWithRecordsFromUPENN::set_defaults()
{
  starting_stream_position = 0;
  N = 0;
  abrupt_counter = N;
  minE_chan = 0;
  maxE_chan = 1000;
  keep_prompt = true;
  keep_delayed = true;
  abrupt_stop = false;
}

void
InputStreamWithRecordsFromUPENN::initialise_keymap()
{
  this->parser.add_key("name of data file", &this->filename);
  this->parser.add_key("low energy window (chan)", &this->minE_chan);
  this->parser.add_key("upper energy window (chan)", &this->maxE_chan);
  this->parser.add_key("keep prompts", &this->keep_prompt);
  this->parser.add_key("keep delayed", &this->keep_delayed);
}

bool
InputStreamWithRecordsFromUPENN::post_processing()
{
  return false;
}

END_NAMESPACE_STIR
