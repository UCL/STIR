//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup data_buildblock
  \brief Implementation of class stir::MultipleDataSetHeader
  \author Richard Brown

*/

#include "stir/MultipleDataSetHeader.h"
#include "stir/format.h"
#include "stir/warning.h"

START_NAMESPACE_STIR

const char* const MultipleDataSetHeader::registered_name = "MultipleDataSetHeader";

MultipleDataSetHeader::MultipleDataSetHeader()
{
  this->set_defaults();
  this->initialise_keymap();
}

void
MultipleDataSetHeader::set_defaults()
{
  _num_data_sets = 0;
}

void
MultipleDataSetHeader::initialise_keymap()
{
  this->add_start_key("Multi");
  this->add_stop_key("End");

  this->add_key("total number of data sets",
                KeyArgument::INT,
                static_cast<KeywordProcessor>(&MultipleDataSetHeader::read_num_data_sets),
                &_num_data_sets);
  this->add_vectorised_key("data set", &_filenames);
}

bool
MultipleDataSetHeader::post_processing()
{
  bool empty_filenames = false;
  for (int i = 0; i < _num_data_sets; ++i)
    {
      if (_filenames[i].size() == unsigned(0))
        {
          warning(format("MultipleDataSetHeader: Data set[{}] is empty.", i));
          empty_filenames = true;
        }
    }

  return empty_filenames;
}

void
MultipleDataSetHeader::read_num_data_sets()
{
  set_variable();
  _filenames.resize(_num_data_sets);
}

END_NAMESPACE_STIR
