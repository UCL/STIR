//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London
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
/*!
  \file
  \ingroup data_buildblock
  \brief Implementation of class stir::MultipleDataSetHeader
  \author Richard Brown
  
*/

#include "stir/MultipleDataSetHeader.h"
#include <boost/format.hpp>

START_NAMESPACE_STIR

const char * const MultipleDataSetHeader::
registered_name = "MultipleDataSetHeader";

MultipleDataSetHeader::
MultipleDataSetHeader()
{
    this->set_defaults();
    this->initialise_keymap();
}

void MultipleDataSetHeader::
set_defaults()
{
    _num_data_sets = 0;
}

void MultipleDataSetHeader::
initialise_keymap()
{
    this->add_start_key("Multi");
    this->add_stop_key("End");

    this->add_key("total number of data sets",
                  KeyArgument::INT,
                  (KeywordProcessor)&MultipleDataSetHeader::read_num_data_sets,
                  &_num_data_sets);
    this->add_key("data set",
                  KeyArgument::ASCII,
                  &_filenames);
}

bool MultipleDataSetHeader::
post_processing()
{
    bool empty_filenames = false;
    for (int i=0; i<_num_data_sets; ++i) {
        if (_filenames[i] == "") {
          warning(boost::format("MultipleDataSetHeader: Data set[%1%] is empty.") % i);
            empty_filenames = true;
        }
    }

    return empty_filenames;
}

void MultipleDataSetHeader::
read_num_data_sets()
{
    set_variable();
    _filenames.resize(_num_data_sets);
}

END_NAMESPACE_STIR
