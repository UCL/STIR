/*
    Copyright (C) 2018, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_MultipleDataSetHeader__H__
#define __stir_MultipleDataSetHeader__H__

/*!
  \file
  \ingroup data_buildblock
  \brief Declaration of class stir::MultipleDataSetHeader
  \author Richard Brown
  
  Give a txt file with the names of the individual filenames, e.g.,:

  Multi :=
    total number of data sets := 2
    data set[1] := sinogram_1.hs
    data set[2] := sinogram_2.hs
  end :=
*/

#include "stir/KeyParser.h"

START_NAMESPACE_STIR

class MultipleDataSetHeader : public KeyParser
{
public:

    //! Default constructor
    MultipleDataSetHeader();

    //! Name which will be used when parsing a GeneralisedObjectiveFunction object
    static const char * const registered_name;

    //! Get number of data sets
    std::size_t get_num_data_sets() const { return _num_data_sets; }

    //! Get ith filename
    /*! \warning Indexing here starts from 0*/
    std::string get_filename(std::size_t i) const { return _filenames.at(i); }

    //! Create a Multi header pointing to a set of filenames
    template <class VectorOfStringsT>
      inline static void write_header(const std::string& filename, const VectorOfStringsT& individual_filenames);

protected:

//! Read number of data sets
void read_num_data_sets();

//! Sets defaults before parsing
virtual void set_defaults();
//! Initialise keymap
virtual void initialise_keymap();
//! Post process
virtual bool post_processing();

int _num_data_sets;
std::vector<std::string> _filenames;

};

END_NAMESPACE_STIR

#include "stir/MultipleDataSetHeader.inl"
#endif
