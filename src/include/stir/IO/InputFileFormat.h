//
//
#ifndef __stir_IO_InputFileFormat_h__
#define __stir_IO_InputFileFormat_h__
/*
    Copyright (C) 2006- 2011, Hammersmith Imanet Ltd
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
  \ingroup IO
  \brief Declaration of class stir::InputFileFormat

  \author Kris Thielemans

*/
#include "stir/IO/FileSignature.h"
#include "stir/utilities.h"
#include "stir/unique_ptr.h"
#include <fstream>
#include <string>


START_NAMESPACE_STIR

//! Base-class for file-formats for reading
/*! \ingroup IO
    \preliminary

    \todo should be able to open for input-output maybe
    \todo there is overlap between function having filenames or istreams.
       This is a bit of a mess. Also, some file formats we might only have an API for
       using C-style FILE.
*/
template <class DataT>
class InputFileFormat
{
 public:
  typedef DataT data_type;
  virtual ~InputFileFormat() {}

  virtual bool
    can_read(const FileSignature& signature,
	     std::istream& input) const
  {
    return this->actual_can_read(signature, input);
  }
  virtual bool 
    can_read(const FileSignature& signature,
	     const std::string& filename) const
  {
    std::ifstream input;
    open_read_binary(input, filename);
    return this->actual_can_read(signature, input);
  }
  
  virtual unique_ptr<DataT>
    read_from_file(std::istream& input) const = 0;
  
  // note: need to have 2 for header-based file-formats which might need directory info
  virtual 
    unique_ptr<DataT>
    read_from_file(const std::string& filename) const
  {
    std::ifstream input;
    open_read_binary(input, filename);
    return read_from_file(input);
  }

  virtual const std::string
    get_name() const = 0;

 protected:
  virtual 
    bool 
    actual_can_read(const FileSignature& signature,
		    std::istream& input) const = 0;
};

END_NAMESPACE_STIR

#endif
