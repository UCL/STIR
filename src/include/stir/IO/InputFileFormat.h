//
//
#ifndef __stir_IO_InputFileFormat_h__
#define __stir_IO_InputFileFormat_h__
/*
    Copyright (C) 2006- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2021, University College London
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0

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


  //! read data from a stream
  /*! This function should throw an exception if the read fails.

      \warning This member throws for most implementations. Use a filename instead.
  */
  virtual unique_ptr<DataT>
    read_from_file(std::istream& input) const = 0;
  
  //! read data from a filename
  /*! This function should throw an exception if the read fails.

    Default implementation used \c open_read_binary and \c read_from_file(std::istream&).
  */
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
