/*
    Copyright (C) 2006-2010, Hammersmith Imanet Ltd
    Copyright (C) 2013, University College London
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
#ifndef __stir_IO_FileSignature_h__
#define __stir_IO_FileSignature_h__
/*!
  \file
  \ingroup IO
  \brief Declaration of class stir::FileSignature

  \author Kris Thielemans
*/
#include "stir/utilities.h"
#include <fstream>
#include <string>
#include <algorithm>

START_NAMESPACE_STIR

//! A class to read/store the file signature
/*! \ingroup IO
   Most file formats have a 'magic number' at the start, which allows easy
   recognition. This class provides an interface to reading and storing that
   signature.

   The current implementation reads a signature of maximum 1024 bytes. The buffer
   is initialised to zero before reading.
*/
class FileSignature
{
 public:

  //! read signature
  /*! The signature will be read from the current point in the stream. The seekg pointer 
    will be advanced by max 1024.
  */
  explicit FileSignature(std::istream& input)
    {
      this->_read_signature(input);
    }
  //! open file and read signature
  explicit FileSignature(const std::string& filename)
    {
      std::ifstream input;
      open_read_binary(input, filename);
      this->_read_signature(input);
    }

  //! get access to the signature
  /*! Data is only valid up to size() (currently max 1024) */
  const char * get_signature() const
    {
      return this->_signature;
    }

  //! return size of valid signature read from the file
  std::size_t size() const
    {
      return this->_size;
    }
 private:
  static const std::size_t _max_signature_size = 1024U;
  char _signature[_max_signature_size];
  std::size_t _size;

  void _read_signature(std::istream& input)
    {
      // first initialise to zero, just in case the whole buffer isn't set
      std::fill(this->_signature, this->_signature + this->_max_signature_size, '\0');
      input.read(this->_signature, this->_max_signature_size);
      this->_signature[this->_max_signature_size-1]='\0';
      this->_size = static_cast<std::size_t>(input.gcount());
    }
};

END_NAMESPACE_STIR
#endif
