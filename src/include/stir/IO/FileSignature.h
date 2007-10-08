//
// $Id$
//
/*
    Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
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

  $Date$
  $Revision$
*/
#include "stir/common.h"
#include "stir/utilities.h"
#include <fstream>
#include <string>


START_NAMESPACE_STIR

//! A class to read/store the file signature
/*! \ingroup IO
   Most file formats have a 'magic number' at the start, which allows easy
   recognition. This class provides an interface to reading and storing that
   signature.
*/
class FileSignature
{
 public:

  explicit FileSignature(std::istream& input)
    {
      this->_read_signature(input);
    }
  explicit FileSignature(const std::string& filename)
    {
      std::ifstream input;
      open_read_binary(input, filename);
      this->_read_signature(input);
    }
  
  const char * get_signature() const
    {
      return this->_signature;
    }

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
      input.read(this->_signature, this->_max_signature_size);
      this->_signature[this->_max_signature_size-1]='\0';
      this->_size = input.gcount();
    }
};

END_NAMESPACE_STIR
#endif
