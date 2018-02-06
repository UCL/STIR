//
//
#ifndef __stir_IO_read_from_file_H__
#define __stir_IO_read_from_file_H__
/*
    Copyright (C) 2006 - 2008-10-01, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2013, Kris Thielemans
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
  \brief Declaration of stir::read_from_file functions (providing
  easy access to class stir::InputFileFormatRegistry)

  \author Kris Thielemans

*/
#include "stir/IO/InputFileFormatRegistry.h"
#include "stir/unique_ptr.h"

START_NAMESPACE_STIR

//! Function that reads data from file using the default InputFileFormatRegistry, using the provided FileSignature to find the matching file format
/*! \ingroup IO
    This is a convenience function that uses InputFileFormatRegistry::find_factory() to find the
    InputFileFormat factory, and uses it to create the \c DataT object.

    Note that (at the time of writing) InputFileFormatRegistry::find_factory() calls
    error() if no matching file format was found.

    You probably want to use read_from_file(filename);
 */
template <class DataT, class FileT>
inline 
unique_ptr<DataT>
read_from_file(const FileSignature& signature, FileT file)
{
  const InputFileFormat<DataT>& factory = 
    InputFileFormatRegistry<DataT>::default_sptr()->
    find_factory(signature, file);
  return factory.read_from_file(file);
}

//! Function that reads data from file using the default InputFileFormatRegistry
/*! \ingroup IO
    This is a convenience function that first reads the FileSignature, then uses 
    InputFileFormatRegistry::find_factory() to find the factory, which then is used
    to create the object.

    Note that (at the time of writing) InputFileFormatRegistry::find_factory() calls
    error() if no matching file format was found.

    \see read_from_file(const FileSignature&, FileT)
    \par Example

    \code
    typedef DiscretisedDensity<3,float> DataType ;
    unique_ptr<DataType> density_uptr(read_from_file<DataType>("my_file.hv"));
    shared_ptr<DataType> density_sptr(read_from_file<DataType>("another_file.hv"));
    \endcode
*/
template <class DataT, class FileT>
inline
unique_ptr<DataT>
read_from_file(FileT file)
{
  const FileSignature signature(file);
  return read_from_file<DataT>(signature, file);
}

END_NAMESPACE_STIR

#endif
