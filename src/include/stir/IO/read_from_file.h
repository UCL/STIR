//
//
#ifndef __stir_IO_read_from_file_H__
#define __stir_IO_read_from_file_H__
/*
    Copyright (C) 2006 - 2008-10-01, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2013, Kris Thielemans
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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

    The input file format class used is not for \c DataT but actually for
    \c DataT::hierarchy_base_type. This is necessary such that this function can
    work for \c data being from a derived class (e.g. VoxelsOnCartesianGrid)
    while the input file format is defined for the base (i.e. DiscretisedDensity).

    Sadly, this requires that the DataT::hierarchy_base_type typedef exists.
 */
template <class DataT, class FileT>
inline 
unique_ptr<DataT>
read_from_file(const FileSignature& signature, FileT file)
{
  using hierarchy_base_type = typename DataT::hierarchy_base_type;
  const InputFileFormat<hierarchy_base_type>& factory =
    InputFileFormatRegistry<hierarchy_base_type>::default_sptr()->
    find_factory(signature, file);
  auto uptr(factory.read_from_file(file));
  // There is no dynamic_pointer_cast for unique_ptr
  // See https://stackoverflow.com/questions/11002641/dynamic-casting-for-unique-ptr why
  // We use a trick mentioned in that link
  auto data_ptr = dynamic_cast<DataT*>(uptr.get());
  if (!data_ptr)
    {
      // TODO improve on the following cryptic error message
      error("data read from file is an incorrect type");
    }
  // get rid of the original uptr (but not the object pointed to) and create a new one
  uptr.release();
  return unique_ptr<DataT>(data_ptr);
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
