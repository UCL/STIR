//
//
#ifndef __stir_IO_write_to_file_H__
#define __stir_IO_write_to_file_H__
/*
    Copyright (C) 2015, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup IO
  \brief Declaration of stir::write_to_file function (providing
  easy access to the default stir::OutputFileFormat)

  \author Kris Thielemans

*/
#include "stir/IO/OutputFileFormat.h"
#include "stir/Succeeded.h"
#include "stir/error.h"
#include "stir/format.h"

START_NAMESPACE_STIR

//! Function that writes data to file using the default OutputFileFormat
/*! \ingroup IO
    This is a convenience function that uses OutputFileFormat::default_sptr().
    It calls error() when the writing failed.

    \return The actual filename being used (which might be different if no extension was specified).

    \par warning

    The output file format class used is not for \c DataT but actually for
    \c DataT::hierarchy_base_type. This is necessary such that this function can
    work for \c data being from a derived class (e.g. VoxelsOnCartesianGrid)
    while the output file format is defined for the base (i.e. DiscretisedDensity).

    Sadly, this requires that the DataT::hierarchy_base_type typedef exists.
 */
template <class DataT>
inline std::string
write_to_file(const std::string& filename, const DataT& data)
{
  std::string filename_used(filename);

  if (OutputFileFormat<typename DataT::hierarchy_base_type>::default_sptr()->write_to_file(filename_used, data) != Succeeded::yes)
    {
      error(format("Error writing data to file '{}'", filename));
    }
  return filename_used;
}

END_NAMESPACE_STIR

#endif
