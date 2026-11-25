/*
    Copyright (C) 2025, London South Bank University
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Declaration of class stir::ListAccumulatedSingles
  \author Alan Osys
*/

#ifndef __stir_listmode_ListAccumulatedSingles_H__
#define __stir_listmode_ListAccumulatedSingles_H__

#include "stir/Succeeded.h"

START_NAMESPACE_STIR

/*!
  \ingroup listmode
  \brief Class for an accumulated singles record in a listmode file

  Provides an interface to singles bucket data embedded in
  list-mode streams, following the pattern of ListTime and ListEvent.

  Each record holds the number of singles accumulated by one singles unit (bucket)
  over some time interval. This is not a histogram in the sense of having multiple bins,
  and it does not describe individual singles events (which would need a different class).
*/
class ListAccumulatedSingles
{
public:
  virtual ~ListAccumulatedSingles() {}

  //! Returns the singles bucket index
  virtual unsigned int get_bucket_index() const = 0;

  //! Returns the accumulated singles count for this bucket
  virtual float get_singles_count() const = 0;
};

END_NAMESPACE_STIR

#endif