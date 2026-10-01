/*
    Copyright (C) 2025, University College London
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Declaration of class stir::ListSingles
  \author Alan Osys
*/

#ifndef __stir_listmode_ListSingles_H__
#define __stir_listmode_ListSingles_H__

#include "stir/Succeeded.h"

START_NAMESPACE_STIR

/*!
  \ingroup listmode
  \brief Class for a singles record in a listmode file

  Provides an interface to singles bucket data embedded in
  list-mode streams, following the pattern of ListTime and ListEvent.
*/
class ListSingles
{
public:
  virtual ~ListSingles() {}

  //! Returns the singles bucket index
  virtual unsigned int get_bucket_index() const = 0;

  //! Returns the singles count for this bucket
  virtual float get_singles_count() const = 0;
};

END_NAMESPACE_STIR

#endif