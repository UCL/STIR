///
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of class stir::ListGatingInput, which
  is used for list mode data.

  \author Daniel Deidda
  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2019, National Physical Laboratory
    Copyright (C) 2019, University College of London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_ListGatingInput_H__
#define __stir_listmode_ListGatingInput_H__

#include "stir/Succeeded.h"

START_NAMESPACE_STIR
class Succeeded;

//! A class recording external input to the scanner (normally used for gating)
/*! For some scanners, the state of some external measurements can be recorded in the
   list file, such as ECG triggers etc. We currently assume that these take discrete values.

   If your scanner has more data available, you can provide it in the derived class.
*/
class ListGatingInput
{
public:
  virtual ~ListGatingInput() {}

  //! get gating-related info
  /*! Generally, gates are numbered from 0 to some maximum value.
   */
  virtual unsigned int get_gating() const = 0;

  virtual Succeeded set_gating(unsigned int) = 0;
};

END_NAMESPACE_STIR

#endif
