///
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes stir::SPECTListRecordWithGatingInput which
  is used for list mode data.

  \author Daniel Deidda
  \author Kris Thielemans

*/
/*
    Copyright (C) 2019, National Physical Laboratory
    Copyright (C) 2019, University College of London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_SPECTListRecordWithGatingInput_H__
#define __stir_listmode_SPECTListRecordWithGatingInput_H__

#include "SPECTListRecord.h"
#include "ListGatingInput.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

class SPECTListRecordWithGatingInput : public virtual SPECTListRecord
{
 public:
  virtual bool is_gating_input() const { return false; }
  virtual ListGatingInput&  gating_input() = 0;
  virtual const ListGatingInput&  gating_input() const = 0;
};

END_NAMESPACE_STIR

#endif
