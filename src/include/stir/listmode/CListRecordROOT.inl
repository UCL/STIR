/*
 *  Copyright (C) 2015, 2016 University of Leeds
    Copyright (C) 2016, UCL
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup IO
  \brief Implementation of class stir::InputStreamFromROOTFile

  \author Nikos Efthimiou
*/

#include "stir/listmode/CListRecord.h"

START_NAMESPACE_STIR

bool CListRecordROOT::is_time() const
{ return true; }

bool CListRecordROOT::is_event() const
{ return true; }

bool CListRecordROOT::is_full_event() const
{ return true; }

END_NAMESPACE_STIR
