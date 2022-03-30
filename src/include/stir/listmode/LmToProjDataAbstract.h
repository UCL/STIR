#pragma once
//
//
/*!
  \file 
  \ingroup listmode

  \brief Abstract base class for listmode to projection data conversion.
 
  \author Richard Brown
  
*/
/*
    Copyright (C) 2020, University College of London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/ParsingObject.h"
#include "stir/ProjData.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

/*!
  \ingroup listmode
  \ingroup NiftyPET

  \brief This class is the abstract base class fir binning listmode data to projection data,
  i.e. (3d) sinograms.

  It provides the basic machinery to go through a list mode data file,
  and write projection data for each time frame. 
*/

class LmToProjDataAbstract : public ParsingObject
{
public:

    /// Destructor
    virtual ~LmToProjDataAbstract() {}

    /// Set up
    virtual Succeeded set_up() { return Succeeded::yes; }

    //! This function does the actual work
    virtual void process_data(shared_ptr<ProjData> proj_data_sptr = nullptr) = 0;
};

END_NAMESPACE_STIR