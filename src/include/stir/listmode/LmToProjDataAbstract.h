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

#include "stir/ParsingObject.h"
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
    virtual void process_data() = 0;
};

END_NAMESPACE_STIR