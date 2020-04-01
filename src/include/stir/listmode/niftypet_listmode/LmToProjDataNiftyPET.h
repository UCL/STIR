#pragma once
//
//
/*!
  \file 
  \ingroup listmode

  \brief Wrapper to NiftyPET's listmode to projection data converter
 
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

#include "stir/listmode/LmToProjDataAbstract.h"

START_NAMESPACE_STIR

/*!
  \ingroup listmode

  \brief This class is used to bin listmode data to projection data,
  i.e. (3d) sinograms using NiftyPET functionality.
*/

class LmToProjDataNiftyPET : public LmToProjDataAbstract
{
public:

    //! This function does the actual work
    virtual void process_data();

};

END_NAMESPACE_STIR