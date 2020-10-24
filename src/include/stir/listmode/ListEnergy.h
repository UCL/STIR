//
/*!
  \file
  \ingroup listmode
  \brief Declarations of class stir::ListEnergy  which
  is used for list mode data.

  \author Ludovica Brusaferri

*/
/*
    Copyright (C) 2019, University College of London
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

#ifndef __stir_listmode_ListEnergy_H__
#define __stir_listmode_ListEnergy_H__

#include "stir/Succeeded.h"
#include "stir/round.h"

START_NAMESPACE_STIR
//class Succeeded;

//! A class for storing and using an energy record from a listmode file
/*! \ingroup listmode
    ListEnergy is used to provide an interface to the 'energy' events
    in the list mode stream.

    \todo this is still under development.
*/
class ListEnergy
{
public:
  virtual ~ListEnergy() {}

    virtual double get_energyA_in_keV() const = 0;
    virtual double get_energyB_in_keV() const = 0;

};

END_NAMESPACE_STIR

#endif
