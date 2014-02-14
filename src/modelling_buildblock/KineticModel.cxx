//
//
/*
    Copyright (C) 2006 - 2009, Hammersmith Imanet Ltd
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

  \file
  \ingroup modelling
  \brief Implementations of inline functions of class stir::KineticModel

  \author Charalampos Tsoumpas

  This is the most basic class for including kinetic models. 

*/


#include "stir/modelling/KineticModel.h"


START_NAMESPACE_STIR

const char * const 
KineticModel::registered_name = "Kinetic Model Type";

KineticModel::KineticModel()    //!< default constructor
{ }
KineticModel::~KineticModel()   //!< default destructor
{ }

END_NAMESPACE_STIR
