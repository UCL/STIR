//
//
/*
    Copyright (C) 2006 - 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
