//
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class stir::Verbosity

  \author Matthias Ehrhardt

*/
/*
    Copyright (C) 2014, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/Verbosity.h"

START_NAMESPACE_STIR

// Global static pointer used to ensure a single instance of the class.
Verbosity* Verbosity::_instance = NULL; 

Verbosity::Verbosity(){
  _verbosity_level = 2;
};

int Verbosity::get() 
{
  if (!_instance)   // Only allow one instance of class to be generated.
    _instance = new Verbosity;

  return _instance->_verbosity_level;
}

void Verbosity::set(int level) 
{
  if (!_instance)   // Only allow one instance of class to be generated.
    _instance = new Verbosity;

  _instance->_verbosity_level = level;
}

END_NAMESPACE_STIR
