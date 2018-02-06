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

#include "stir/Verbosity.h"

START_NAMESPACE_STIR

// Global static pointer used to ensure a single instance of the class.
Verbosity* Verbosity::_instance = NULL; 

Verbosity::Verbosity(){
  _verbosity_level = 1;
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
