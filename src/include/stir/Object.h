//
// $Id$
//
/*!

  \file

  \brief Declaration of class stir::Object

  \author Kris Thielemans
  
  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

#include "stir/common.h"

#include <string>
#ifndef STIR_NO_NAMESPACE
using std::string;
#endif

#ifndef __stir_Object_H__
#define __stir_Object_H__

START_NAMESPACE_STIR

/*! \brief Base class for all classes that can parse .par files (and more?) 
  \ingroup buildblock
  The main reason that this class exists is such that KeyParser can store
  different types of objects, and get some basic info from it.

  \see RegisteredObject for more info on the registries etc that are used
  for parsing..
*/

class Object
{
public:
  virtual ~Object() {}
  /*! \brief return a string describing all parameters of the object

  There is currently no requirement on the format of the returned string.
  Ideally it would be such that the object can be constructed by parsing
  the string again.

  \todo Ideally this function would be a const member, but we cannot
  do this because ParsingObject::parameter_info is not const.
  */
  virtual string parameter_info()  = 0;
  /*! \brief Returns the name of the type of the object.

  Each type that can be parsed has a unique (within its hierarchy) name
  associated with it. This function returns that name.
  KeyParser::parameter_info() needs to know this
  name such that it can fill it in.
  */
  virtual string get_registered_name() const= 0;
};
END_NAMESPACE_STIR
#endif

