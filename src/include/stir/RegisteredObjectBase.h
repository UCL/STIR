//
//
/*!

  \file

  \brief Declaration of class stir::RegisteredObjectBase

  \author Kris Thielemans
  

*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/ParsingObject.h"

#include <string>

#ifndef __stir_RegisteredObjectBase_H__
#define __stir_RegisteredObjectBase_H__

START_NAMESPACE_STIR

/*! \brief Base class for all classes that can parse .par files (and more?) 
  \ingroup buildblock
  The only reason that this class exists is such that KeyParser can store
  different types of objects, and get some basic info from it.

  \see RegisteredObject for more info on the registries etc that are used
  for parsing..
*/

class RegisteredObjectBase : public ParsingObject
{
public:
  virtual ~RegisteredObjectBase() {}

  /*! \brief Returns the name of the type of the object.

  Each type that can be parsed has a unique (within its hierarchy) name
  associated with it. This function returns that name.
  KeyParser::parameter_info() needs to know this
  name such that it can fill it in.
  */
  virtual std::string get_registered_name() const= 0;
};
END_NAMESPACE_STIR
#endif

