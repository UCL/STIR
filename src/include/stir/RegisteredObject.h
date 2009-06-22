//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief Declaration of class stir::RegisteredObject

  \author Kris Thielemans
  \author Sanida Mustafovic

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
#ifndef __stir_RegisteredObject_H__
#define __stir_RegisteredObject_H__

#include "stir/Object.h"
#include "stir/FactoryRegistry.h"
#include "stir/interfile_keyword_functions.h"
#include <iostream>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::istream;
#endif

START_NAMESPACE_STIR



/*!
  \brief Helper class to provide registry mechanisms to a Base class
  \ingroup buildblock

  Suppose you have a hierarchy of classes with (nearly) all public
  functionality provided by virtual functions of the Base class.
  The aim is then to be able to select <i>at run-time</i> which 
  of the nodes will be used. 

  To do this, one needs to enter all
  node classes in a registry. This registry contains a key and a
  &quot;Base factory&quot; for every node-class. The factory for
  the node-class returns (a pointer to) a new node-class object, 
  which of course is also a Base object.

  In STIR, FactoryRegistry provides the type for the registry.

  In many cases, the factory constructs the new object from a stream. 
  The current class provides the basic mechanisms for this, i.e.
  a registry, and a function that looks up the relevant factory
  in the registry and uses it to construct the object from a stream.
  In addition, there is an interactive function for asking the type
  and its parameters. This makes only sense if the object construction
  can be interactive as well (see ask_type_and_parameters()).

  In case the construction of the object is done by using ParsingObject,
  nearly all of the necessary functionality can be provided to the
  hierarchy by using RegisteredParsingObject in the hierarchy. In such 
  a case, the hierarchy looks normally as follows:
  \code
  RegisteredObject<Base>
  Base
  ...
  Parent   <- ParsingObject
  RegisteredParsingObject<Derived,Base,Parent>
  Derived
  \endcode

  A <strong>recommended</strong> variation on this is the following:
  \code
  RegisteredObject<Base>
  Base <- ParsingObject
  ...
  RegisteredParsingObject<Derived,Base,Base>
  Derived
  \endcode
  Aside from the fact that this is simpler, it also is more future proof.
  Suppose that at some point you want to add parsing keys that are common to 
  the whole hierarchy. The best/only place to do this is to add them to 
  the \a Base class. But this can only be done if \a Base is derived from 
  ParsingObject, and hence there is can be no other ParsingObject in the 
  hierarchy. So, the recommended variation will need no change at all, while
  the general version would need change to all nodes.

  As a final note, RegisteredObject could also be used for hierarchies that
  do not use ParsingObject.
  
  \see RegisteredParsingObject

  \todo Currently there is a hard-wired value of &quot;None&quot;
  for the default key (with a 0 factory). This is inappropriate 
  in some cases.

  \warning Visual C++ cannot inline the registry() function. As a result,
  all possible instantiations of the RegisteredObject template have to be
  defined in RegisteredObject.cxx file(s). You will have link errors if
  you forgot to do this.
*/
template <typename Root>
class RegisteredObject : public Object
{
public:
  inline RegisteredObject();

  /*!
    \brief Construct a new object (of a type derived from Root, its actual type determined by the  registered_name parameter) by parsing the istream
  
    This works by finding the 'root factory' object in a registry that corresponds to
    \a registered_name, and calling the factory on this istream*.
  */
  inline static Root* read_registered_object(istream* in, const string& registered_name);

  //! \brief ask the user for the type, and then calls read_registered_object(0, type)
  /*! 
    \warning Relies on read_registered_object to be interactive when its first argument is 0.

    Sadly, this function cannot be called ask_parameters() because of conflicts with
    ParsingObject::ask_parameters() in the derived classes.
  */
  inline static Root* ask_type_and_parameters();

  //! List all possible registered names to the stream
  /*! Names are separated with newlines. */
  inline static void list_registered_names(ostream& stream);

  
protected:
  //! The type of a root factory is a function, taking an istream* as argument, and returning a Root*
  typedef Root * (*RootFactory)(istream*);
  //! The type of the registry
  typedef FactoryRegistry<string, RootFactory, interfile_less> RegistryType;

#if defined(_MSC_VER) && _MSC_VER<=1300
#  define __STIR_REGISTRY_NOT_INLINE
#endif

  //! Static function returning the registry
  /*! \warning This function is non inline when using Visual C++ 6.0 because of
      a compiler limitation. This means that when using this compiler,
      RegisteredObject will need explicit instantiations for all derived classes.
  */
#ifndef __STIR_REGISTRY_NOT_INLINE
  inline
#endif
    static RegistryType& registry();

};


END_NAMESPACE_STIR
#include "stir/RegisteredObject.inl"

#endif

