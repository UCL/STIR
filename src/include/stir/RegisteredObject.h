//
//
/*!
  \file
  \ingroup buildblock
  \brief Declaration of class stiir::RegisteredObject

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
/*
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_RegisteredObject_H__
#define __stir_RegisteredObject_H__

#include "stir/RegisteredObjectBase.h"
#include "stir/FactoryRegistry.h"
#include "stir/interfile_keyword_functions.h"
#include <iostream>
#include <string>

START_NAMESPACE_STIR



/*!
  \brief Helper class to provide registry mechanisms to a \c Base class
  \ingroup buildblock

  Suppose you have a hierarchy of classes with (nearly) all public
  functionality provided by virtual functions of the \c Base (here called
  \c Root) class.
  The aim is then to be able to select <i>at run-time</i> which 
  of the nodes will be used. 

  To do this, one needs to enter all
  node classes in a registry. This registry contains a key and a
  &quot;Root factory&quot; for every node-class. The factory for
  the node-class returns (a pointer to) a new node-class object, 
  which of course is also a Root object.

  In STIR, FactoryRegistry provides the type for the registry.

  In many cases, the factory constructs the new object from a stream. 
  The current class provides the basic mechanisms for this, i.e.
  a registry, and a function that looks up the relevant factory
  in the registry and uses it to construct the object from a stream.
  In addition, there is an interactive function for asking the type
  and its parameters. This makes only sense if the object construction
  can be interactive as well (see ask_type_and_parameters()).

  We currently assume that the construction of the object is done by using ParsingObject.
  Nearly all of the necessary functionality can be provided to the
  hierarchy by using RegisteredParsingObject in the hierarchy.
  The hierarchy looks normally as follows:
  \code
  ParsingObject
  RegisteredObjectBase
  RegisteredObject<Root>
  Root
  ...
  Parent
  RegisteredParsingObject<Derived,Root,Parent>
  Derived
  \endcode

  When there is no intermediate class in hierarchy, this is simplified to:
  \code
  ParsingObject
  RegisteredObjectBase
  RegisteredObject<Root>
  Root
  RegisteredParsingObject<Derived,Root,Root>
  Derived
  \endcode
  
  \see RegisteredParsingObject

  \todo Currently there is a hard-wired value of &quot;None&quot;
  for the default key (with a 0 factory). This is inappropriate 
  in some cases.

  \warning old versions of Visual C++ cannot inline the registry() function. As a result,
  all possible instantiations of the RegisteredObject template have to be
  defined in RegisteredObject.cxx file(s). You will have link errors if
  you forgot to do this.

  \par Limitation: 

  In the previous (including STIR 4.x) version of this hierarchy, ParsingObject wasn't at the
  root of everything. However, the current hierarchy is simpler to use, and you can still
  override relevant members such that ParsingObject is effectively not used.
*/
template <typename Root>
class RegisteredObject : public RegisteredObjectBase
{
public:
  inline RegisteredObject();

  /*!
    \brief Construct a new object (of a type derived from Root, its actual type determined by the  registered_name parameter) by parsing the istream
  
    This works by finding the 'root factory' object in a registry that corresponds to
    \a registered_name, and calling the factory on this istream*.
  */
  inline static Root* read_registered_object(std::istream* in, const std::string& registered_name);

  //! \brief ask the user for the type, and then calls read_registered_object(0, type)
  /*! 
    \warning Relies on read_registered_object to be interactive when its first argument is 0.

    Sadly, this function cannot be called ask_parameters() because of conflicts with
    ParsingObject::ask_parameters() in the derived classes.
  */
  inline static Root* ask_type_and_parameters();

  //! List all possible registered names to the stream
  /*! Names are separated with newlines. */
  inline static void list_registered_names(std::ostream& stream);

  
protected:
  //! The type of a root factory is a function, taking an istream* as argument, and returning a Root*
  typedef Root * (*RootFactory)(std::istream*);
  //! The type of the registry
  typedef FactoryRegistry<std::string, RootFactory, interfile_less> RegistryType;

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

