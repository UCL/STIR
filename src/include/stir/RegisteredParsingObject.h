//
// $Id$
//
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
/*!
  \file
  \ingroup buildblock
  \brief Declaration of class stir::RegisteredParsingObject

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/

#ifndef __stir_RegisteredParsingObject_H__
#define __stir_RegisteredParsingObject_H__


#include "stir/ParsingObject.h"
#include <string>

#ifndef STIR_NO_NAMESPACE
using std::string;
#endif


START_NAMESPACE_STIR


//! Auxiliary class for RegisteredParsingObject
/*!
  \ingroup buildblock
   This class simply makes a class derived from Base and ParsingObject. Its use
    should be restricted to the default value of the RegisteredParsingObject
    template.
*/
template <typename Base>
class AddParser : public Base, public ParsingObject
{};



/*!
  \brief Parent class for all leaves in a RegisteredObject hierarchy that
  do parsing of parameter files.
  \ingroup buildblock
  \see RegisteredObject for an explanation why you would use this class.
  
  RegisteredParsingObject::read_from_stream is implemented in terms of
  ParsingObject::parse.

  Requirements on the class \a Base:
  - It needs to be derived from RegisteredObject<Base>

  Requirements on the class \a Derived:
  - It needs to have a static member static const char * const \a registered_name
  - It needs to have a default constructor
  - It needs to be derived from \a RegisteredParsingObject<Derived,Base,Parent>

  Requirements on the class \a Parent:
  - It needs to be derived from ParsingObject
  - It needs to be derived from \a Base

  Use the 2 parameter form if there is no ParsingObject anywhere in the
  hierarchy yet. However, we recommend to immediately derive \a Base from
  ParsingObject. 

  \par How to add a leaf to the registry at run-time.
  Constructing the hierarchy as above makes sure that everything is
  ready. However, the registry needs to be filled at run-time.
  This could be done with user selection of the desired leaves
  (based on their \a registered_name), or just by entering all leaves
  in the registry.

  A leaf will be entered in the hierarchy by declaring a variable as
  follows:
  \code
   Derived::RegisterIt dummy;
  \endcode
  As soon as the variable is destructed, the leaf will be taken out of
  the registry (but see todo). If you want to add it as long as the program runs, use 
  a static variable.

  Currently, STIR has static variables in files for each module
  (for instance, buildblock_registries.cxx). Note that these files
  have to be linked explicitly into your program, as opposed to
  sticking it in a library. This is because the linker will think that
  the variables in that file are never referenced, so would not include
  it in the final executable (to try to remove redundant object files).
*/
template <typename Derived, typename Base, typename Parent = AddParser<Base> >
class RegisteredParsingObject : public Parent
{
public:
  //! Construct a new object (of type Derived) by parsing the istream
  /*! When the istream * is 0, questions are asked interactively. 
  
      \todo Currently, the return value is a \a Base*. Preferably, it should be a 
      \a Derived*, but it seems the registration machinery would need extra 
      (unsafe) reinterpret_casts to get that to work.
      (TODO find a remedy).
  */
  inline static Base* read_from_stream(istream*); 

  //! Returns  Derived::registered_name
  inline string get_registered_name() const;
  //! Returns a string with all parameters and their values, in a form suitable for parsing again
  inline string parameter_info();

public:

  //! A helper class to allow automatic registration.
  struct RegisterIt
  {
    //! Default constructor adds the type to the registry.
    RegisterIt()
    {
      //std::cerr << "Adding " << Derived::registered_name <<" to registry"<<std::endl;
      // note: VC 7.0 needs a '&' in front of read_from_stream for some reason
      Parent::registry().add_to_registry(Derived::registered_name, &read_from_stream);  
    }
   
    /*! \brief Destructor should remove it from the registry.
      \todo At present, the object remain in the registry, as there is 
      a potential conflict in the order of destruction of the registry and
      the RegisterIt objects. This can be solved with shared_ptr s.
    */
    ~RegisterIt()
    {
#if 0
      // does not work yet, as registry might be destructed before this
      // RegisterIt object. A solution to this problem is coming up.
      cerr << "In RegisterIt destructor for " << Derived::registered_name<<endl;
      cerr <<"Current keys: ";
      Parent::registry().list_keys(cerr);
      Parent::registry().remove_from_registry(Derived::registered_name);
#endif
    }
  };
  // RegisterIt needs to be a friend to have access to registry()
  friend struct RegisterIt;
  
};


END_NAMESPACE_STIR

#include "stir/RegisteredParsingObject.inl"

#endif

