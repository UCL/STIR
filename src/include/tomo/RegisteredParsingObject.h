//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class 

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/

#ifndef __Tomo_RegisteredParsingObject_H__
#define __Tomo_RegisteredParsingObject_H__


#include "tomo/ParsingObject.h"
#include <string>

#ifndef TOMO_NO_NAMESPACE
using std::string;
#endif


START_NAMESPACE_TOMO


template <typename Base>
class AddParser : public Base, public ParsingObject
{};



/*!
  \brief Parent class for all leaves in a RegisteredObject hierarchy that
  do parsing of parameter files.

  \see RegisteredObject
  
  RegisteredParsingObject::read_from_stream is implemented in terms of
  ParsingObject::parse.

  Requirements on the class Base:
  - It needs to be derived from RegisteredObject<Base>

  Requirements on the class Derived:
  - It needs to have a static member static const char * const registered_name
  - It needs to have a default constructor
  - It needs to be derived from RegisteredParsingObject<Derived,Base,Parent>

  Requirements on the class Parent:
  - It needs to be derived from ParsingObject
  - It needs to be derived from Base
*/
template <typename Derived, typename Base, typename Parent = AddParser<Base> >
class RegisteredParsingObject : public Parent
{
public:
  //! Construct a new object (of type Derived) by parsing the istream
  /*! When the istream * is 0, questions are asked interactively. 
  
      Currently, the return value is a Base*. Preferably, it should be a 
      Derived*, but it seems the registration machinery would need extra 
      (unsafe) reinterpret_casts to get that to work.
      (TODO find a remedy).
  */
  inline static Base* read_from_stream(istream*); 

  //! Returns  Derived::registered_name
  inline string get_registered_name() const;
  //! Returns a string with all parameters and their values, in a form suitable for parsing again
  inline string parameter_info();

protected:
#ifdef _MSC_VER
public:
#endif
  //! A helper class to allow automatic registration.
  struct RegisterIt
  {
    RegisterIt()
    {
      std::cerr << "Adding " << Derived::registered_name <<" to registry"<<std::endl;
      registry().add_to_registry(Derived::registered_name, read_from_stream);  
    }
    ~RegisterIt()
    {
      registry().remove_from_registry(Derived::registered_name);
    }
  };
  // RegisterIt needs to be a friend to have access to registry()
  friend RegisterIt;
  
};


END_NAMESPACE_TOMO

#include "tomo/RegisteredParsingObject.inl"

#endif

