//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class RegisteredObject

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/
#ifndef __Tomo_RegisteredObject_H__
#define __Tomo_RegisteredObject_H__

#include "tomo/Object.h"
#include "tomo/FactoryRegistry.h"
#include "tomo/interfile_keyword_functions.h"
#include <iostream>
#include <string>

#ifndef TOMO_NO_NAMESPACES
using std::string;
using std::istream;
#endif

START_NAMESPACE_TOMO



/*!
  \brief Helper class to provide registry mechanisms to a Base class

  Hierarchy looks normally as follows:
  \code
  RegisteredObject<Base>
  Base
  ...
  Parent   <- ParsingObject
  RegisteredParsingObject<Derived,Base,Parent>
  Derived
  \endcode


  \warning Visual C++ cannot inline the registry() function. As a result,
  all possible instantiations of the RegisteredObject template have to be
  defined in RegisteredObject.cxx file(s). 
*/
//TODOdoc more
template <typename Root>
class RegisteredObject : public Object
{
public:
  inline RegisteredObject();

  /*!
    \brief Construct a new object (of Root, but actual type determined by the  registered_name parameter) by parsing the istream
  
    This works by finding a 'root factory' object in a registry, and calling the factory on this istream*.
  */
  inline static Root* read_registered_object(istream* in, const string& registered_name);

  //! \brief ask the user for which type, and then calls read_registered_object(0, type)
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
  //! Static function returning the registry
  /*! \warning This function is non inline when using Visual C++ because of
      a compiler limitation. This means that when using this compiler,
      RegisteredObject will need explicit instantiations for all derived classes.
  */
#ifndef _MSC_VER
  inline 
#endif
    static RegistryType& registry();

};


END_NAMESPACE_TOMO
#include "tomo/RegisteredObject.inl"

#endif

