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

  inline static void list_registered_names(ostream& stream);

protected:
  //! The type of a root factory is a function, taking an istream* as argument, and returning a Root*
  typedef Root * (*RootFactory)(istream*);
  //! The type of the registry
  typedef FactoryRegistry<string, RootFactory> RegistryType;
  //! Static function returning the registry
  /*! \warning This function is non inline when using Visual C++. */
  //TODOdoc more
#ifndef _MSC_VER
  inline 
#endif
    static RegistryType& registry();

};


END_NAMESPACE_TOMO
#include "RegisteredObject.inl"

#endif

