//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Inline implementations for class RegisteredObject

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/utilities.h"
#include <iostream>
#ifndef STIR_NO_NAMESPACE
using std::cout;
#endif

START_NAMESPACE_STIR 

template <class Root>
RegisteredObject<Root>::RegisteredObject()
{}


#ifndef __STIR_REGISTRY_NOT_INLINE
template <class Root>
typename RegisteredObject<Root>::RegistryType& 
RegisteredObject<Root>::registry ()
{
  static RegistryType the_registry("None", 0);
  return the_registry;
}
#endif

template <class Root>
Root*
RegisteredObject<Root>::read_registered_object(istream* in, const string& registered_name)
{
  RootFactory factory = registry().find_factory(registered_name);
  return factory==0 ? 0 : (*factory)(in);
}

template <class Root>
Root*
RegisteredObject<Root>::ask_type_and_parameters()
{
  cout << "Which type do you want? Possible values are:\n";
  list_registered_names(cout);
  const string registered_name = ask_string("Enter type", "None");
  return read_registered_object(0, registered_name);
}
 
template <class Root>
void 
RegisteredObject<Root>::
list_registered_names(ostream& stream)
{
  registry().list_keys(stream);
}


END_NAMESPACE_STIR
