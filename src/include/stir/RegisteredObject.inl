//
//
/*!

  \file
  \ingroup buildblock
  \brief Inline implementations for class stir::RegisteredObject

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
/*
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
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

#include "stir/utilities.h"
#include <iostream>

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
RegisteredObject<Root>::read_registered_object(std::istream* in, const std::string& registered_name)
{
  RootFactory factory = registry().find_factory(registered_name);
  return factory==0 ? 0 : (*factory)(in);
}

template <class Root>
Root*
RegisteredObject<Root>::ask_type_and_parameters()
{
  std::cout << "Which type do you want? Possible values are:\n";
  list_registered_names(std::cout);
  const std::string registered_name = ask_string("Enter type", "None");
  return read_registered_object(0, registered_name);
}
 
template <class Root>
void 
RegisteredObject<Root>::
list_registered_names(std::ostream& stream)
{
  registry().list_keys(stream);
}


END_NAMESPACE_STIR
