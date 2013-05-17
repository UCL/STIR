//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class stir::FactoryRegistry

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- 2009, Hammersmith Imanet Ltd
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

#ifndef __stir_FactoryRegistry_H__
#define __stir_FactoryRegistry_H__


#include "stir/common.h"
#include <iostream>
#include <map>
#include <functional>

START_NAMESPACE_STIR

/*!
  \ingroup buildblock
  \brief This class can be used to store 'factories' and their corresponding
   keys. It is essentially a map, but with some extra embelishments.

  A factory is supposed to be an object which can create another object, 
  although this is not really enfored by the implementation of 
  FactoryRegistry.

  \par Type requirements

  Key,Factory,Compare must be suitable as defined for std::map. In addition,
  FactoryRegistry::list_keys() requires that operator<<(ostream&, const Key&)
  is defined.

  \todo Probably it would be better to store pointers to factories. However,
  in that case, the destructor of FactoryRegistry would have to deallocate
  these factory objects. This would mean that factories have to be allocated
  with new, and hence would prevent using simple function pointers.
*/
template <typename Key, typename Factory, 
	  typename Compare = std::less<Key> > 
class FactoryRegistry
{

public:
  // sadly, all of these have to be inline to prevent problems
  // with instantiation


  //! Default constructor without defaults (see find_factory())
  inline FactoryRegistry(); 
  /*! \brief
    constructor with default values which will be returned when no 
    match is found (see find_factory())
  */
  inline FactoryRegistry(const Key& default_key,
			 const Factory& default_factory);

  inline ~FactoryRegistry();
			 

  /*! \brief 
    Add a pair to the registry

    Adding the same key twice will overwrite the first value.
  */
  inline void add_to_registry(const Key& key,Factory const & factory);
  
  //! Remove a pair from the registry
  inline void remove_from_registry(const Key& key);

  //! List all keys to an ostream, separated by newlines.
  inline void list_keys(std::ostream& s) const;
  
  //! Find a factory corresponding to a key
  /*! If the key is not found, the behaviour depends on which constructor
      was used. If the (default) no-argument constructor is used, an 
      error message is printed, and the program aborts. 
      If the 2nd constructor with default values, the default_factory is 
      returned.
  */
  inline Factory const & find_factory(const Key& key) const;

private:
  typedef std::map<Key, Factory, Compare > FactoryMap;
  FactoryMap m;	  
  const bool has_defaults;
  const Key default_key;
  const Factory default_factory;

};


END_NAMESPACE_STIR

#include "stir/FactoryRegistry.inl"


#endif
