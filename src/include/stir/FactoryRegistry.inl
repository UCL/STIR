//
//
/*!

  \file
  \ingroup buildblock
  \brief Inline implementations for stir::FactoryRegistry

  \author Kris Thielemans

*/
/*
    Copyright (C) 2001- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include <utility>
#include <sstream>
#include "stir/warning.h"
#include "stir/error.h"

START_NAMESPACE_STIR

template <class Key, class Factory, class Compare>
FactoryRegistry<Key, Factory, Compare>::FactoryRegistry()
    : has_defaults(false)
{}

template <class Key, class Factory, class Compare>
FactoryRegistry<Key, Factory, Compare>::FactoryRegistry(const Key& default_key, const Factory& default_factory)
    : has_defaults(true),
      default_key(default_key),
      default_factory(default_factory)
{
  add_to_registry(default_key, default_factory);
}

template <class Key, class Factory, class Compare>
FactoryRegistry<Key, Factory, Compare>::~FactoryRegistry()
{
  // TODO not so sure how to get rid of them
  // we can only use delete if the factories are allocated with new
  // and that's currently not the case
  //       for (FactoryMap::iterator i = m.begin(); i != m.end(); ++i)
  //         delete i->second;
}

template <class Key, class Factory, class Compare>
void
FactoryRegistry<Key, Factory, Compare>::add_to_registry(const Key& key, Factory const& factory)
{
  // cerr << "Adding "<< key << "to registry\n";
#ifndef NDEBUG
  typename FactoryMap::iterator iter = m.find(key);
  if (iter != m.end())
    {
      std::ostringstream s;
      s << "FactoryRegistry:: overwriting previous value of key in registry.\n     key: " << key;
      warning(s.str());
    }

#endif
  m.insert(std::pair<Key, Factory>(key, factory));
}

template <class Key, class Factory, class Compare>
void
FactoryRegistry<Key, Factory, Compare>::remove_from_registry(const Key& key)
{
  // cerr << "Removing "<< key << "to registry\n";
  typename FactoryMap::iterator iter = m.find(key);
  if (iter == m.end())
    {
#ifndef _NDEBUG
      // TODO don't output to cerr, but use only warning()
      std::ostringstream s;
      s << "FactoryRegistry:: Attempt to remove key from registry, but it's not in there...\n     key: " << key;
      warning(s.str());
#endif
    }
  else
    m.erase(iter);
}

template <class Key, class Factory, class Compare>
void
FactoryRegistry<Key, Factory, Compare>::list_keys(std::ostream& s) const
{
  for (typename FactoryMap::const_iterator i = m.begin(); i != m.end(); ++i)
    s << i->first << '\n';
}

template <class Key, class Factory, class Compare>
Factory const&
FactoryRegistry<Key, Factory, Compare>::find_factory(const Key& key) const /*throw(unknown_typename)*/
{
  typename FactoryMap::const_iterator i = m.find(key);
  if (i != m.end())
    return i->second;

  // key not found

  // throw(unknown_typename(key));
  // TODO don't output to cerr, but use only error()
  std::cerr << "FactoryRegistry: key " << key << " not found in current registry\n" << m.size() << " possible values are:\n";
  list_keys(std::cerr);
  std::cerr << '\n';
  if (has_defaults)
    {
      std::cerr << "Using value corresponding to key \"" << default_key << "\"" << std::endl;
      return default_factory;
    }
  else
    {
      error("FactoryRegistry: aborting");
      // stupid line to prevent warning messages of good compilers
      return default_factory;
    }
}

END_NAMESPACE_STIR
