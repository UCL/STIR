//
// $Id$
//
/*
    Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
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
  \ingroup IO
  \brief Implementation of class stir::InputFileFormatRegistry

  This file provides the implementations of the template class
  stir::InputFileFormatRegistry. It has to be included by a .cxx
  file that needs to instantiate this class.

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/IO/InputFileFormatRegistry.h"
#include "stir/IO/FileSignature.h"
#include "stir/utilities.h" // for open_read_binary
#include <utility> // for make_pair
#include <typeinfo>

START_NAMESPACE_STIR

template <class DataT>
shared_ptr<InputFileFormatRegistry<DataT > >&
InputFileFormatRegistry<DataT>::default_sptr()
{
  /* Implementation note:
     It is safer to have a static member inside this function than
     to have a static member of the class. The reason for this is
     that there is no guarantee on the order in which static
     class members are initialised. So, if other static initialisers 
     would call default_sptr(), we wouldn't be sure if the current
     _default_sptr is already initialised.
     With the current implementation, there is no such problem
     (except potentially in threaded cases. You have to make sure
     that this function is called once before starting any threads).

     Note that this function cannot be inline because the static member
     should be initialised in one translation unit only.
  */
  static shared_ptr<InputFileFormatRegistry<DataT> > 
    _default_sptr = new InputFileFormatRegistry<DataT>;

  // std::cerr<< "\ndefault_sptr value " << _default_sptr.get() << std::endl;
  return _default_sptr;
}

template <class DataT>
void
InputFileFormatRegistry<DataT>::
add_to_registry(FactorySPtr const & factory, const unsigned ranking)
{
  this->_registry.insert(std::make_pair(ranking, factory));
}

template <class DataT>
void
InputFileFormatRegistry<DataT>::
remove_from_registry(const Factory& factory)
{
  iterator iter = this->_registry.begin();
  iterator const end = this->_registry.end();
  while (iter != end)
    {
      if (typeid(*iter->second) == typeid(factory))
	{
	  this->_registry.erase(iter);
	  return;
	}
      ++iter;
    }
}

template <class DataT>
typename InputFileFormatRegistry<DataT>::Factory const & 
InputFileFormatRegistry<DataT>::
find_factory(const FileSignature& signature,
	     std::istream& input) const
{
  const_iterator iter= this->_actual_find_factory(signature, input);
  if (this->_valid(iter))
    return *(iter->second);
  else
    {
      std::cerr << "Available input file formats:\n";
      this->list_registered_names(std::cerr);
      error("no file format found that can read this data");
    }
  // we never get here, but most compilers will complain here
  // so we 'return' a bogus factory
  return (*iter->second);
}

template <class DataT>
typename InputFileFormatRegistry<DataT>::Factory const & 
InputFileFormatRegistry<DataT>::
find_factory(const FileSignature& signature,
	     const std::string& filename) const
{
  const_iterator iter= this->_actual_find_factory(signature, filename);
  if (this->_valid(iter))
    return *(iter->second);
  else
    {
      std::cerr << "Available input file formats:\n";
      this->list_registered_names(std::cerr);
      error("no file format found that can read file '%s'", filename.c_str());
    }
  // we never get here, but most compilers will complain here
  // so we 'return' a bogus factory
  return (*iter->second);
}

template <class DataT>
typename InputFileFormatRegistry<DataT>::Factory const & 
InputFileFormatRegistry<DataT>::
find_factory(const std::string& filename) const
{
  return this->find_factory(FileSignature(filename), filename);
}

template <class DataT>
typename InputFileFormatRegistry<DataT>::Factory const & 
InputFileFormatRegistry<DataT>::
find_factory(std::istream& input) const
{
  return this->find_factory(FileSignature(input), input);
}

template <class DataT>
void
InputFileFormatRegistry<DataT>::
list_registered_names(std::ostream& stream) const
{
  const_iterator iter = this->_registry.begin();
  const_iterator const end = this->_registry.end();
  while (iter != end)
    {
      stream << iter->second->get_name() << '\n';
      ++iter;
    }
}

END_NAMESPACE_STIR

