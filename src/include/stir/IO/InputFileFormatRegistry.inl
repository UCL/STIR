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

  \author Kris Thielemans

  $Date$
  $Revision$
*/

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
     With the current implementation, there is no such problem.
  */
  static shared_ptr<InputFileFormatRegistry<DataT> > 
    _default_sptr(new InputFileFormatRegistry<DataT>);

  //std::cerr<< "\ndefault_sptr value " << _default_sptr.get() << std::endl;
  return _default_sptr;
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
