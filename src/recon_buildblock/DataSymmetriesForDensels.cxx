//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
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
  \ingroup symmetries

  \brief implementations for class stir::DataSymmetriesForDensels

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/

#include "stir/recon_buildblock/DataSymmetriesForDensels.h"
#include "stir/Densel.h"
#include "stir/recon_buildblock/SymmetryOperation.h"
#include <typeinfo>

START_NAMESPACE_STIR

DataSymmetriesForDensels::
DataSymmetriesForDensels()
{}

/*! Default implementation always returns \c true. Needs to be overloaded.
 */
bool
DataSymmetriesForDensels::
blindly_equals(const root_type * const) const
{ 
  return true;
}

bool
DataSymmetriesForDensels::
operator ==(const root_type& that) const
{ 
  return
    typeid(*this) == typeid(that) &&
    this->blindly_equals(&that);
}

bool
DataSymmetriesForDensels::
operator !=(const root_type& that) const
{ 
  return !((*this) == that);
}

/*! default implementation in terms of get_related_densels, will be slow of course */
int
DataSymmetriesForDensels::num_related_densels(const Densel& b) const
{
  vector<Densel> rel_b;
  get_related_densels(rel_b, b);
  return rel_b.size();
}

/*! default implementation in terms of find_symmetry_operation_from_basic_densel */
bool DataSymmetriesForDensels::find_basic_densel(Densel& b) const
{
  auto_ptr<SymmetryOperation> sym_op =
    find_symmetry_operation_from_basic_densel(b);
  return sym_op->is_trivial();
}


END_NAMESPACE_STIR
