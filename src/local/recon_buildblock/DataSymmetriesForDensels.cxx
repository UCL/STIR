//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief implementations for class DataSymmetriesForDensels

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/recon_buildblock/DataSymmetriesForDensels.h"
#include "local/stir/Densel.h"
#include "stir/recon_buildblock/SymmetryOperation.h"

START_NAMESPACE_STIR

DataSymmetriesForDensels::
DataSymmetriesForDensels()
{}

/*! default implementation in terms of get_related_densels, will be slow of course */
int
DataSymmetriesForDensels::num_related_densels(const Densel& b) const
{
  vector<Densel> rel_b;
  get_related_densels(rel_b, b);
  return rel_b.size();
}

/*! default implementation in terms of find_symmetry_operation_to_basic_densel */
bool DataSymmetriesForDensels::find_basic_densel(Densel& b) const
{
  auto_ptr<SymmetryOperation> sym_op =
    find_symmetry_operation_to_basic_densel(b);
  return sym_op->is_trivial();
}


END_NAMESPACE_STIR
