//
//
/*!

  \file
  \ingroup symmetries

  \brief Implementations of inline functions of class stir::RelatedDensels

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
#include "stir/Densel.h"
#include "stir/recon_buildblock/DataSymmetriesForDensels.h"

START_NAMESPACE_STIR

RelatedDensels::RelatedDensels()
    : related_densels(),
      symmetries()
{}

RelatedDensels::RelatedDensels(const std::vector<Densel>& related_densels_v,
                               const shared_ptr<DataSymmetriesForDensels>& symmetries_used)
    : related_densels(related_densels_v),
      symmetries(symmetries_used)
{}

int
RelatedDensels::get_num_related_densels() const
{
  return related_densels.size();
}

Densel
RelatedDensels::get_basic_densel() const
{
  assert(related_densels.size() != 0);
  return related_densels[0];
}

#if 0
const ProjDataInfo *
RelatedDensels:: get_proj_data_info_sptr() const
{
 
  return related_densels[0].get_proj_data_info_sptr();
}
#endif

const DataSymmetriesForDensels*
RelatedDensels::get_symmetries_ptr() const
{
  return symmetries.get();
}

RelatedDensels::iterator
RelatedDensels::begin()
{
  return related_densels.begin();
}

RelatedDensels::iterator
RelatedDensels::end()
{
  return related_densels.end();
}

RelatedDensels::const_iterator
RelatedDensels::begin() const
{
  return related_densels.begin();
}

RelatedDensels::const_iterator
RelatedDensels::end() const
{
  return related_densels.end();
}

END_NAMESPACE_STIR
