//
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Implementations of inline functions of class stir::RelatedBins

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
#include "stir/Bin.h"
#include "stir/recon_buildblock/DataSymmetriesForBins.h"

START_NAMESPACE_STIR



RelatedBins::RelatedBins()
:related_bins(),symmetries()
{}

RelatedBins::RelatedBins(const std::vector< Bin>& related_bins_v,
                         const shared_ptr<DataSymmetriesForBins>& symmetries_used)
:related_bins(related_bins_v),symmetries(symmetries_used)
{}

int
RelatedBins::get_num_related_bins() const
{
  return static_cast<int>(related_bins.size());
}

Bin
RelatedBins::get_basic_bin() const
{
  assert(related_bins.size() != 0);
  return related_bins[0];
}


#if 0
const ProjDataInfo *
RelatedBins:: get_proj_data_info_sptr() const
{
 
  return related_bins[0].get_proj_data_info_sptr();
}
#endif


const DataSymmetriesForBins*
RelatedBins::get_symmetries_ptr() const
{
  return symmetries.get();
}


RelatedBins::iterator 
RelatedBins::begin()
{ return related_bins.begin();}

RelatedBins::iterator
RelatedBins::end()
{return related_bins.end();}

RelatedBins::const_iterator 
RelatedBins::begin() const
{return related_bins.begin();}

RelatedBins::const_iterator 
RelatedBins::end() const
{return related_bins.end();}


END_NAMESPACE_STIR

