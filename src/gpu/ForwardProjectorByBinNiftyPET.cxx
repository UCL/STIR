//
//
/*!

  \file
  \ingroup projection

  \brief non-inline implementations for stir::ForwardProjectorByBinNiftyPET

  \author Richard Brown


*/
/*
    Copyright (C) 2019, University College London
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

#include "stir/gpu/ForwardProjectorByBinNiftyPET.h"
#include "stir/recon_buildblock/find_basic_vs_nums_in_subsets.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjData.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/error.h"
#include <boost/format.hpp>
#include <iostream>

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const 
ForwardProjectorByBinNiftyPET::registered_name =
  "NiftyPET";

ForwardProjectorByBinNiftyPET::ForwardProjectorByBinNiftyPET()
{
    this->_already_set_up = false;
}

ForwardProjectorByBinNiftyPET::~ForwardProjectorByBinNiftyPET()
{
}

void
ForwardProjectorByBinNiftyPET::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr, 
       const shared_ptr<DiscretisedDensity<3,float> >& density_info_sptr)
{
    check(*this->_proj_data_info_sptr, *_density_sptr);
    ForwardProjectorByBin::set_up(proj_data_info_sptr, density_info_sptr);
    _symmetries_sptr.reset(new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_sptr, density_info_sptr));

    // Initialise projected_data_sptr from this->_proj_data_info_sptr
    _projected_data_sptr.reset(
                new ProjDataInMemory(this->_density_sptr->get_exam_info_sptr(), this->_proj_data_info_sptr));
}

const DataSymmetriesForViewSegmentNumbers *
ForwardProjectorByBinNiftyPET::
get_symmetries_used() const
{
  if (!this->_already_set_up)
    error("ForwardProjectorByBin method called without calling set_up first.");
  return _symmetries_sptr.get();
}

void
ForwardProjectorByBinNiftyPET::
actual_forward_project(RelatedViewgrams<float>&, 
      const DiscretisedDensity<3,float>&,
        const int min_axial_pos_num, const int max_axial_pos_num,
        const int min_tangential_pos_num, const int max_tangential_pos_num)
{
    throw std::runtime_error("Need to use set_input() if wanting to use ForwardProjectorByBinNiftyPET.");
}

void
ForwardProjectorByBinNiftyPET::
actual_forward_project(RelatedViewgrams<float>& viewgrams,
        const int min_axial_pos_num, const int max_axial_pos_num,
        const int min_tangential_pos_num, const int max_tangential_pos_num)
{
//    if (min_axial_pos_num != _proj_data_info_sptr->get_min_axial_pos_num() ||
//         … )
//       error();

    viewgrams = _projected_data_sptr->get_related_viewgrams(
        viewgrams. get_basic_view_segment_num(), _symmetries_sptr);
}

void
ForwardProjectorByBinNiftyPET::
set_input(const shared_ptr<DiscretisedDensity<3,float> >& density_sptr)
{
    _density_sptr.reset(density_sptr->clone());
    throw std::runtime_error("This method needs to do the actual GPU projection.");
}

END_NAMESPACE_STIR
