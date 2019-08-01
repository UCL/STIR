//
//
/*!
  \file
  \ingroup projection

  \brief non-inline implementations for stir::BackProjectorByBinNiftyPET

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


#include "stir/gpu/BackProjectorByBinNiftyPET.h"
#include "stir/DiscretisedDensity.h"
#include <prjb.h>

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const
BackProjectorByBinNiftyPET::registered_name =
  "NiftyPET";

BackProjectorByBinNiftyPET::BackProjectorByBinNiftyPET()
{
    this->_already_set_up = false;
}

BackProjectorByBinNiftyPET::~BackProjectorByBinNiftyPET()
{
}

void
BackProjectorByBinNiftyPET::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr, 
       const shared_ptr<DiscretisedDensity<3,float> >& density_info_sptr)
{
    BackProjectorByBin::set_up(proj_data_info_sptr,density_info_sptr);
    check(*this->_proj_data_info_sptr, *_density_sptr);
    _symmetries_sptr.reset(new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_sptr, density_info_sptr));
}

const DataSymmetriesForViewSegmentNumbers *
BackProjectorByBinNiftyPET::
get_symmetries_used() const
{
  if (!this->_already_set_up)
    error("BackProjectorByBin method called without calling set_up first.");
  return _symmetries_sptr.get();
}

void
BackProjectorByBinNiftyPET::
start_accumulating_in_new_image()
{
    _density_sptr->fill(0.);

    // TODO - Actual back projection
}

void
BackProjectorByBinNiftyPET::
get_output(DiscretisedDensity<3,float> &density) const
{
    if (!density.has_same_characteristics(*_density_sptr))
            error("Images should have similar characteristics.");
    std::copy(_density_sptr->begin_all(), _density_sptr->end_all(), density.begin_all());
}

void
BackProjectorByBinNiftyPET::
actual_back_project(DiscretisedDensity<3,float>&,
                    const RelatedViewgrams<float>&,
                         const int, const int,
                         const int, const int)
{
    throw std::runtime_error("Need to use set_input() if wanting to use BackProjectorByBinNiftyPET.");
}

void
BackProjectorByBinNiftyPET::
actual_back_project(const RelatedViewgrams<float>& viewgrams,
                         const int min_axial_pos_num, const int max_axial_pos_num,
                         const int min_tangential_pos_num, const int max_tangential_pos_num)
{
    // TODO - dont think we do anything here...
}

END_NAMESPACE_STIR
