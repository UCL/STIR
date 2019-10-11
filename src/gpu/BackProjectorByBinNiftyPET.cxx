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

#include <fstream>
#include "stir/gpu/BackProjectorByBinNiftyPET.h"
#include "stir/gpu/ProjectorByBinNiftyPETHelper.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ProjDataInMemory.h"

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const
BackProjectorByBinNiftyPET::registered_name =
  "NiftyPET";

BackProjectorByBinNiftyPET::BackProjectorByBinNiftyPET() :
    _cuda_device(0)
{
    this->_already_set_up = false;
}

BackProjectorByBinNiftyPET::~BackProjectorByBinNiftyPET()
{
}

void
BackProjectorByBinNiftyPET::
initialise_keymap()
{
  parser.add_start_key("Back Projector Using NiftyPET Parameters");
  parser.add_stop_key("End Back Projector Using NiftyPET Parameters");
  parser.add_key("CUDA device", &_cuda_device);
}

void
BackProjectorByBinNiftyPET::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr, 
       const shared_ptr<DiscretisedDensity<3,float> >& density_info_sptr)
{
    BackProjectorByBin::set_up(proj_data_info_sptr,density_info_sptr);
    check(*this->_proj_data_info_sptr, *_density_sptr);
    _symmetries_sptr.reset(new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_sptr, density_info_sptr));

    // For the NiftyPET projector, we need to create an empty projdata. This
    // will slowly get filled up by actual_back_project. And then get_output
    // will use the now-filled projdata to back project.
    _proj_data_sptr.reset(
                new ProjDataInMemory(
                    density_info_sptr->get_exam_info_sptr(),
                    proj_data_info_sptr,
                    false));

    // Set up the niftyPET binary helper
    _helper.set_li2rng_filename("li2rng.dat"  );
    _helper.set_li2sn_filename ("li2sn.dat"   );
    _helper.set_li2nos_filename("li2nos.dat"  );
    _helper.set_s2c_filename   ("s2c.dat"     );
    _helper.set_aw2ali_filename("aw2ali.dat"  );
    _helper.set_crs_filename   ( "crss.dat"   );
    _helper.set_cuda_device_id ( _cuda_device );
    _helper.set_span           ( char(_proj_data_info_sptr->get_num_segments()) );
    _helper.set_att(0);
    _helper.set_up();
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
back_project(const ProjData& proj_data, int, int)
{
    // Copy the data across. Don't actually do the back projection yet.
    _proj_data_sptr.reset(new ProjDataInMemory(proj_data));
}

void
BackProjectorByBinNiftyPET::
get_output(DiscretisedDensity<3,float> &density) const
{
    // --------------------------------------------------------------- //
    //   STIR -> NiftyPET projection data conversion
    // --------------------------------------------------------------- //

    std::vector<float> sino_w_gaps = _helper.create_niftyPET_sinogram_with_gaps();
    _helper.convert_proj_data_stir_to_niftyPET(sino_w_gaps,*_proj_data_sptr);

    // --------------------------------------------------------------- //
    //   Remove gaps from sinogram
    // --------------------------------------------------------------- //

    std::vector<float> sino_no_gaps = _helper.create_niftyPET_sinogram_no_gaps();
    _helper.remove_gaps(sino_no_gaps, sino_w_gaps);

    // --------------------------------------------------------------- //
    //   Back project
    // --------------------------------------------------------------- //

    std::vector<float> np_im = _helper.create_niftyPET_image();
    _helper.back_project(np_im,sino_no_gaps);

    // --------------------------------------------------------------- //
    //   NiftyPET -> STIR image conversion
    // --------------------------------------------------------------- //

    _helper.convert_image_niftyPET_to_stir(density,np_im);
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
actual_back_project(const RelatedViewgrams<float>& related_viewgrams,
                         const int, const int,
                         const int, const int)
{
    _proj_data_sptr->set_related_viewgrams(related_viewgrams);
}

END_NAMESPACE_STIR
