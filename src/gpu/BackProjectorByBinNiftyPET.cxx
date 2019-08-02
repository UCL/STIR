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
#include <prjb.h>
#include <auxmath.h>

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

    // Set up the niftyPET binary helper
    _helper.set_li2rng_filename("li2rng.dat"  );
    _helper.set_li2sn_filename ("li2sn.dat"   );
    _helper.set_li2nos_filename("li2nos.dat"  );
    _helper.set_s2c_filename   ("s2c.dat"     );
    _helper.set_aw2ali_filename("aw2ali.dat"  );
    _helper.set_crs_filename   ( "crss.dat"   );
    _helper.set_cuda_device_id ( _cuda_device );
    _helper.set_span           ( _proj_data_info_sptr->get_num_segments() );
    std::cout << "\n\n TODO still need to check att\n\n";
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
    // --------------------------------------------------------------- //
    //   Get arguments
    // --------------------------------------------------------------- //

    std::vector<float> li2rng = _helper.get_li2rng();
    std::vector<short> li2sn  = _helper.get_li2sn();
    std::vector<char>  li2nos = _helper.get_li2nos();
    std::vector<short> s2c    = _helper.get_s2c();
    std::vector<int  > aw2ali = _helper.get_aw2ali();
    std::vector<float> crs    = _helper.get_crs();
    std::vector<int>   isub   = _helper.get_isub();
    Cnst Cnt                  = _helper.get_cnst();
    int Naw                   = _helper.get_naw();
    int n0crs                 = _helper.get_n0crs();
    int n1crs                 = _helper.get_n1crs();
    char att                  = _helper.get_att();
    int nsinos                = _helper.get_nsinos();

    // --------------------------------------------------------------- //
    //   STIR -> NiftyPET projection data conversion
    // --------------------------------------------------------------- //

    std::vector<float> sino_w_gaps = _helper.create_niftyPET_sinogram_with_gaps();
    _helper.convert_proj_data_stir_to_niftyPET(sino_w_gaps,proj_data);

    // --------------------------------------------------------------- //
    //   Remove gaps from sinogram
    // --------------------------------------------------------------- //

    std::vector<float> sinog = _helper.create_niftyPET_sinogram_no_gaps();
    remove_gaps(sinog.data(),sino_w_gaps.data(),nsinos,aw2ali.data(),Cnt);

    // --------------------------------------------------------------- //
    //   Back project
    // --------------------------------------------------------------- //

    std::vector<float> np_im = _helper.create_niftyPET_image();

    gpu_bprj(np_im.data(),sinog.data(),
             li2rng.data(),li2sn.data(),li2nos.data(),s2c.data(),aw2ali.data(),crs.data(),
             isub.data(), int(isub.size()),
             Naw,n0crs,n1crs,
             Cnt);

    // --------------------------------------------------------------- //
    //   NiftyPET -> STIR image conversion
    // --------------------------------------------------------------- //

    _helper.convert_image_niftyPET_to_stir(*_density_sptr,np_im);
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
actual_back_project(const RelatedViewgrams<float>&,
                         const int, const int,
                         const int, const int)
{
    // TODO - dont think we do anything here...
}

END_NAMESPACE_STIR
