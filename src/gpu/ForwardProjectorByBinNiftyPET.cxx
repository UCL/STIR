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

#include <fstream>
#include "stir/gpu/ForwardProjectorByBinNiftyPET.h"
#include "stir/gpu/ProjectorByBinNiftyPETHelper.h"
#include "stir/RelatedViewgrams.h"
#include <prjf.h>
#include <auxmath.h>

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const
ForwardProjectorByBinNiftyPET::registered_name =
  "NiftyPET";

ForwardProjectorByBinNiftyPET::ForwardProjectorByBinNiftyPET() :
    _cuda_device(0)
{
    this->_already_set_up = false;
}

ForwardProjectorByBinNiftyPET::~ForwardProjectorByBinNiftyPET()
{
}

void
ForwardProjectorByBinNiftyPET::
initialise_keymap()
{
  parser.add_start_key("Forward Projector Using NiftyPET Parameters");
  parser.add_stop_key("End Forward Projector Using NiftyPET Parameters");
  parser.add_key("CUDA device", &_cuda_device);
}

void
ForwardProjectorByBinNiftyPET::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
       const shared_ptr<DiscretisedDensity<3,float> >& density_info_sptr)
{
    ForwardProjectorByBin::set_up(proj_data_info_sptr,density_info_sptr);
    check(*this->_proj_data_info_sptr, *_density_sptr);
    _symmetries_sptr.reset(new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_sptr, density_info_sptr));

    // Initialise projected_data_sptr from this->_proj_data_info_sptr
    _projected_data_sptr.reset(
                new ProjDataInMemory(this->_density_sptr->get_exam_info_sptr(), this->_proj_data_info_sptr));

    // Set up the niftyPET binary helper
    _helper.set_li2rng_filename("li2rng.dat"  );
    _helper.set_li2sn_filename ("li2sn.dat"   );
    _helper.set_li2nos_filename("li2nos.dat"  );
    _helper.set_s2c_filename   ("s2c.dat"     );
    _helper.set_aw2ali_filename("aw2ali.dat"  );
    _helper.set_crs_filename   ( "crss.dat"   );
    _helper.set_cuda_device_id ( _cuda_device );
    _helper.set_span           ( char(_projected_data_sptr->get_num_segments()) );
    std::cout << "\n\n TODO still need to check att\n\n";
    _helper.set_att(0);
    _helper.set_up();
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
        const int, const int,
        const int, const int)
{
    throw std::runtime_error("Need to use set_input() if wanting to use ForwardProjectorByBinNiftyPET.");
}

void
ForwardProjectorByBinNiftyPET::
actual_forward_project(RelatedViewgrams<float>& viewgrams,
        const int, const int,
        const int, const int)
{
//    if (min_axial_pos_num != _proj_data_info_sptr->get_min_axial_pos_num() ||
//         … )
//       error();

    viewgrams = _projected_data_sptr->get_related_viewgrams(
        viewgrams.get_basic_view_segment_num(), _symmetries_sptr);
}

void
ForwardProjectorByBinNiftyPET::
set_input(const shared_ptr<DiscretisedDensity<3,float> >& density_sptr)
{
    // TODO: required?
    _density_sptr.reset(density_sptr->clone());

    // Create NiftyPET image
    std::vector<float> np_vec = _helper.create_niftyPET_image();
    _helper.convert_image_stir_to_niftyPET(np_vec,*_density_sptr);

    // --------------------------------------------------------------- //
    //   Other arguments
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

    std::vector<float> sinog  = _helper.create_niftyPET_sinogram_no_gaps();

    // --------------------------------------------------------------- //
    //   Do the forward projection!
    // --------------------------------------------------------------- //

    gpu_fprj(sinog.data(), np_vec.data(),
        li2rng.data(), li2sn.data(), li2nos.data(), s2c.data(), aw2ali.data(), crs.data(),
        isub.data(), int(isub.size()),
        Naw, n0crs, n1crs,
        Cnt, att);

    // --------------------------------------------------------------- //
    //   Put gaps back into sinogram
    // --------------------------------------------------------------- //

    std::vector<float> sino_w_gaps = _helper.create_niftyPET_sinogram_with_gaps();
    put_gaps(sino_w_gaps.data(),sinog.data(),aw2ali.data(),Cnt);

    // --------------------------------------------------------------- //
    //   NiftyPET -> STIR projection data conversion
    // --------------------------------------------------------------- //

    _helper.convert_proj_data_niftyPET_to_stir(*_projected_data_sptr,sino_w_gaps);
}

END_NAMESPACE_STIR
