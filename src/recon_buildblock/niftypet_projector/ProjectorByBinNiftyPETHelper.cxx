//
//
/*!

  \file
  \ingroup projection

  \brief non-inline implementations for stir::ProjectorByBinNiftyPETHelper

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
#include "stir/recon_buildblock/niftypet_projector/ProjectorByBinNiftyPETHelper.h"
#include "def.h"
#include <boost/format.hpp>
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/is_null_ptr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_array_functions.h"
#include "driver_types.h"
#include "auxmath.h"
#include "prjb.h"
#include "prjf.h"
#include "scanner_0.h"
#include "recon.h"
#include "lmproc.h"

START_NAMESPACE_STIR

/// Read NiftyPET binary file
template <class dataType>
std::vector<dataType>
ProjectorByBinNiftyPETHelper::
read_binary_file(const std::string &data_path)
{
    std::ifstream file(data_path, std::ios::in | std::ios::binary);

    // get its size:
    file.seekg(0, std::ios::end);
    long file_size = file.tellg();
    unsigned long num_elements = static_cast<unsigned long>(file_size) / static_cast<unsigned long>(sizeof(dataType));
    file.seekg(0, std::ios::beg);

    std::vector<dataType> contents(num_elements);
    file.read(reinterpret_cast<char*>(contents.data()), file_size);

    return contents;
}

template <class dataType>
static
std::vector<dataType>
read_binary_file_in_examples(const std::string &file_name)
{
    const char* stir_path = std::getenv("STIR_PATH");
    if (!stir_path)
        throw std::runtime_error("STIR_PATH not defined, cannot find data");

    std::string data_path = stir_path;
    data_path += "/examples/niftypet_mMR_params/" + file_name;
    return ProjectorByBinNiftyPETHelper::read_binary_file<dataType>(data_path);
}

void
ProjectorByBinNiftyPETHelper::
set_up()
{

    // Intensities from projections do not have to match between
    // reconstruction packages. To account for that we need to
    // divide by this value after forward projection and multiply
    // after back projection.
    // This value is fixed for the mMR, but may have to change
    // if other scanners are incorporated.
    _niftypet_to_stir_ratio = 1.f;//.25f;

    if (_span < 0)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::set_up() "
                                "sinogram span not set.");

    if (_att < 0)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::set_up() "
                                "emission or transmission mode (att) not set.");

    // Read binaries
    if (_fname_li2rng.size()     == 0 ||
            _fname_li2sn.size()  == 0 ||
            _fname_li2nos.size() == 0 ||
            _fname_s2c.size()    == 0 ||
            _fname_aw2ali.size() == 0 ||
            _fname_crs.size()    == 0)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::set_up() "
                                 "not all filenames have been set.");

    _li2rng   = read_binary_file_in_examples<float>(_fname_li2rng);
    _li2sn    = read_binary_file_in_examples<short>(_fname_li2sn );
    _li2nos   = read_binary_file_in_examples<char> (_fname_li2nos);
    _s2c      = read_binary_file_in_examples<short>(_fname_s2c   );
    _aw2ali   = read_binary_file_in_examples<int>  (_fname_aw2ali);
    _crs      = read_binary_file_in_examples<float>(_fname_crs   );

    // Set up cnst - backwards engineered from def.h, scanner.h and resources.py
    _cnt.reset(new Cnst);
    _cnt->SPN      = _span;
    _cnt->RNG_STRT = 0;
    _cnt->RNG_END  = NRINGS;
    _cnt->VERBOSE  = true;
    _cnt->DEVID    = _devid;
    _cnt->NSN11    = NSINOS11;
    _cnt->NSEG0    = SEG0;
    _cnt->NCRS     = nCRS;
    _cnt->OFFGAP   = 1;
    _cnt->TGAP     = 9;
    _cnt->A        = NSANGLES;
    _cnt->W        = NSBINS;
    _cnt->NCRSR    = nCRSR;
    _cnt->B        = NBUCKTS;

    _cnt->MRD =  mxRD;
    _cnt->ALPHA =  aLPHA;
    _cnt->AXR =  SZ_RING;
    _cnt->BTP =  0;
    _cnt->BTPRT =  1.0;
    _cnt->COSUPSMX =  0.725f;
    _cnt->COSSTP = (1-_cnt->COSUPSMX)/(255);
    _cnt->ETHRLD =  0.05f;
    _cnt->NRNG =  NRINGS;
    _cnt->ITOFBIND =  0.08552925517901334f;
    _cnt->NSN1 =  NSINOS;
    _cnt->NSN64 =  4096;
    _cnt->NSRNG =  8;
    _cnt->RE =  33.47f;
    _cnt->TOFBIND =  11.691905862f;
    _cnt->TOFBINN =  1;
    _cnt->TOFBINS =  3.9e-10f;

    switch(_cnt->SPN){
      case 11:
        _nsinos = _cnt->NSN11; break;
      case 1:
        _nsinos = _cnt->NSEG0; break;
      default:
        throw std::runtime_error("Unsupported span");
    }

    // isub
    _isub = std::vector<int>(unsigned(AW));
    for (unsigned i = 0; i<unsigned(AW); i++) _isub[i] = int(i);

    _already_set_up = true;
}

void
ProjectorByBinNiftyPETHelper::
check_set_up() const
{
    if (!_already_set_up)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::check_set_up() "
                                 "Make sure filenames have been set and set_up has been run.");
}

std::vector<float>
ProjectorByBinNiftyPETHelper::
create_niftyPET_image()
{
    return std::vector<float>(SZ_IMZ*SZ_IMX*SZ_IMY,0);
}

std::vector<float>
ProjectorByBinNiftyPETHelper::
create_niftyPET_sinogram_no_gaps() const
{
    check_set_up();
    return std::vector<float>(_isub.size() * static_cast<unsigned long>(_nsinos), 0);
}

std::vector<float>
ProjectorByBinNiftyPETHelper::
create_niftyPET_sinogram_with_gaps() const
{
    return std::vector<float>(NSBINS*NSANGLES*unsigned(_nsinos), 0);
}

void get_stir_indices_and_dims(int stir_dim[3], Coordinate3D<int> &min_indices, Coordinate3D<int> &max_indices, const DiscretisedDensity<3,float >&stir)
{
    if (!stir.get_regular_range(min_indices, max_indices))
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::set_input - "
                                 "expected image to have regular range.");
    for (int i=0; i<3; ++i)
        stir_dim[i] = max_indices[i + 1] - min_indices[i + 1] + 1;
}

unsigned convert_niftypet_im_3d_to_1d_idx(const unsigned x, const unsigned y, const unsigned z)
{
    return z*SZ_IMX*SZ_IMY + y*SZ_IMX + x;
}

unsigned
ProjectorByBinNiftyPETHelper::
convert_niftypet_proj_3d_to_1d_idx(const unsigned ang, const unsigned bins, const unsigned sino) const
{
    return sino*NSANGLES*NSBINS + ang*NSBINS + bins;
}

void
ProjectorByBinNiftyPETHelper::
permute(std::vector<float> &output_array, const std::vector<float> &orig_array, const unsigned output_dims[3], const unsigned permute_order[3]) const
{
#ifndef NDEBUG
    // Check that in the permute order, each number is between 0 and 2 (can't be <0 because it's unsigned)
    for (unsigned i=0; i<3; ++i)
        if (permute_order[i]>2)
            throw std::runtime_error("Permute order values should be between 0 and 2.");
    // Check that each number is unique
    for (unsigned i=0; i<3; ++i)
        for (unsigned j=i+1; j<3; ++j)
            if (permute_order[i] == permute_order[j])
                throw std::runtime_error("Permute order values should be unique.");
    // Check that size of output_dims==arr.size()
    assert(orig_array.size() == output_dims[0]*output_dims[1]*output_dims[2]);
    // Check that output array is same size as input array
    assert(orig_array.size() == output_array.size());
#endif

    // Calculate old dimensions
    unsigned old_dims[3];
    for (unsigned i=0; i<3; ++i)
        old_dims[permute_order[i]] = output_dims[i];

    // Loop over all elements
    unsigned old_3d_idx[3], new_3d_idx[3], new_1d_idx;
    for (unsigned old_1d_idx=0; old_1d_idx<orig_array.size(); ++old_1d_idx) {

        // From the 1d index, generate the old 3d index
        old_3d_idx[2] =  old_1d_idx %  old_dims[2];
        old_3d_idx[1] = (old_1d_idx /  old_dims[2]) % old_dims[1];
        old_3d_idx[0] =  old_1d_idx / (old_dims[2]  * old_dims[1]);

        // Get the corresponding new 3d index
        for (unsigned i=0; i<3; ++i)
            new_3d_idx[i] = old_3d_idx[permute_order[i]];

        // Get the new 1d index from the new 3d index
        new_1d_idx = new_3d_idx[0]*output_dims[2]*output_dims[1] + new_3d_idx[1]*output_dims[2] + new_3d_idx[2];

        // Fill the data
        output_array[new_1d_idx] = orig_array[old_1d_idx];
    }
}

void
ProjectorByBinNiftyPETHelper::
remove_gaps(std::vector<float> &sino_no_gaps, const std::vector<float> &sino_w_gaps) const
{
    check_set_up();
    assert(!sino_no_gaps.empty());

    if (_verbose)
        getMemUse();

    ::remove_gaps(sino_no_gaps.data(),
                  const_cast<std::vector<float>&>(sino_w_gaps).data(),
                  _nsinos,
                  const_cast<std::vector<int  >&>(_aw2ali).data(),
                  *_cnt);
}

void
ProjectorByBinNiftyPETHelper::
put_gaps(std::vector<float> &sino_w_gaps, const std::vector<float> &sino_no_gaps) const
{
    check_set_up();
    assert(!sino_w_gaps.empty());

    std::vector<float> unpermuted_sino_w_gaps = this->create_niftyPET_sinogram_with_gaps();

    if (_verbose)
        getMemUse();

    ::put_gaps(unpermuted_sino_w_gaps.data(),
               const_cast<std::vector<float>&>(sino_no_gaps).data(),
               const_cast<std::vector<int  >&>(_aw2ali).data(),
               *_cnt);

    // Permute the data (as this is done on the NiftyPET python side after put gaps
    unsigned output_dims[3] = {837, 252, 344};
    unsigned permute_order[3] = {2,0,1};
    this->permute(sino_w_gaps,unpermuted_sino_w_gaps,output_dims,permute_order);
}

void
ProjectorByBinNiftyPETHelper::
back_project(std::vector<float> &image, const std::vector<float> &sino_no_gaps) const
{
    check_set_up();
    assert(!image.empty());

    std::vector<float> unpermuted_image = this->create_niftyPET_image();

    if (_verbose)
        getMemUse();

    gpu_bprj(unpermuted_image.data(),
             const_cast<std::vector<float>&>(sino_no_gaps).data(),
             const_cast<std::vector<float>&>(_li2rng).data(),
             const_cast<std::vector<short>&>(_li2sn).data(),
             const_cast<std::vector<char >&>(_li2nos).data(),
             const_cast<std::vector<short>&>(_s2c).data(),
             const_cast<std::vector<int  >&>(_aw2ali).data(),
             const_cast<std::vector<float>&>(_crs).data(),
             const_cast<std::vector<int  >&>(_isub).data(),
             int(_isub.size()),
             this->get_naw(),
             this->get_n0crs(),
             this->get_n1crs(),
             *_cnt);

    // Permute the data (as this is done on the NiftyPET python side after back projection
    unsigned output_dims[3] = {127,320,320};
    unsigned permute_order[3] = {2,0,1};
    this->permute(image,unpermuted_image,output_dims,permute_order);

    // Scale to account for niftypet-to-stir ratio
    for (unsigned i=0; i<image.size(); ++i)
        image[i] *= _niftypet_to_stir_ratio;
}

void
ProjectorByBinNiftyPETHelper::
forward_project(std::vector<float> &sino_no_gaps, const std::vector<float> &image) const
{
    check_set_up();
    assert(!sino_no_gaps.empty());

    // Permute the data (as this is done on the NiftyPET python side before forward projection
    unsigned output_dims[3] = {320,320,127};
    unsigned permute_order[3] = {1,2,0};
    std::vector<float> permuted_image = this->create_niftyPET_image();
    this->permute(permuted_image,image,output_dims,permute_order);

    if (_verbose)
        getMemUse();

    gpu_fprj(sino_no_gaps.data(),
             permuted_image.data(),
             const_cast<std::vector<float>&>(_li2rng).data(),
             const_cast<std::vector<short>&>(_li2sn).data(),
             const_cast<std::vector<char >&>(_li2nos).data(),
             const_cast<std::vector<short>&>(_s2c).data(),
             const_cast<std::vector<int  >&>(_aw2ali).data(),
             const_cast<std::vector<float>&>(_crs).data(),
             const_cast<std::vector<int  >&>(_isub).data(),
             int(_isub.size()),
             this->get_naw(),
             this->get_n0crs(),
             this->get_n1crs(),
             *_cnt,
             _att);

    // Scale to account for niftypet-to-stir ratio
    for (unsigned i=0; i<sino_no_gaps.size(); ++i)
        sino_no_gaps[i] /= _niftypet_to_stir_ratio;
}

void
ProjectorByBinNiftyPETHelper::
lm_to_proj_data() const
{
    // Get listmode info
    std::string lm_binary_file = "/home/rich/Documents/Data/NiftyPET_example/LM/17598013_1946_20150604155500.000000.bf";
    char *flm = new char[lm_binary_file.length() + 1];
    strcpy(flm, lm_binary_file.c_str());
    getLMinfo(flm, *_cnt);

    // preallocate all the output arrays - in def.h VTIME=2 (), MXNITAG=5400 (max time 1h30)
    const int nitag = lmprop.nitag;
    const int pow_2_MXNITAG = pow(2,VTIME);
    int tn;
    if (nitag>MXNITAG)
        tn = MXNITAG/pow_2_MXNITAG;
    else
        tn = (nitag+pow_2_MXNITAG-1)/pow_2_MXNITAG;

    unsigned short frames(0);
    int nfrm(1);
    int tstart(0), tstop(3600);

    // structure of output data
    // var   | type               | python var | description                      | shape
    // ------+--------------------|------------+----------------------------------+-----------------------------------------------------------------
    // nitag | int                |            | gets set inside lmproc           | 
	// sne   | int                |            | gets set inside lmproc           | 
    // snv   | unsigned int *     | pvs        | sino views                       | [ tn,           Cnt['NSEG0'],    Cnt['NSBINS']                  ]
    // hcp   | unsigned int *     | phc        | head curve prompts               | [ nitag                                                         ]
    // hcd   | unsigned int *     | dhc        | head curve delayeds              | [ nitag                                                         ]
    // fan   | unsigned int *     | fan        | fansums                          | [ nfrm,         Cnt['NRNG'],     Cnt['NCRS']                    ]
    // bck   | unsigned int *     | bck        | buckets (singles)                | [ 2,            nitag,           Cnt['NBCKT']                   ]
    // mss   | float *            | mss        | centre of mass (axially)         | [ nitag                                                         ]
    // ssr   | unsigned int *     | ssr        |                                  | [ Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']                  ]
    // psn   | void *             | psino      | if nfrm==1, unsigned int*        | [ nfrm,          nsinos,         Cnt['NSANGLES'], Cnt['NSBINS'] ]
    // dsn   | void *             | dsino      | if nfrm==1, unsigned int*        | [ nfrm,          nsinos,         Cnt['NSANGLES'], Cnt['NSBINS'] ]
    // psm   | unsigned long long |            | gets set inside lmproc           |
    // dsm   | unsigned long long |            | gets set inside lmproc           |
    // tot   | unsigned int       |            | gets set inside lmproc           |
    hstout dicout; 
    dicout.snv = new unsigned int[tn * _cnt->NSEG0 * _cnt->W];
    dicout.hcp = new unsigned int[nitag];
    dicout.hcd = new unsigned int[nitag];
    dicout.fan = new unsigned int[nfrm * _cnt->NRNG * _cnt->NCRS];
    dicout.bck = new unsigned int[2 * nitag * _cnt->B];
    dicout.mss = new float       [nitag];
    dicout.ssr = new unsigned int[_cnt->NSEG0 * _cnt->A * _cnt->W];
    if (nfrm == 1)  {
        dicout.psn =  new unsigned int[nfrm * _nsinos * _cnt->A * _cnt->W];
        dicout.dsn =  new unsigned int[nfrm * _nsinos * _cnt->A * _cnt->W];
    }
    else
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::lm_to_proj_data: If nfrm>1, "
                                  "dicout.psn and dicout.dsn should be unsigned char*. Not "
                                  "tested, but should be pretty easy.");

    // structure of axial LUTs for LM processing
    // var        | type    | required? | description                                              |
    // -----------+---------+-----------+----------------------------------------------------------|
    // li2rno     | int *   |           | linear indx to ring indx                                 |
	// li2sn      | int *   |           | linear michelogram index (along diagonals) to sino index |
    // li2nos     | int *   |           | linear indx to no of sinos in span-11                    |
    // sn1_rno    | short * | yes       |                                                          |
    // sn1_ssrb   | short * | yes       |                                                          |
    // sn1_sn11no | short * | yes       |                                                          |
    // Nli2rno    | int[2]  |           | array sizes                                              |
    // Nli2sn     | int[2]  |           |                                                          |
    // Nli2nos    | int     |           |                                                          |
    axialLUT axLUT;
    if (_fname_sn1_rno.empty() || _fname_sn1_sn11.empty() || _fname_sn1_ssrb.empty())
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::lm_to_proj_data: Filenames missing for "
                                    ".dat files required for unlisting.");
    std::vector<short> _sn1_rno   = read_binary_file_in_examples<short>(_fname_sn1_rno );
    std::vector<short> _sn1_sn11  = read_binary_file_in_examples<short>(_fname_sn1_sn11);
    std::vector<short> _sn1_ssrb  = read_binary_file_in_examples<short>(_fname_sn1_ssrb);
    axLUT.sn1_rno  = const_cast<std::vector<short>&>(_sn1_rno).data();
    axLUT.sn1_sn11 = const_cast<std::vector<short>&>(_sn1_sn11).data();
    axLUT.sn1_ssrb = const_cast<std::vector<short>&>(_sn1_ssrb).data();

    // check that s2c contains correct number of elements
    if (_s2c.size() != AW*2)
        throw std::runtime_error("ProjectorByBinNiftyPETHelper::lm_to_proj_data: Expected "
                                 "s2c to have AW*2 number of elements.");
    std::vector<LORcc> s2c(AW);
    for (unsigned i = 0; i<unsigned(AW); i++) {
        s2c[i].c0 = _s2c[ 2*i ];
        s2c[i].c1 = _s2c[2*i+1];
        if (i<10)
            std::cout << "i=" << i << ", c0=" << s2c[i].c0 << ", c1=" << s2c[i].c1 << "\n";
    }

    std::cout << "\n\n\n\nabout to start unlisting\n\n\n\n\n";
    lmproc(dicout, // hstout (struct): output
           flm, // char *: binary filename (.s, .bf)
           &frames, // unsigned short *: think for one frame, frames = 0
           nfrm, // int: num frames
           tstart, // int
           tstop, // int
           const_cast<std::vector<LORcc>&>(s2c).data(), // *LORcc (struct)
           axLUT, // axialLUT (struct)
           *_cnt); // Cnst (struct)

    // Clear up
    delete [] flm;
    delete [] dicout.snv;
    delete [] dicout.hcp;
    delete [] dicout.hcd;
    delete [] dicout.fan;
    delete [] dicout.bck;
    delete [] dicout.ssr;
    delete [] dicout.psn;
    delete [] dicout.dsn;
    std::cout << "\n\n\n\n\n made it! \n\n\n";
}

void check_im_sizes(const int stir_dim[3], const int np_dim[3])
{
    for (int i=0; i<3; ++i)
        if (stir_dim[i] != np_dim[i])
            throw std::runtime_error((boost::format(
                                      "ProjectorByBinNiftyPETHelper::check_im_sizes() - "
                                      "STIR image (%1%, %2%, %3%) should be == (%4%,%5%,%6%).")
                                      % stir_dim[0] % stir_dim[1] % stir_dim[2]
                                      % np_dim[0]   % np_dim[1]   % np_dim[2]).str());
}

void check_voxel_spacing(const DiscretisedDensity<3, float> &stir)
{
    // Requires image to be a VoxelsOnCartesianGrid
    const VoxelsOnCartesianGrid<float> &stir_vocg =
            dynamic_cast<const VoxelsOnCartesianGrid<float>&>(stir);
    const BasicCoordinate<3,float> stir_spacing = stir_vocg.get_grid_spacing();

    // Get NiftyPET image spacing (need to *10 for mm)
    float np_spacing[3] = { 10.f*SZ_VOXZ, 10.f*SZ_VOXY, 10.f*SZ_VOXY };

    for (unsigned i=0; i<3; ++i)
        if (std::abs(stir_spacing[int(i)+1] - np_spacing[i]) > 1e-4f)
            throw std::runtime_error((boost::format(
                                      "ProjectorByBinNiftyPETHelper::check_voxel_spacing() - "
                                      "STIR image (%1%, %2%, %3%) should be == (%4%,%5%,%6%).")
                                      % stir_spacing[1] % stir_spacing[2] % stir_spacing[3]
                                      % np_spacing[0]   % np_spacing[1]   % np_spacing[2]).str());
}

void
ProjectorByBinNiftyPETHelper::
convert_image_stir_to_niftyPET(std::vector<float> &np_vec, const DiscretisedDensity<3, float> &stir)
{
    // Get the dimensions of the input image
    Coordinate3D<int> min_indices;
    Coordinate3D<int> max_indices;
    int stir_dim[3];
    get_stir_indices_and_dims(stir_dim,min_indices,max_indices,stir);

    // NiftyPET requires the image to be (z,x,y)=(SZ_IMZ,SZ_IMX,SZ_IMY)
    // which at the time of writing was (127,320,320).
    const int np_dim[3] = {SZ_IMZ,SZ_IMX,SZ_IMY};
    check_im_sizes(stir_dim,np_dim);
    check_voxel_spacing(stir);

    // Copy data from STIR to NiftyPET image
    unsigned np_z, np_y, np_x, np_1d;
    for (int z = min_indices[1]; z <= max_indices[1]; z++) {
        for (int y = min_indices[2]; y <= max_indices[2]; y++) {
            for (int x = min_indices[3]; x <= max_indices[3]; x++) {
                // Convert the stir 3d index to a NiftyPET 1d index
                np_z = unsigned(z - min_indices[1]);
                np_y = unsigned(y - min_indices[2]);
                np_x = unsigned(x - min_indices[3]);
                np_1d = convert_niftypet_im_3d_to_1d_idx(np_x,np_y,np_z);
                np_vec[np_1d] = stir[z][y][x];
            }
        }
    }
}

void
ProjectorByBinNiftyPETHelper::
convert_image_niftyPET_to_stir(DiscretisedDensity<3,float> &stir, const std::vector<float> &np_vec)
{
    // Get the dimensions of the input image
    Coordinate3D<int> min_indices;
    Coordinate3D<int> max_indices;
    int stir_dim[3];
    get_stir_indices_and_dims(stir_dim,min_indices,max_indices,stir);

    // NiftyPET requires the image to be (z,x,y)=(SZ_IMZ,SZ_IMX,SZ_IMY)
    // which at the time of writing was (127,320,320).
    const int np_dim[3] = {SZ_IMZ,SZ_IMX,SZ_IMY};
    check_im_sizes(stir_dim,np_dim);
    check_voxel_spacing(stir);

    // Copy data from NiftyPET to STIR image
    unsigned np_z, np_y, np_x, np_1d;
    for (int z = min_indices[1]; z <= max_indices[1]; z++) {
        for (int y = min_indices[2]; y <= max_indices[2]; y++) {
            for (int x = min_indices[3]; x <= max_indices[3]; x++) {
                // Convert the stir 3d index to a NiftyPET 1d index
                np_z = unsigned(z - min_indices[1]);
                np_y = unsigned(y - min_indices[2]);
                np_x = unsigned(x - min_indices[3]);
                np_1d = convert_niftypet_im_3d_to_1d_idx(np_x,np_y,np_z);
                stir[z][y][x] = np_vec[np_1d];
            }
        }
    }

    // After the back projection, we enforce a truncation outside of the FOV.
    // This is because the NiftyPET FOV is smaller than the STIR FOV and this
    // could cause some voxel values to spiral out of control.
    // truncate_rim(stir,17);
}

void
get_vals_for_proj_data_conversion(std::vector<int> &sizes, std::vector<int> &segment_sequence,
                                  int &num_sinograms, int &min_view, int &max_view,
                                  int &min_tang_pos, int &max_tang_pos,
                                  const ProjDataInfo& proj_data_info, const std::vector<float> &np_vec)
{
    const ProjDataInfoCylindricalNoArcCorr * info_sptr =
            dynamic_cast<const ProjDataInfoCylindricalNoArcCorr *>(&proj_data_info);
    if (is_null_ptr(info_sptr))
        error("ProjectorByBinNiftyPETHelper: only works with cylindrical projection data without arc-correction");

    const int max_ring_diff   = info_sptr->get_max_ring_difference(info_sptr->get_max_segment_num());
    const int max_segment_num = info_sptr->get_max_segment_num();

    segment_sequence.resize(unsigned(2*max_ring_diff+1));
    sizes.resize(unsigned(2*max_ring_diff+1));
    segment_sequence[unsigned(0)]=0;
    sizes[0]=info_sptr->get_num_axial_poss(0);
    for (int segment_num=1; segment_num<=max_segment_num; ++segment_num) {
       segment_sequence[unsigned(2*segment_num-1)] = -segment_num;
       segment_sequence[unsigned(2*segment_num)] = segment_num;
       sizes [unsigned(2*segment_num-1)] =info_sptr->get_num_axial_poss(-segment_num);
       sizes [unsigned(2*segment_num)] =info_sptr->get_num_axial_poss(segment_num);
    }

    // Get dimensions of STIR sinogram
    min_view      = proj_data_info.get_min_view_num();
    max_view      = proj_data_info.get_max_view_num();
    min_tang_pos  = proj_data_info.get_min_tangential_pos_num();
    max_tang_pos  = proj_data_info.get_max_tangential_pos_num();


    num_sinograms = proj_data_info.get_num_axial_poss(0);
    for (int s=1; s<= proj_data_info.get_max_segment_num(); ++s)
        num_sinograms += 2* proj_data_info.get_num_axial_poss(s);

    int num_proj_data_elems = num_sinograms * (1+max_view-min_view) * (1+max_tang_pos-min_tang_pos);

    // Make sure they're the same size
    if (np_vec.size() != unsigned(num_proj_data_elems))
        error(boost::format(
                  "ProjectorByBinNiftyPETHelper::get_vals_for_proj_data_conversion "
                  "NiftyPET and STIR sinograms are different sizes (%1% for STIR versus %2% for NP")
              % num_proj_data_elems % np_vec.size());
}

void get_stir_segment_and_axial_pos_from_niftypet_sino(int &segment, int &axial_pos, const unsigned np_sino, const std::vector<int> &sizes, const std::vector<int> &segment_sequence)
{
    int z = int(np_sino);
    for (unsigned i=0; i<segment_sequence.size(); ++i) {
        if (z < sizes[i]) {
            axial_pos = z;
            segment = segment_sequence[i];
            return;
          }
        else {
            z -= sizes[i];
        }
    }
}

void get_niftypet_sino_from_stir_segment_and_axial_pos(unsigned &np_sino, const int segment, const int axial_pos, const std::vector<int> &sizes, const std::vector<int> &segment_sequence)
{
    np_sino = 0U;
    for (unsigned i=0; i<segment_sequence.size(); ++i) {
        if (segment == segment_sequence[i]) {
            np_sino += axial_pos;
            return;
          }
        else {
            np_sino += sizes[i];
        }
    }
    throw std::runtime_error("ProjectorByBinNiftyPETHelper::get_niftypet_sino_from_stir_segment_and_axial_pos(): Failed to find NiftyPET sinogram.");
}

void
ProjectorByBinNiftyPETHelper::
convert_viewgram_stir_to_niftyPET(std::vector<float> &np_vec, const Viewgram<float>& viewgram) const
{
    // Get the values (and LUT) to be able to switch between STIR and NiftyPET projDatas
    std::vector<int> sizes, segment_sequence;
    int num_sinograms, min_view, max_view, min_tang_pos, max_tang_pos;
    get_vals_for_proj_data_conversion(sizes, segment_sequence, num_sinograms, min_view, max_view,
                                      min_tang_pos, max_tang_pos, *viewgram.get_proj_data_info_sptr(), np_vec);

    const int segment = viewgram.get_segment_num();
    const int view = viewgram.get_view_num();

    // Loop over the STIR view and tangential position
    for (int ax_pos=viewgram.get_min_axial_pos_num(); ax_pos<=viewgram.get_max_axial_pos_num(); ++ax_pos) {

        unsigned np_sino;

        // Convert the NiftyPET sinogram to STIR's segment and axial position
        get_niftypet_sino_from_stir_segment_and_axial_pos(np_sino, segment, ax_pos, sizes, segment_sequence);

        for (int tang_pos=min_tang_pos; tang_pos<=max_tang_pos; ++tang_pos) {

            unsigned np_ang  = unsigned(view-min_view);
            unsigned np_bin  = unsigned(tang_pos-min_tang_pos);
            unsigned np_1d = convert_niftypet_proj_3d_to_1d_idx(np_ang,np_bin,np_sino);
            np_vec.at(np_1d) = viewgram.at(ax_pos).at(tang_pos);
        }
    }
}

void
ProjectorByBinNiftyPETHelper::
convert_proj_data_stir_to_niftyPET(std::vector<float> &np_vec, const ProjData& stir) const
{
    const int min_view = stir.get_min_view_num();
    const int max_view = stir.get_max_view_num();
    const int min_segment = stir.get_min_segment_num();
    const int max_segment = stir.get_max_segment_num();

    for (int view=min_view; view<=max_view; ++view) {
        for (int segment=min_segment; segment<=max_segment; ++segment) {
            convert_viewgram_stir_to_niftyPET(np_vec, stir.get_viewgram(view,segment));
        }
    }
}

void
ProjectorByBinNiftyPETHelper::
convert_proj_data_niftyPET_to_stir(ProjData &stir, const std::vector<float> &np_vec) const
{
    // Get the values (and LUT) to be able to switch between STIR and NiftyPET projDatas
    std::vector<int> sizes, segment_sequence;
    int num_sinograms, min_view, max_view, min_tang_pos, max_tang_pos;
    get_vals_for_proj_data_conversion(sizes, segment_sequence, num_sinograms, min_view, max_view,
                                      min_tang_pos, max_tang_pos, *stir.get_proj_data_info_sptr(), np_vec);

    int segment, axial_pos;
    // Loop over all NiftyPET sinograms
    for (unsigned np_sino = 0; np_sino < unsigned(num_sinograms); ++np_sino) {

        // Convert the NiftyPET sinogram to STIR's segment and axial position
        get_stir_segment_and_axial_pos_from_niftypet_sino(segment, axial_pos, np_sino, sizes, segment_sequence);

        // Get the corresponding STIR sinogram
        Sinogram<float> sino = stir.get_empty_sinogram(axial_pos,segment);

        // Loop over the STIR view and tangential position
        for (int view=min_view; view<=max_view; ++view) {
            for (int tang_pos=min_tang_pos; tang_pos<=max_tang_pos; ++tang_pos) {

                unsigned np_ang  = unsigned(view-min_view);
                unsigned np_bin  = unsigned(tang_pos-min_tang_pos);
                unsigned np_1d = convert_niftypet_proj_3d_to_1d_idx(np_ang,np_bin,np_sino);
                sino.at(view).at(tang_pos) = np_vec.at(np_1d);
            }
        }
        stir.set_sinogram(sino);
    }
}

int
ProjectorByBinNiftyPETHelper::
get_naw()
{
    return AW;
}

int
ProjectorByBinNiftyPETHelper::
get_n0crs()
{
    return 4;
}

int
ProjectorByBinNiftyPETHelper::
get_n1crs()
{
    return nCRS;
}

END_NAMESPACE_STIR
