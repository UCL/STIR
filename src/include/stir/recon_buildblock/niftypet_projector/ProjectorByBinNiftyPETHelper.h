//
//

#ifndef __stir_gpu_ProjectorByBinNiftyPETHelper_h__
#define __stir_gpu_ProjectorByBinNiftyPETHelper_h__
/*!
  \file
  \ingroup projection

  \brief Helper functions for projection using NiftyPET's GPU implementation.

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

#include "stir/common.h"
#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ProjData.h"

// Forward declaration of Cnst
struct Cnst;

START_NAMESPACE_STIR

/*!
  \ingroup projection
  \brief Abstract base class for all forward projectors
*/
class ProjectorByBinNiftyPETHelper
{
public:

    /// Default constructor
    ProjectorByBinNiftyPETHelper() :
        _already_set_up(false), _span(-1), _devid(0), _att(-1) {}

    /// Destructor
    ~ProjectorByBinNiftyPETHelper(){}

    /// Set li2rng filename
    void set_li2rng_filename(const std::string &fname_li2rng) { _fname_li2rng = fname_li2rng; }

    /// Set li2sn filename
    void set_li2sn_filename(const std::string &fname_li2sn)   { _fname_li2sn = fname_li2sn;   }

    /// Set li2nos filename
    void set_li2nos_filename(const std::string &fname_li2nos) { _fname_li2nos = fname_li2nos; }

    /// Set s2c filename
    void set_s2c_filename(const std::string &fname_s2c)       { _fname_s2c = fname_s2c;       }

    /// Set aw2ali filename
    void set_aw2ali_filename(const std::string &fname_aw2ali) { _fname_aw2ali = fname_aw2ali; }

    /// Set crs filename
    void set_crs_filename(const std::string &fname_crs)       { _fname_crs = fname_crs;       }

    /// Set CUDA device ID
    void set_cuda_device_id(const int devid)                 { _devid = char(devid);          }

    /// Set span
    void set_span(const char span)                            { _span = span;                 }

    /// Set emission (0) or transmission (1) - whether to exp{-result} for attenuation maps
    void set_att(const char att)                              { _att = att;                   }

    /// Set verbosity level for CUDA output
    void set_verbose(const bool verbose)                      { _verbose = verbose;           }

    /// Set up
    void set_up();

    /// Create NiftyPET image
    static std::vector<float> create_niftyPET_image();

    /// Create NiftyPET singram with no gaps. Forward project into this
    std::vector<float> create_niftyPET_sinogram_no_gaps() const;

    /// Create NiftyPET singram with gaps. Use this before converting to stir.
    std::vector<float> create_niftyPET_sinogram_with_gaps() const;

    /// Convert STIR image to NiftyPET image
    static void convert_image_stir_to_niftyPET(std::vector<float> &np, const DiscretisedDensity<3,float> &stir);

    /// Convert NiftyPET image to STIR image
    static void convert_image_niftyPET_to_stir(DiscretisedDensity<3,float> &stir, const std::vector<float> &np_vec);

    /// Convert STIR proj data to NiftyPET proj data
    void convert_proj_data_stir_to_niftyPET(std::vector<float> &np_vec, const ProjData& stir) const;

    ///
    void convert_viewgram_stir_to_niftyPET(std::vector<float> &np_vec, const Viewgram<float>& viewgram) const;

    /// Convert NiftyPET proj data to STIR proj data
    void convert_proj_data_niftyPET_to_stir(ProjData &stir_sptr, const std::vector<float> &np_vec) const;

    /// Remove gaps from sinogram. Do some unavoidable const_casting as the wrapped methods don't use const
    void remove_gaps(std::vector<float> &sino_no_gaps, const std::vector<float> &sino_w_gaps) const;

    /// Put gaps into sinogram. Do some unavoidable const_casting as the wrapped methods don't use const
    void put_gaps(std::vector<float> &sino_w_gaps, const std::vector<float> &sino_no_gaps) const;

    /// Back project. Do some unavoidable const_casting as the wrapped methods don't use const
    void back_project(std::vector<float> &image, const std::vector<float> &sino_no_gaps) const;

    /// Forward project, returns sinogram without gaps. Do some unavoidable const_casting as the wrapped methods don't use const
    void forward_project(std::vector<float> &sino_no_gaps, const std::vector<float> &image) const;

private:

    /// Check that set up has been run before returning data
    void check_set_up() const;

    /// Permute the data
    void permute(std::vector<float> &output_array, const std::vector<float> &orig_array, const unsigned output_dims[3], const unsigned *permute_order) const;

    /// Convert 3d niftypet proj data index to 1d
    unsigned convert_niftypet_proj_3d_to_1d_idx(const unsigned ang, const unsigned bins, const unsigned sino) const;

    /// Convert 1d niftypet proj data index to 3d
    void convert_niftypet_proj_1d_to_3d_idx(unsigned &ang, unsigned &bins, unsigned &sino, const unsigned idx) const;

    /// Get Cnst
    const Cnst & get_cnst() const { check_set_up(); return *_cnt; }

    /// Get Cnst
    Cnst & get_cnst() { check_set_up(); return *_cnt; }

    /// Get Naw - number of active bins in 2d sino
    static int get_naw();// { return AW; }

    /// Get n0crs
    static int get_n0crs();// { return 4; } // not sure which one this is in def.h

    /// Get n1crs
    static int get_n1crs();// { return nCRS; }

    float _niftypet_to_stir_ratio;
    bool _already_set_up;
    std::string _fname_li2rng, _fname_li2sn, _fname_li2nos, _fname_s2c, _fname_aw2ali, _fname_crs;
    std::vector<float> _li2rng;
    std::vector<short> _li2sn;
    std::vector<char>  _li2nos;
    std::vector<short> _s2c;
    std::vector<int>   _aw2ali;
    std::vector<float> _crs;
    char _span;
    char _devid;
    shared_ptr<Cnst> _cnt;
    int _nsinos;
    char _att;
    std::vector<int> _isub;
    bool _verbose;
};

END_NAMESPACE_STIR

#endif // __stir_gpu_ProjectorByBinNiftyPETHelper_h__
