//
//

#ifndef __stir_gpu_NiftyPETHelper_h__
#define __stir_gpu_NiftyPETHelper_h__
/*!
  \file
  \ingroup projection
  \ingroup NiftyPET

  \brief Helper class for NiftyPET's GPU implementation.

  \author Richard Brown

  Helper class for NiftyPET's GPU functionality. Wrapped
  functionality includes projection, unlisting and
  estimtaion of randoms and norms.

  \todo NiftyPET limitations - currently limited
  to the Siemens mMR scanner and requires to CUDA.

  \todo STIR wrapper limitations - currently only
  projects all of the data (no subsets). NiftyPET
  currently supports spans 0, 1 and 11, but the STIR
  wrapper has only been tested for span-11.

  DOI - https://doi.org/10.1007/s12021-017-9352-y
*/
/*
    Copyright (C) 2019-2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/shared_ptr.h"
#include "stir/Scanner.h"

// Forward declarations
struct Cnst;
struct txLUTs;
struct axialLUT;

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class ProjData;
template <typename elemT> class Viewgram;
template <typename elemT> class VoxelsOnCartesianGrid;

/*!
  \ingroup projection
  \brief Helper class for the wrapped NiftyPET projectors.
*/
class NiftyPETHelper
{
public:

    /// Default constructor
    NiftyPETHelper() :
        _already_set_up(false), _span(-1), _devid(0), _att(-1), _scanner_type(Scanner::Unknown_scanner)
    {}

    /// Destructor
    virtual ~NiftyPETHelper();

    /// Set CUDA device ID
    void set_cuda_device_id(const int devid)                 { _devid = char(devid);          }

    /// Set span
    void set_span(const char span)                            { _span = span;                 }

    /// Set emission (0) or transmission (1) - whether to exp{-result} for attenuation maps
    void set_att(const char att)                              { _att = att;                   }

    /// Set verbosity level for CUDA output
    void set_verbose(const bool verbose)                      { _verbose = verbose;           }

    /// Set scanner type
    void set_scanner_type(const Scanner::Type scanner_type)   { _scanner_type = scanner_type; }

    /// Set up
    void set_up();

    /// Create NiftyPET image
    static std::vector<float> create_niftyPET_image();

    /// Create STIR image with mMR dimensions
    static shared_ptr<VoxelsOnCartesianGrid<float> > create_stir_im();

    /// Create NiftyPET singram with no gaps. Forward project into this
    std::vector<float> create_niftyPET_sinogram_no_gaps() const;

    /// Create NiftyPET sinogram with gaps. Use this before converting to stir.
    std::vector<float> create_niftyPET_sinogram_with_gaps() const;

    /// Convert STIR image to NiftyPET image
    static void convert_image_stir_to_niftyPET(std::vector<float> &np, const DiscretisedDensity<3,float> &stir);

    /// Convert NiftyPET image to STIR image
    static void convert_image_niftyPET_to_stir(DiscretisedDensity<3,float> &stir, const std::vector<float> &np_vec);

    /// Convert STIR proj data to NiftyPET proj data
    void convert_proj_data_stir_to_niftyPET(std::vector<float> &np_vec, const ProjData& stir) const;

    /// Convert STIR viewgram to NiftyPET
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

    /// Create a STIR sinogram
    static shared_ptr<ProjData> create_stir_sino();

    /// Listmode to sinogram
    void lm_to_proj_data(shared_ptr<ProjData> &prompts_sptr, shared_ptr<ProjData> &delayeds_sptr,
                         shared_ptr<ProjData> &randoms_sptr, shared_ptr<ProjData> &norm_sptr,
                         const int tstart, const int tstop,
                         const std::string &lm_binary_file, const std::string &norm_binary_file="") const;

private:

    /// Check that set up has been run before returning data
    void check_set_up() const;

    /// Permute the data
    void permute(std::vector<float> &output_array, const std::vector<float> &orig_array, const unsigned output_dims[3], const unsigned *permute_order) const;

    /// Convert 3d NiftyPET proj data index to 1d
    unsigned convert_NiftyPET_proj_3d_to_1d_idx(const unsigned ang, const unsigned bins, const unsigned sino) const;

    /// Convert 1d NiftyPET proj data index to 3d
    void convert_NiftyPET_proj_1d_to_3d_idx(unsigned &ang, unsigned &bins, unsigned &sino, const unsigned idx) const;

    bool _already_set_up;
    char _span;
    char _devid;
    shared_ptr<Cnst> _cnt_sptr;
    int _nsinos;
    char _att;
    std::vector<int> _isub;
    bool _verbose;
    Scanner::Type _scanner_type;
    shared_ptr<txLUTs> _txlut_sptr;
    shared_ptr<axialLUT> _axlut_sptr;

    std::vector<float> _crs;
    std::vector<short> _s2c;

    // Get axLUT
    std::vector<float> _li2rng;
    std::vector<short> _li2sn;
    std::vector<char>  _li2nos;
};

END_NAMESPACE_STIR

#endif // __stir_gpu_NiftyPETHelper_h__
