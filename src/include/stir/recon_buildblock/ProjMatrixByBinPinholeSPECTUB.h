/*
    Copyright (C) 2022, Matthew Strugari
    Copyright (C) 2021, University College London
    This file is part of STIR.

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.

    See STIR/LICENSE.txt for details
*/
/*!
    \file
    \ingroup projection

    \brief stir::ProjMatrixByBinPinholeSPECTUB's definition 

    \author Matthew Strugari
    \author Carles Falcon
    \author Kris Thielemans
*/

#ifndef __stir_recon_buildblock_ProjMatrixByBinPinholeSPECTUB__
#define __stir_recon_buildblock_ProjMatrixByBinPinholeSPECTUB__

//system libraries
#include <iostream>

//user defined libraries
#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/ProjDataInfo.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange.h"
#include "stir/shared_ptr.h"
#include "stir/recon_buildblock/PinholeSPECTUB_Tools.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Bin;
/*!
  \ingroup projection
  \brief Generates projection matrix for pinhole SPECT studies

  \warning this class currently only works with VoxelsOnCartesianGrid. 

  \par Sample parameter file

\verbatim
        projector pair type := Matrix
            Projector Pair Using Matrix Parameters :=
            Matrix type := Pinhole SPECT UB
            Projection Matrix By Bin Pinhole SPECT UB Parameters:=
            
                ; Minimum weight to take into account. Makes reference just to the geometric (PSF) part of the weight. 
                ;  Weight could be lower after applying the attenuation factor (typically 0.005 - 0.02)
                minimum weight := 0.00001

                ;Maximum number of sigmas to consider in PSF calculation (typically 1.5 - 2.5)
                maximum number of sigmas := 2.0

                ; Spatial high resolution in which to sample distributions (typically 0.001 - 0.0001)
                spatial resolution PSF := 0.001

                ; Subsampling factor to compute convolutions for mid resolution. This reduces temporally the PSF resolution to
                ; perform more accurate calculus and then down sample the final PSF to the bin size (typically 1 - 8)
                subsampling factor PSF := 1

                ;  Detector and collimator parameter files
                detector file := detector.txt
                collimator file := collimator.txt

                ;Correction for intrinsic PSF { Yes // No }
                psf correction := no

                ; Correction for depth of impact { Yes // No }
                doi correction : no

                ; Attenuation correction { Simple // Full // No }
                attenuation type := no  
                ; Values in attenuation map in cm-1
                attenuation map := 

                ; Voxels not belonging to the cylinder defined by this radius are masked by default.
                object radius (cm) := 1.5
                ; Mask properties { Attenuation Map // Explicit Mask // No }. Default mask - cylinder object radius.
                mask type := No 
                ; In case of explicit mask.
                mask file := 

                keep all views in cache := 0

            End Projection Matrix By Bin Pinhole SPECT UB Parameters:=

        End Projector Pair Using Matrix Parameters :=
\endverbatim
*/

class ProjMatrixByBinPinholeSPECTUB : 
    public RegisteredParsingObject<
        ProjMatrixByBinPinholeSPECTUB,
        ProjMatrixByBin,
        ProjMatrixByBin
        >
{
  public :
    //! Name which will be used when parsing a ProjMatrixByBin object
    static const char * const registered_name; 

    //! Default constructor (calls set_defaults())
    ProjMatrixByBinPinholeSPECTUB();

    //! Destructor (deallocates UB SPECT memory)
    ~ProjMatrixByBinPinholeSPECTUB();

    //! Checks all necessary geometric info
    virtual void set_up(		 
        const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
        const shared_ptr<const DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
        );

    //! Enable keeping the matrix in memory
    /*!
        This speeds-up the calculations, but can use a lot of memory.
    */
    bool get_keep_all_views_in_cache() const;
    void set_keep_all_views_in_cache(bool value = true);
    std::string get_attenuation_type() const;


    //! Set type of attenuation modelling
    /*!
        Has to be "no", "simple" or "full".
    */
    void set_attenuation_type(const std::string& value);
    shared_ptr<const DiscretisedDensity<3,float> >
        get_attenuation_image_sptr() const;
  

    //! Set attenuation image
    /*!
        The image has to have same characteristics as the emission image currently.
        Will call set_attenuation_type() to set to "simple" if it was set to "no".
    */
    void
        set_attenuation_image_sptr(const shared_ptr<const DiscretisedDensity<3,float> > value);
    void
        set_attenuation_image_sptr(const std::string& value);


  private:

    // parameters that will be parsed

    float object_radius;
    float minimum_weight;
    float maximum_number_of_sigmas;
    float spatial_resolution_PSF;
    int subsampling_factor_PSF;
    std::string detector_file;
    std::string collimator_file;
    std::string psf_correction;
    std::string doi_correction;
    std::string attenuation_type;
    std::string attenuation_map;
    std::string mask_type;
    std::string mask_file;
    bool keep_all_views_in_cache; //!< if set to false, only a single view is kept in memory

    // explicitly list necessary members for image details (should use an Info object instead)
    CartesianCoordinate3D<float> voxel_size;
    CartesianCoordinate3D<float> origin;  
    IndexRange<3> densel_range;

    shared_ptr<const ProjDataInfo> proj_data_info_ptr;

    bool already_setup;

    virtual void 
        calculate_proj_matrix_elems_for_one_bin(ProjMatrixElemsForOneBin&) const;

    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();

    shared_ptr<const DiscretisedDensity<3,float> > attenuation_image_sptr;

    // wm_SPECT starts here ---------------------------------------------------------------------------------------------
    bool  *msk_3d;        // voxels to be included in matrix (no weight calculated outside the mask)
    float *attmap;        // attenuation map

    //... variables for estimated sizes of arrays to allocate ................................
    int **Nitems;         //!< number of non-zero elements for each weight matrix row

    //... user defined structures (types defined in PinholeSPECTUB_Tools.h) .....................................

    SPECTUB_mph::volume_type vol;       //!< structure with volume (image) information
    SPECTUB_mph::prj_mph_type prj;      //!< structure with projection information
    SPECTUB_mph::bin_type bin;          //!< structure with bin information

    // mutable to allow compute_one_subset() const function to change the fields
    mutable SPECTUB_mph::psf2d_type psf_bin;   // structure for total psf distribution in bins (bidimensional)
    mutable SPECTUB_mph::psf2d_type psf_subs;  // structure for total psf distribution: mid resolution (bidimensional)
    mutable SPECTUB_mph::psf2d_type psf_aux;   // structure for total psf distribution: mid resolution auxiliar for convolution (bidimensional)
    mutable SPECTUB_mph::psf2d_type kern;      // structure for intrinsic psf distribution: mid resolution (bidimensional)

    void compute_one_subset(const int kOS) const;
    void delete_PinholeSPECTUB_arrays();
};

END_NAMESPACE_STIR

#endif
