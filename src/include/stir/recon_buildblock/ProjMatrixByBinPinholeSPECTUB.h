/*
    Copyright (C) 2022, Matthew Strugari
    Copyright (C) 2021, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
    Projection Matrix By Bin Pinhole SPECT UB Parameters:=

        maximum number of sigmas := 2.0
        spatial resolution PSF := 0.001
        subsampling factor PSF := 1

        detector file := detector.txt
        collimator file := collimator.txt

        ; PSF and DOI correction { Yes // No }
        psf correction := no
        doi correction := no

        ; Attenuation correction { Simple // Full // No }
        attenuation type := no
        attenuation map :=

        object radius (cm) := 2.3
        mask file := 
        ; If no mask file is set, we can either compute it from attenuation map or object radius
        mask from attenuation map := 0

        keep all views in cache := 0

    End Projection Matrix By Bin Pinhole SPECT UB Parameters:=
\endverbatim

  \par Sample detector file
  
\verbatim
    Information of detector
    Comments are allowed here or anywhere in lines not containing parameters.
    Parameters are defined with a delimiting colon. Avoid using a colon elsewehere.
    # Sigma = FWHM/(2*sqrt(2*ln(2))) where FWHM = 0.85 mm
    # CsI at 140.5 keV from NIST
    Number of rings: 1
    #intrinsic PSF#
    Sigma (cm): 0.0361
    Crystal thickness (cm): 0.3
    Crystal attenuation coefficient (cm -1): 4.407   
    \#……repeat for each ring …………\#
    Nangles: 4
    ang0 (deg): 180.
    incr (deg): 90.0
    z0 (cm): 0.
    \#…………until here……………\#
\endverbatim

  \par Sample collimator file
\verbatim
    Information of collimator
    Comments are allowed here or anywhere in lines not containing parameters.
    Parameters are defined with a delimiting colon. Avoid using a colon elsewehere.
    Model (cyl/pol): pol
    Collimator radius (cm): 2.8
    Wall thickness (cm): 1.
    #holes#
    Number of holes: 4
    nh / ind / x(cm) / y(cm) / z(cm) / shape(rect-round) / sizex(cm) / sizez(cm)
    / angx(deg) / angz(deg) / accx(deg) / accz(deg)
    h1:	1	0.	0.	0.	round	0.1	0.1	0.	0.	45.	45.
    h2:	2	0.	0.	0.	round	0.1	0.1	0.	0.	45.	45.
    h3:	3	0.	0.	0.	round	0.1	0.1	0.	0.	45.	45.
    h4:	4	0.	0.	0.	round	0.1	0.1	0.	0.	45.	45.
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

    // disable copy-constructor as currently unsafe to copy due to bare pointers
    ProjMatrixByBinPinholeSPECTUB(const ProjMatrixByBinPinholeSPECTUB&) = delete;

    //! Destructor (deallocates UB SPECT memory)
    ~ProjMatrixByBinPinholeSPECTUB();

    //! Checks all necessary geometric info
    virtual void set_up(		 
        const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
        const shared_ptr<const DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
        );

    //******************** get/set functions *************

    //! Minimum weight
    // no longer set minimum weight, always use 0.
    // float get_minimum_weight() const;
    // void set_minimum_weight( const float value );

    //! Maximum number of sigmas
    float get_maximum_number_of_sigmas() const;
    void set_maximum_number_of_sigmas( const float value );
        
    //! Spatial resolution PSF
    float get_spatial_resolution_PSF() const;
    void set_spatial_resolution_PSF( const float value );

    //! Subsampling factor PSF
    int get_subsampling_factor_PSF() const;
    void set_subsampling_factor_PSF( const int value );
    
    //! Detector file
    //std::string get_detector_file() const;
    void set_detector_file( const std::string& value );

    //! Collimator file
    //std::string get_collimator_file() const;
    void set_collimator_file( const std::string& value );
                
    //! PSF correction
    std::string get_psf_correction() const;
    void set_psf_correction( const std::string& value );

    //! Set DOI correction
    std::string get_doi_correction() const;
    void set_doi_correction( const std::string& value );

    //! Object radius (cm)
    float get_object_radius() const;
    void set_object_radius( const float value );
    
    //! Type of attenuation modelling    
    std::string get_attenuation_type() const;
    void set_attenuation_type(const std::string& value);
    shared_ptr<const DiscretisedDensity<3,float> >
        get_attenuation_image_sptr() const;
  
    //! Attenuation image
    void set_attenuation_image_sptr(const shared_ptr<const DiscretisedDensity<3,float> > value);
    void set_attenuation_image_sptr(const std::string& value);

    //! Type of masking
    // no longer use mask type
    //std::string get_mask_type() const;
    //void set_mask_type(const std::string& value);
        
    //! Mask from mask file
    shared_ptr<const DiscretisedDensity<3,float> >
        get_mask_image_sptr() const;
    void set_mask_image_sptr(const shared_ptr<const DiscretisedDensity<3,float> > value);
    void set_mask_image_sptr(const std::string& value);
    
    //! Mask from attenuation map
    bool get_mask_from_attenuation_map() const;
    void set_mask_from_attenuation_map(bool value = false);
  
    //! Enable keeping the matrix in memory
    bool get_keep_all_views_in_cache() const;
    void set_keep_all_views_in_cache(bool value = false);

    ProjMatrixByBinPinholeSPECTUB * clone() const;

  private:

    // parameters that will be parsed

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
    //std::string mask_type;
    float object_radius;
    std::string mask_file;
    bool mask_from_attenuation_map;
    bool keep_all_views_in_cache; //!< if set to false, only a single view is kept in memory

    // explicitly list necessary members for image details (should use an Info object instead)
    CartesianCoordinate3D<float> voxel_size;
    CartesianCoordinate3D<float> origin;  
    IndexRange<3> densel_range;

    shared_ptr<const ProjDataInfo> proj_data_info_ptr;

    bool already_setup;

    mutable SPECTUB_mph::wmh_mph_type wmh;       // weight matrix header.
    mutable SPECTUB_mph::wm_da_type wm;          // double array weight matrix structure.
    mutable SPECTUB_mph::pcf_type pcf;           // pre-calculated functions

    virtual void 
        calculate_proj_matrix_elems_for_one_bin(ProjMatrixElemsForOneBin&) const;

    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();

    shared_ptr<const DiscretisedDensity<3,float> > attenuation_image_sptr;
    shared_ptr<const DiscretisedDensity<3,float> > mask_image_sptr;

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
