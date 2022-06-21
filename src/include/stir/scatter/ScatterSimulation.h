/*
    Copyright (C) 2018 - 2019 University of Hull
    Copyright (C) 2004 - 2009 Hammersmith Imanet Ltd
    Copyright (C) 2013 - 2016, 2019, 2020, 2022 University College London
    Copyright (C) 2022, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_scatter_ScatterSimulation_H__
#define __stir_scatter_ScatterSimulation_H__

/*!
  \file
  \ingroup scatter
  \brief Definition of class stir::ScatterSimulation

  \author Charalampos Tsoumpas
  \author Nikolaos Dikaios
  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#include "stir/ParsingObject.h"
#include "stir/RegisteredObject.h"
#include "stir/ProjData.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"

START_NAMESPACE_STIR

/*!
  \ingroup scatter
  \brief Simulate the scatter probability using a model-based approach

  This base class intends to computes a Compton scatter estimate using an analytical
  approximation of an integral. It takes as input an emission image and an attenuation image.
  This is effectively an implementation of the simulation part of the algorithms of
  Watson and Ollinger. The base class takes care of downsampling and upsampling, book-keeping,
  caching and looping over bins.

  One non-standard feature is that you can specify a different attenuation image to find the
  scatter points and one to compute the integrals over the attenuation image. The idea is that
  maybe you want to compute the integrals on a finer grid then you sample the attenuation image.
  This is probably not very useful though.

  \todo detector coordinates are derived from ProjDataInfo, but areas and orientations are
  determined by using a cylindrical scanner.

  \todo variables/function named \c density really should use \c attenuation. This is currently only
  done for a few variables, but parsing keywords are correct.

  \par References
  This implementation is described in the following
  <ol>
  <li> C. Tsoumpas, P. Aguiar, K. S. Nikita, D. Ros, K. Thielemans,
  <i>Evaluation of the Single Scatter Simulation Algorithm Implemented in the STIR Library</i>,
  Proc. of IEEE Medical Imaging Conf. 2004, vol. 6 pp. 3361 - 3365.
  </li>
  </ol>
  Please refer to the above if you use this implementation.

  See also these papers for extra evaluation

  <ol>
  <li>N. Dikaios , T. J. Spinks , K. Nikita , K. Thielemans,
     <i>Evaluation of the Scatter Simulation Algorithm Implemented in STIR,</i>
     proc. 5th ESBME, Patras, Greece.
  </li>
  <li>P. Aguiar, Ch. Tsoumpas, C. Crespo, J. Pavia, C. Falcon, A. Cot, K. Thielemans and D. Ros,
     <i>Assessment of scattered photons in the quantification of the small animal PET studies,</i>
     Eur J Nucl Med Mol I 33:S315-S315 Sep 2006, Proc. EANM 2006, Athens, Greece.
  </li>
  <li>I Polycarpou, P K Marsden and C Tsoumpas,
      <i>A comparative investigation of scatter correction in 3D PET</i>,
      J Phys: Conference Series (317), conference 1, 2011
  </li>
  </ol>
*/

class ScatterSimulation : public RegisteredObject<ScatterSimulation>
{
public:

    //! Default constructor
    ScatterSimulation();

    virtual ~ScatterSimulation();

    virtual Succeeded process_data();
    //! gives method information
    virtual std::string method_info() const = 0;
    //! prompts the user to enter parameter values manually
    virtual void ask_parameters();
    //! \name check functions
    //@{
    inline bool has_template_proj_data_info() const
    { return !stir::is_null_ptr(proj_data_info_sptr); }
    //! Returns true if template_exam_info_sptr has been set.
    inline bool has_exam_info() const
    { return !stir::is_null_ptr(template_exam_info_sptr);}
    //@}

    //! \name get functions
    //@{
    shared_ptr<ProjData>
    get_output_proj_data_sptr() const;

    inline int get_num_scatter_points() const
    { return static_cast<int>(this->scatt_points_vector.size());}
    //! Get the template ProjDataInfo
    shared_ptr<const ProjDataInfo> get_template_proj_data_info_sptr() const;
    //! Get the ExamInfo
    shared_ptr<const ExamInfo> get_exam_info_sptr() const;

    const DiscretisedDensity<3,float>& get_activity_image() const;
    const DiscretisedDensity<3,float>& get_attenuation_image() const;
    const DiscretisedDensity<3,float>& get_attenuation_image_for_scatter_points() const;
    //! \deprecated
    shared_ptr<const DiscretisedDensity<3,float> > get_density_image_for_scatter_points_sptr() const;
    //@}

    //! \name set functions
    //@{

    void set_template_proj_data_info(const std::string&);

    void set_template_proj_data_info(const ProjDataInfo&);

    void set_activity_image_sptr(const shared_ptr<const DiscretisedDensity<3,float> >);

    void set_activity_image(const std::string& filename);
    //! \details Since July 2016, the information for the energy window and energy
    //! resolution are stored in ExamInfo.
    void set_exam_info(const ExamInfo&);
    void set_exam_info_sptr(const shared_ptr<const ExamInfo>);

    void set_output_proj_data_sptr(shared_ptr<ProjData>);

    void set_density_image_sptr(const shared_ptr<const DiscretisedDensity<3,float> >);

    void set_density_image(const std::string&);
    //! This function depends on the ProjDataInfo of the scanner.
    //! You first have to set that.
    void set_output_proj_data(const std::string&);

    void
    set_output_proj_data_sptr(const shared_ptr<const ExamInfo>,
                              const shared_ptr<const ProjDataInfo>,
                              const std::string &);

    void set_density_image_for_scatter_points_sptr(shared_ptr<const DiscretisedDensity<3,float> >);

    void set_image_downsample_factors(float factor_xy = 1.f, float factor_z = 1.f,
                                      int _size_zoom_xy = -1, int _size_zoom_z = -1);
        //! set_density_image_for_scatter_points
    void set_density_image_for_scatter_points(const std::string&);
    //! set the attenuation threshold
    void set_attenuation_threshold(const float);
    //! The scattering point in the voxel will be chosen randomly, instead of choosing the centre.
    /*! This was first recommended by Watson. It is recommended to leave this on, as otherwise
       discretisation errors are more obvious.

       Note that the random generator is seeded via date/time, so re-running the scatter
       simulation will give a slightly different result if this boolean is on.
    */
    void set_randomly_place_scatter_points(const bool);

    void set_cache_enabled(const bool);

    //@}

    //! This function is a less powerfull tool than directly zooming the image.
    //! However it will check that the downsampling is done in manner compatible with the
    //! ScatterSimulation.
    void downsample_density_image_for_scatter_points(float _zoom_xy, float _zoom_z,
                          int _size_xy = -1, int _size_z = -1);

    //! Downsample the scanner keeping the total axial length the same.
    /*! If \c new_num_rings<=0, use rings of approximately 2 cm thickness.
        If \c new_num_dets <=0, use the default set (currently set in set_defaults())
    */
    Succeeded downsample_scanner(int new_num_rings = -1, int new_num_dets = -1);
    //! Downsamples activity and attenuation images to voxel sizes appropriate for the (downsampled) scanner.
    /*! This step is not necessary but could result in a speed-up in computing the line integrals.
        It also avoids problems with too high resolution images compared to the downsampled scanner.

	Another way to resolve that is to smooth the images before the scatter simulation.
	This is currently not implemented in this class.
	\warning This function should be called after having set all data.
    */
    Succeeded downsample_images_to_scanner_size();

    //! \name Compton scatter cross sections
    //@{
    static
    inline float
    dif_Compton_cross_section(const float cos_theta, float energy);

    static
    inline float
    total_Compton_cross_section(float energy);

    static
    inline float
    photon_energy_after_Compton_scatter(const float cos_theta, const float energy);

    static
    inline float
    photon_energy_after_Compton_scatter_511keV(const float cos_theta);

    static
    inline float
    total_Compton_cross_section_relative_to_511keV(const float energy);
    //@}

    virtual Succeeded set_up();

    //! Output the log of the process.
    virtual void write_log(const double simulation_time, const float total_scatter);

    //! Enable/disable caching of line integrals
    void set_use_cache(const bool);
    //! Return if line integrals are cached or not
    bool get_use_cache() const;

protected:

    //! computes scatter for one viewgram
    /*! \return total scatter estimated for this viewgram */
    virtual double
    process_data_for_view_segment_num(const ViewSegmentNumbers& vs_num);

    float
    compute_emis_to_det_points_solid_angle_factor(const CartesianCoordinate3D<float>& emis_point,
                                                  const CartesianCoordinate3D<float>& detector_coord);



    virtual void set_defaults();
    virtual void initialise_keymap();
    //! \warning post_processing will set everything that has a file name in
    //! the par file. The corresponding set functions should be used either
    //! for files that are not stored in the drive.
    virtual bool post_processing();

    enum image_type{act_image_type, att_image_type};
    struct ScatterPoint
    {
        CartesianCoordinate3D<float> coord;
        float mu_value;
    };

    std::vector< ScatterPoint> scatt_points_vector;

    float scatter_volume;

    //! find scatter points
    /*! This function sets scatt_points_vector and scatter_volume. It will also
        remove any cached integrals as they would be incorrect otherwise.
    */
    void
    sample_scatter_points();

    //! remove cached attenuation integrals
    /*! should be used before recalculating scatter for a new attenuation image or
      when changing the sampling of the detector etc */
    virtual void remove_cache_for_integrals_over_attenuation();

    //! reset cached activity integrals
    /*! should be used before recalculating scatter for a new activity image or
      when changing the sampling of the detector etc */
    virtual void remove_cache_for_integrals_over_activity();

    /** \name detection related functions
     *
     * @{
     */

    float detection_efficiency(const float energy) const;


    //! maximum angle to consider above which detection after Compton scatter is considered too small
    static
    float
    max_cos_angle(const float low, const float approx, const float resolution_at_511keV);

    //! mimumum energy to consider above which detection after Compton scatter is considered too small
    static
    float
    energy_lower_limit(const float low, const float approx, const float resolution_at_511keV);

    virtual
    void
    find_detectors(unsigned& det_num_A, unsigned& det_num_B, const Bin& bin) const;

    unsigned
    find_in_detection_points_vector(const CartesianCoordinate3D<float>& coord) const;

    CartesianCoordinate3D<float>  shift_detector_coordinates_to_origin;

    //! average detection efficiency of unscattered counts
    double
    detection_efficiency_no_scatter(const unsigned det_num_A,
                                    const unsigned det_num_B) const;

    // next needs to be mutable because find_in_detection_points_vector is const
    mutable std::vector<CartesianCoordinate3D<float> > detection_points_vector;

    //!@}

    //! virtual function that computes the scatter for one (downsampled) bin
    virtual double
      scatter_estimate(const Bin& bin) = 0;

    //! \name integrating functions
    //@{
    static
    float
    integral_between_2_points(const DiscretisedDensity<3,float>& density,
                              const CartesianCoordinate3D<float>& point1,
                              const CartesianCoordinate3D<float>& point2);

    float
    exp_integral_over_attenuation_image_between_scattpoint_det (const CartesianCoordinate3D<float>& scatter_point,
                                                                const CartesianCoordinate3D<float>& detector_coord);



    float
    integral_over_activity_image_between_scattpoint_det (const CartesianCoordinate3D<float>& scatter_point,
                                                         const CartesianCoordinate3D<float>& detector_coord);



    float
    cached_integral_over_activity_image_between_scattpoint_det(const unsigned scatter_point_num,
                                                               const unsigned det_num);

    float
    cached_exp_integral_over_attenuation_image_between_scattpoint_det(const unsigned scatter_point_num,
                                                                      const unsigned det_num);
    //@}

    std::string template_proj_data_filename;

    shared_ptr<ProjDataInfo> proj_data_info_sptr;
    //! \details Exam info extracted from the scanner template
    shared_ptr<ExamInfo> template_exam_info_sptr;

    std::string density_image_filename;

    std::string density_image_for_scatter_points_filename;

    std::string density_image_for_scatter_points_output_filename;

    shared_ptr< const DiscretisedDensity<3, float> > density_image_sptr;

    //! Pointer to hold the current activity estimation
    shared_ptr<const DiscretisedDensity<3,float> > activity_image_sptr;

    //! set-up cache for attenuation integrals
    /*! \warning This will not remove existing cached data (if the sizes match). If you need this,
        call remove_cache_for_scattpoint_det_integrals_over_attenuation() first.
    */
    void initialise_cache_for_scattpoint_det_integrals_over_attenuation();
    //! set-up cache for activity integrals
    /*! \warning This will not remove existing cached data (if the sizes match). If you need this,
        call remove_cache_for_scattpoint_det_integrals_over_activity() first.
    */
    void initialise_cache_for_scattpoint_det_integrals_over_activity();

    //! Output proj_data fileanme prefix
    std::string output_proj_data_filename;
    //! Shared ptr to hold the simulated data.
    shared_ptr<ProjData> output_proj_data_sptr;

    //! threshold below which a voxel in the attenuation image will not be considered as a candidate scatter point
    float attenuation_threshold;

    //! boolean to see if we need to move the scatter point randomly within in its voxel
    bool randomly_place_scatter_points;
    //! boolean to see if we need to cache the integrals
    /*! By default, we cache the integrals over the emission and attenuation image. If you run out
        of memory, you can switch this off, but performance will suffer dramatically.
    */
    bool use_cache;
    //! Filename for the initial activity estimate.
    std::string activity_image_filename;
    //! Zoom factor on plane XY. Defaults on 1.f.
    float zoom_xy;
    //! Zoom factor on Z axis. Defaults on 1.f.
    float zoom_z;
    //! Zoomed image size on plane XY. Defaults on -1.
    int zoom_size_xy;
    //! Zoomed image size on Z axis. Defaults on -1.
    int zoom_size_z;
    //! Number of rings of downsampled scanner
    int downsample_scanner_rings;
    //! Number of detectors per ring of downsampled scanner
    int downsample_scanner_dets;

    bool downsample_scanner_bool;

 private:
    int total_detectors;

    Array<2,float> cached_activity_integral_scattpoint_det;
    Array<2,float> cached_attenuation_integral_scattpoint_det;
    shared_ptr< DiscretisedDensity<3, float> > density_image_for_scatter_points_sptr;

    // numbers that we don't want to recompute all the time
    mutable float detector_efficiency_no_scatter;

    bool _already_set_up;

    //! a function that checks if image sizes are ok
    /*! It will call \c error() if not.

      Currently, STIR shifts the middle of the image to the middle of the scanner. This
      is dangerous when using image zooming.
      This function currently checks if \a _image is consistent with the \c activity_image_sptr.

      See https://github.com/UCL/STIR/issues/495 for more information.
     */
    void check_z_to_middle_consistent(const DiscretisedDensity<3,float>& _image, const std::string& name) const;
};

END_NAMESPACE_STIR

#include "stir/scatter/ScatterSimulation.inl"

#endif


