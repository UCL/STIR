#ifndef __stir_scatter_ScatterSimulation_H__
#define __stir_scatter_ScatterSimulation_H__

/*!
  \file
  \ingroup scatter
  \brief Definition of class stir::ScatterSimulation

  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#include "stir/ParsingObject.h"
#include "stir/Succeeded.h"
#include "stir/RegisteredObject.h"

#include "stir/ProjData.h"
#include "stir/VoxelsOnCartesianGrid.h"

START_NAMESPACE_STIR

class Succeeded;
class ProjDataInfoCylindricalNoArcCorr;
class ViewSegmentNumbers;
class BinNormalisation;

/*!
  \ingroup scatter
  \brief Simuate the scatter probability using a model-based approach

  N.E. : This class is roughly the base class of what used to be the ScatterEstimationByBin.
  Because there are different approaches on the actual simulation process, this base class will
  be in charge of hold projection data and subsample the attenuation image, while more core function
  will deligate to classes like SignelScatterSimulation.

  This class computes the single Compton scatter estimate for PET data using an analytical
  approximation of an integral. It takes as input an emission image and an attenuation image.
  This is effectively an implementation of the simulation part of the algorithms of
  Watson and Ollinger.

  One non-standard feature is that you can specify a different attenuation image to find the
  scatter points and one to compute the integrals over the attenuation image. The idea is that
  maybe you want to compute the integrals on a finer grid than you sample the attenuation image.
  This is probably not very useful though.

  \todo Currently this can only be run by initialising it via parsing of a file. We need
  to add a lot of set/get members.

  \todo This class currently uses a simple Gaussian model for the energy resolution. This model
  and its parameters (\a reference_energy and \a energy_resolution) really should be moved the
  the Scanner class. Also the \a lower_energy_threshold and \a upper_energy_threshold should
  be read from the emission data, as opposed to setting them here.

  \todo detector coordinates are derived from ProjDataInfo, but areas and orientations are
  determined by using a cylindrical scanner.

  \todo This class should be split into a generic class and one specific to PET single scatter.

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
  </ol>
*/

class ScatterSimulation : public RegisteredObject<ScatterSimulation>,
        public ParsingObject
{

public:

    //! \details Default constructor
    ScatterSimulation();

    virtual ~ScatterSimulation();

    virtual Succeeded
    process_data();

    //! gives method information
    virtual std::string method_info() const = 0;
    //! prompts the user to enter parameter values manually
    virtual void ask_parameters();

    //! \details Image which will hold the activity estimates
    shared_ptr<DiscretisedDensity<3,float> > activity_image_sptr;
    //! \details Filename for the initial activity estimate.
    std::string activity_image_filename;

    //!
    //! \brief set_exam_info_sptr
    //! \details Since July 2016, the information for the energy window and energy
    //! resolution come from the ExamInfo.
    inline void
    set_exam_info_sptr(const shared_ptr<ExamInfo>&);


    //! find scatter points
    /*! This function sets scatt_points_vector and scatter_volume. It will also
        remove any cached integrals as they would be incorrect otherwise.
    */
    void sample_scatter_points();

    /**
     *
     * \name functions to set parameters
     * @{
     */

    inline void
    set_output_proj_data(const std::string&);

    inline void
    set_output_proj_data_sptr(const shared_ptr<ExamInfo>&,
                              const shared_ptr<ProjDataInfo>&,
                              const std::string &);

    inline void
    get_output_proj_data(shared_ptr<ProjData>&);

    //! Load the projection template and perform basic checks.
    inline void set_template_proj_data_info_sptr(const shared_ptr<ProjDataInfo>&);
    //! Load the projection template from file
    inline void set_template_proj_data_info(const std::string&);
    //! Set the activity image stpr
    inline void set_activity_image_sptr(const shared_ptr<DiscretisedDensity<3,float> >&);
    //! Set activity image from file name
    inline void set_activity_image(const std::string& filename);

    //! create output projection data of same size as template_proj_data_info
    /*! \warning use set_template_proj_data_info() first.

     Currently always uses Interfile output.
     \warning If the specified file already exists it will be erased.
    */
    inline void set_proj_data_from_file(const std::string& filename,
                                        shared_ptr<ProjData>& _this_projdata);

    //!
    //! \brief set_density_image_sptr

    inline void
    set_density_image_sptr(const shared_ptr<DiscretisedDensity<3,float> >&);

    //!
    //! \brief set_density_image_and_subsampled
    //! \author Nikos Efthimiou
    //! \warning This function has to called only if the density_image_for_scatter_points
    //! has not been set in any other way. Because it is going to erase then and
    //! recalculate them.
    void
    set_density_image_and_subsample(const shared_ptr<DiscretisedDensity<3,float> >&);

    //!
    //! \brief set_projdata_and_subsample
    //!
    void
    set_projdata_and_subsample(const shared_ptr<ExamInfo> &,
                               const shared_ptr<ProjDataInfo > &);

    inline void
    set_density_image(const std::string&);

    //!
    //! \brief set_density_image_for_scatter_points_sptr
    //!
    inline void
    set_density_image_for_scatter_points_sptr(const shared_ptr<DiscretisedDensity<3,float> >&);

    //!
    //! \brief set_density_image_for_scatter_points
    //! \param filename
    //!
    inline void
    set_density_image_for_scatter_points(const std::string&);

    inline void set_attenuation_threshold(const float&);

    inline void set_random_point(const bool&);

    inline void set_cache_enabled(const bool&);

    /** @}*/

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

protected:

    //! computes scatter for one viewgram
    /*! \return total scatter estimated for this viewgram */
    virtual double
    process_data_for_view_segment_num(const ViewSegmentNumbers& vs_num);

    float
    compute_emis_to_det_points_solid_angle_factor(const CartesianCoordinate3D<float>& emis_point,
                                                  const CartesianCoordinate3D<float>& detector_coord) ;

    //! threshold below which a voxel in the attenuation image will not be considered as a candidate scatter point
    float attenuation_threshold;

    //! boolean to see if we need to move the scatter point randomly within in its voxel
    /*! This was first recommended by Watson. It is recommended to leave this on, as otherwise
       discretisation errors are more obvious.

       Note that the random generator is seeded via date/time, so re-running the scatter
       simulation will give a slightly different result if this boolean is on.
    */
    bool random;
    //! boolean to see if we need to cache the integrals
    /*! By default, we cache the integrals over the emission and attenuation image. If you run out
        of memory, you can switch this off, but performance will suffer dramatically.
    */
    bool use_cache;

    virtual void set_defaults();
    virtual void initialise_keymap();

    //!
    //! \brief post_processing
    //! \return
    //! \details used to check acceptable parameter ranges, etc...
    //! It is going to take handle loading directly the proper files
    //! (subsampled attenuation image and scanner). More complex options
    //! will be handled at set_up().
    virtual bool post_processing();

    enum image_type{act_image_type, att_image_type};
    struct ScatterPoint
    {
        CartesianCoordinate3D<float> coord;
        float mu_value;
    };

    std::vector< ScatterPoint> scatt_points_vector;

    float scatter_volume;


    //! remove cached attenuation integrals
    /*! should be used before recalculating scatter for a new attenuation image or
      when changing the sampling of the detector etc */
    virtual void remove_cache_for_integrals_over_attenuation();

    //! reset cached activity integrals
    /*! should be used before recalculating scatter for a new activity image or
      when changing the sampling of the detector etc */
    virtual void remove_cache_for_integrals_over_activity();

    /**
      *
      * \name Variables retated to the scanner geometry
      * @{
      */

    //!
    //! \brief template_proj_data_filename
    //! \details The file name for the Scanner template
    std::string scatter_proj_data_filename;

    //!
    //! \brief proj_data_info_ptr
    //! \details The projection data info of the scanner template
    ProjDataInfoCylindricalNoArcCorr * proj_data_info_ptr;

    shared_ptr<ProjDataInfo> proj_data_info_sptr;

    //!
    //! \brief template_exam_info_sptr
    //! \details Exam info extracted from the scanner template
    shared_ptr<ExamInfo> template_exam_info_sptr;

    /** @}*/

    /**
      *
      * \name Variables related to the subsampled attenuation image.
      * @{
    */

    //!
    //! \brief sub_atten_image_filename
    //! \details Input or output file name of the subsampled
    //! attenuation image, depends on the reconmpute_sub_atten_image
    std::string density_image_filename;

    //!
    //! \brief density_image_for_scatter_points_filename
    //!
    std::string density_image_for_scatter_points_filename;

    shared_ptr< DiscretisedDensity<3, float> > density_image_sptr;

    //!
    //! \brief atten_image_filename∫
    //! \details File name of the original attenuation image.
    std::string orig_atten_image_filename;

    //!
    //! \brief sub_atten_image_sptr
    //! \details This is the image with the anatomical information
    //! \warning Previously density_image_for_scatter_points_sptr
    shared_ptr< DiscretisedDensity<3, float> > density_image_for_scatter_points_sptr;

    /** }@*/

    int total_detectors;

    /** \name detection related functions
     *
     * @{
     */

    //!
    //! \brief detection_efficiency
    //! \param energy
    //! \return
    //! \details energy-dependent detection efficiency (Gaussian model)
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

    /**
     *
     * \name Variables related to the subsampled ProjData.
     * @{
     */

    //!
    //! \brief sub_proj_data_info_ptr
    //!
    shared_ptr<ProjDataInfo> template_proj_data_info_sptr;
    /** }@*/

    virtual double
    scatter_estimate(const unsigned det_num_A,
                     const unsigned det_num_B);

    virtual void
    actual_scatter_estimate(double& scatter_ratio_singles,
                            const unsigned det_num_A,
                            const unsigned det_num_B) = 0;



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

    Array<2,float> cached_activity_integral_scattpoint_det;
    Array<2,float> cached_attenuation_integral_scattpoint_det;

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

    //!
    //! \brief output_proj_data_filename
    //!
    std::string output_proj_data_filename;

    //!
    //! \brief output_proj_data_sptr
    //!
    shared_ptr<ProjData> output_proj_data_sptr;

    int times;

};

END_NAMESPACE_STIR

#include "stir/scatter/ScatterSimulation.inl"

#endif


