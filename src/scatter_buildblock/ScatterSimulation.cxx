
#include "stir/scatter/ScatterSimulation.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/Bin.h"

#include "stir/CPUTimer.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/Viewgram.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/info.h"
#include "stir/error.h"
#include <fstream>
#include <boost/format.hpp>

#include "stir/stir_math.h"
#include "stir/zoom.h"
#include "stir/ArrayFunction.h"

#include "stir/ProjDataInMemory.h"
#include "stir/NumericInfo.h"

START_NAMESPACE_STIR

ScatterSimulation::
ScatterSimulation()
{
    this->set_defaults();
}

ScatterSimulation::
~ScatterSimulation()
{}

Succeeded
ScatterSimulation::
process_data()
{
    //    this->times +=1;
    //    // write after division.
    //    {
    //        std::stringstream nikos;
    //        nikos << "./activity_in_" << this->times ;
    //        std::string output1= nikos.str();
    //        OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    //                write_to_file(output1, *this->activity_image_sptr);
    //    }

    //    // write after division.
    //    {
    //        std::string output1= "./density_in_0";
    //        OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    //                write_to_file(output1, *this->density_image_sptr);
    //    }

    //    {
    //        std::string output1= "./density_scat_points_in_0";
    //        OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    //                write_to_file(output1, *this->density_image_for_scatter_points_sptr);
    //    }

    this->output_proj_data_sptr->fill(0.f);

    //    if (!this->recompute_sub_atten_image && !this->recompute_sub_projdata)
    //    {
    //        if (fabs(this->sub_vox_xy -
    //                 this->sub_proj_data_info_ptr->get_scanner_ptr()->get_ring_spacing()) > 1.E-2)
    //            error("DataSymmetriesForBins_PET_CartesianGrid can currently only support z-grid spacing "
    //                  "equal to the ring spacing of the scanner divided by an integer. Sorry\n");
    //    }

    // The activiy image has been changed, from another class.
    this->remove_cache_for_integrals_over_activity();

    this->initialise_cache_for_scattpoint_det_integrals_over_attenuation();
    this->initialise_cache_for_scattpoint_det_integrals_over_activity();
    ViewSegmentNumbers vs_num;
    /* ////////////////// SCATTER ESTIMATION TIME ////////////////
   */
    CPUTimer bin_timer;
    bin_timer.start();
    // variables to report (remaining) time
    HighResWallClockTimer wall_clock_timer;
    double previous_timer = 0 ;
    int previous_bin_count = 0 ;
    int bin_counter = 0;
    int axial_bins = 0 ;
    wall_clock_timer.start();

    for (vs_num.segment_num() = this->proj_data_info_ptr->get_min_segment_num();
         vs_num.segment_num() <= this->proj_data_info_ptr->get_max_segment_num();
         ++vs_num.segment_num())
        axial_bins += this->proj_data_info_ptr->get_num_axial_poss(vs_num.segment_num());

    const int total_bins =
            this->proj_data_info_ptr->get_num_views() * axial_bins *
            this->proj_data_info_ptr->get_num_tangential_poss();
    /* ////////////////// end SCATTER ESTIMATION TIME ////////////////
   */
    /* Currently, proj_data_info.find_cartesian_coordinates_of_detection() returns
     coordinate in a coordinate system where z=0 in the first ring of the scanner.
     We want to shift this to a coordinate system where z=0 in the middle
     of the scanner.
     We can use get_m() as that uses the 'middle of the scanner' system.
     (sorry)
  */
#ifndef NDEBUG
    {
        CartesianCoordinate3D<float> detector_coord_A, detector_coord_B;
        // check above statement
        this->proj_data_info_ptr->find_cartesian_coordinates_of_detection(
                    detector_coord_A, detector_coord_B, Bin(0, 0, 0, 0));
        assert(detector_coord_A.z() == 0);
        assert(detector_coord_B.z() == 0);
        // check that get_m refers to the middle of the scanner
        const float m_first =
                this->proj_data_info_ptr->get_m(Bin(0, 0, this->proj_data_info_ptr->get_min_axial_pos_num(0), 0));
        const float m_last =
                this->proj_data_info_ptr->get_m(Bin(0, 0, this->proj_data_info_ptr->get_max_axial_pos_num(0), 0));
        assert(fabs(m_last + m_first) < m_last * 10E-4);
    }
#endif
    this->shift_detector_coordinates_to_origin =
            CartesianCoordinate3D<float>(this->proj_data_info_ptr->get_m(Bin(0, 0, 0, 0)), 0, 0);
    float total_scatter = 0 ;

    for (vs_num.segment_num() = this->proj_data_info_ptr->get_min_segment_num();
         vs_num.segment_num() <= this->proj_data_info_ptr->get_max_segment_num();
         ++vs_num.segment_num())
    {
        for (vs_num.view_num() = this->proj_data_info_ptr->get_min_view_num();
             vs_num.view_num() <= this->proj_data_info_ptr->get_max_view_num();
             ++vs_num.view_num())
        {
            total_scatter += this->process_data_for_view_segment_num(vs_num);
            bin_counter +=
                    this->proj_data_info_ptr->get_num_axial_poss(vs_num.segment_num()) *
                    this->proj_data_info_ptr->get_num_tangential_poss();
            /* ////////////////// SCATTER ESTIMATION TIME ////////////////
       */
            {
                wall_clock_timer.stop(); // must be stopped before getting the value
                info(boost::format("%1% bins  Total time elapsed %2% sec "
                                   "\tTime remaining about %3% minutes")
                     % bin_counter
                     % wall_clock_timer.value()
                     % ((wall_clock_timer.value() - previous_timer)
                        * (total_bins - bin_counter) / (bin_counter - previous_bin_count) / 60));
                previous_timer = wall_clock_timer.value() ;
                previous_bin_count = bin_counter ;
                wall_clock_timer.start();
            }
            /* ////////////////// end SCATTER ESTIMATION TIME ////////////////
       */
        }
    }

    bin_timer.stop();
    wall_clock_timer.stop();
    //    this->write_log(wall_clock_timer.value(), total_scatter);

    if (detection_points_vector.size() != static_cast<unsigned int>(total_detectors))
    {
        warning("Expected num detectors: %d, but found %d\n",
                total_detectors, detection_points_vector.size());
        return Succeeded::no;
    }

    //    std::string output_filename = "./scatter_estimated_in_0";

    //    ProjDataInMemory* object =
    //            dynamic_cast<ProjDataInMemory *> (this->output_proj_data_sptr.get());
    //    object->write_to_file(output_filename);

    return Succeeded::yes;
}

//xxx double
double
ScatterSimulation::
process_data_for_view_segment_num(const ViewSegmentNumbers& vs_num)
{
    // First construct a vector of all bins that we'll process.
    // The reason for making this list before the actual calculation is that we can then parallelise over all bins
    // without having to think about double loops.
    std::vector<Bin> all_bins;
    {
        Bin bin(vs_num.segment_num(), vs_num.view_num(), 0, 0);

        for (bin.axial_pos_num() = this->proj_data_info_ptr->get_min_axial_pos_num(bin.segment_num());
             bin.axial_pos_num() <= this->proj_data_info_ptr->get_max_axial_pos_num(bin.segment_num());
             ++bin.axial_pos_num())
        {
            for (bin.tangential_pos_num() = this->proj_data_info_ptr->get_min_tangential_pos_num();
                 bin.tangential_pos_num() <= this->proj_data_info_ptr->get_max_tangential_pos_num();
                 ++bin.tangential_pos_num())
            {
                all_bins.push_back(bin);
            }
        }
    }
    // now compute scatter for all bins
    double total_scatter = 0;
    Viewgram<float> viewgram =
            this->output_proj_data_sptr->get_empty_viewgram(vs_num.view_num(), vs_num.segment_num());
#ifdef STIR_OPENMP
#pragma omp parallel for reduction(+:total_scatter) schedule(dynamic)
#endif

    for (int i = 0; i < static_cast<int>(all_bins.size()); ++i)
    {
        const Bin bin = all_bins[i];
        unsigned det_num_A = 0; // initialise to avoid compiler warnings
        unsigned det_num_B = 0;
        this->find_detectors(det_num_A, det_num_B, bin);
        const double scatter_ratio =
                scatter_estimate(det_num_A, det_num_B);
        viewgram[bin.axial_pos_num()][bin.tangential_pos_num()] =
                static_cast<float>(scatter_ratio);
        total_scatter += scatter_ratio;
    } // end loop over bins

    if (this->output_proj_data_sptr->set_viewgram(viewgram) == Succeeded::no)
        error("ScatterEstimationByBin: error writing viewgram");

    return static_cast<double>(viewgram.sum());
}

void
ScatterSimulation::set_defaults()
{
    this->attenuation_threshold =  0.01f ;
    this->random = true;
    this->use_cache = true;
    this->density_image_filename = "";
    this->density_image_for_scatter_points_filename = "";
    this->scatter_proj_data_filename = "";
    this->remove_cache_for_integrals_over_activity();
    this->remove_cache_for_integrals_over_attenuation();
}

void
ScatterSimulation::
ask_parameters()
{

}

void
ScatterSimulation::initialise_keymap()
{
    this->parser.add_start_key("Scatter Simulation Parameters");
    this->parser.add_stop_key("end Scatter Simulation Parameters");
    this->parser.add_key("scatter projdata filename",
                         &this->scatter_proj_data_filename);
    this->parser.add_key("attenuation image filename",
                         &this->density_image_filename);
    this->parser.add_key("attenuation image for scatter points filename",
                         &this->density_image_for_scatter_points_filename);
    this->parser.add_key("activity image filename",
                         &this->activity_image_filename);
    this->parser.add_key("attenuation threshold",
                         &this->attenuation_threshold);
    this->parser.add_key("random", &this->random);
    this->parser.add_key("use cache", &this->use_cache);
}


bool
ScatterSimulation::
post_processing()
{
    if (this->scatter_proj_data_filename.size() > 0)
        this->set_scatter_proj_data_info(this->scatter_proj_data_filename);

    if (this->activity_image_filename.size() > 0)
        this->set_activity_image(this->activity_image_filename);

    if (this->density_image_filename.size() > 0)
        this->set_density_image(this->density_image_filename);

    if(this->density_image_for_scatter_points_filename.size() > 0 && !this->recompute_sub_atten_image)
        this->set_density_image_for_scatter_points(this->density_image_for_scatter_points_filename);

    return false;
}

END_NAMESPACE_STIR
