/*
    Copyright (C) 2004 - 2009 Hammersmith Imanet Ltd
    Copyright (C) 2013 - 2016, 2019, 2020, 2022  University College London
    Copyright (C) 2018-2019, University of Hull
    Copyright (C) 2021, University of Leeds
    Copyright (C) 2022, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup scatter
  \brief Definition of class stir::ScatterSimulation.

  \author Nikos Efthimiou
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  \author Viet Ahn Dao
  \author Daniel Deidda
*/
#include "stir/scatter/ScatterSimulation.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/Bin.h"
#include "stir/CPUTimer.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/IndexRange3D.h"
#include "stir/Viewgram.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/info.h"
#include "stir/error.h"
#include "stir/stream.h"
#include "stir/round.h"
#include <fstream>
#include <algorithm>
#include <boost/format.hpp>

#include "stir/zoom.h"
#include "stir/SSRB.h"

#include "stir/stir_math.h"
#include "stir/zoom.h"
#include "stir/ZoomOptions.h"
#include "stir/NumericInfo.h"

START_NAMESPACE_STIR

ScatterSimulation::
ScatterSimulation()
{
    this->set_defaults();
}

ScatterSimulation::
~ScatterSimulation()
{
    // Sometimes I get a segfault without this line.
    scatt_points_vector.clear();
}

bool ScatterSimulation::get_use_cache() const
{
  return this->use_cache;
}

void ScatterSimulation::set_use_cache(bool value)
{
  if (value == this->use_cache)
    return;

  this->remove_cache_for_integrals_over_activity();
  this->remove_cache_for_integrals_over_attenuation();
  this->use_cache = value;
}

Succeeded
ScatterSimulation::
process_data()
{
    if (!this->_already_set_up)
        error("ScatterSimulation: need to call set_up() first");
    if(is_null_ptr(output_proj_data_sptr))
        error("ScatterSimulation: output projection data not set. Aborting.");

    // this is useful in the scatter estimation process.
    this->output_proj_data_sptr->fill(0.f);
    // check if output has same info as templates
    {
      if ((*output_proj_data_sptr->get_proj_data_info_sptr()) !=
          (*this->get_template_proj_data_info_sptr()))
        error("ScatterSimulation: output projection data incompatible with what was used for set_up()");
      // TODO enable check on exam_info but this has no operator== yet
    }
    info("ScatterSimulator: Running Scatter Simulation ...");
    info("ScatterSimulator: Initialising ...");

    ViewSegmentNumbers vs_num;
    /* ////////////////// SCATTER ESTIMATION TIME //////////////// */
    CPUTimer bin_timer;
    bin_timer.start();
    // variables to report (remaining) time
    HighResWallClockTimer wall_clock_timer;
    double previous_timer = 0 ;
    int previous_bin_count = 0 ;
    int bin_counter = 0;
    int axial_bins = 0 ;
    wall_clock_timer.start();

    for (vs_num.segment_num() = this->proj_data_info_sptr->get_min_segment_num();
         vs_num.segment_num() <= this->proj_data_info_sptr->get_max_segment_num();
         ++vs_num.segment_num())
        axial_bins += this->proj_data_info_sptr->get_num_axial_poss(vs_num.segment_num());

    const int total_bins =
            this->proj_data_info_sptr->get_num_views() * axial_bins *
            this->proj_data_info_sptr->get_num_tangential_poss();
    /* ////////////////// end SCATTER ESTIMATION TIME //////////////// */
    float total_scatter = 0 ;

    info("ScatterSimulator: Initialization finished ...");
    for (vs_num.segment_num() = this->proj_data_info_sptr->get_min_segment_num();
         vs_num.segment_num() <= this->proj_data_info_sptr->get_max_segment_num();
         ++vs_num.segment_num())
    {
        for (vs_num.view_num() = this->proj_data_info_sptr->get_min_view_num();
             vs_num.view_num() <= this->proj_data_info_sptr->get_max_view_num();
             ++vs_num.view_num())
        {
            total_scatter += this->process_data_for_view_segment_num(vs_num);
            bin_counter +=
                    this->proj_data_info_sptr->get_num_axial_poss(vs_num.segment_num()) *
                    this->proj_data_info_sptr->get_num_tangential_poss();
            /* ////////////////// SCATTER ESTIMATION TIME //////////////// */
            {
                wall_clock_timer.stop(); // must be stopped before getting the value
                info(boost::format("%1$5u / %2% bins done. Total time elapsed %3$5.2f secs, remaining about %4$5.2f mins (ignoring caching).")
                     % bin_counter % total_bins
                     % wall_clock_timer.value()
                     % ((wall_clock_timer.value() - previous_timer)
                        * (total_bins - bin_counter) / (bin_counter - previous_bin_count) / 60),
		     /* verbosity level*/ 3);
                previous_timer = wall_clock_timer.value() ;
                previous_bin_count = bin_counter ;
                wall_clock_timer.start();
            }
            /* ////////////////// end SCATTER ESTIMATION TIME //////////////// */
        }
    }

    bin_timer.stop();
    wall_clock_timer.stop();

    if (detection_points_vector.size() != static_cast<unsigned int>(total_detectors))
    {
        warning("Expected num detectors: %d, but found %d\n",
                total_detectors, detection_points_vector.size());
        return Succeeded::no;
    }

    info(boost::format("TOTAL SCATTER counts before upsampling and norm = %g") % total_scatter);
    this->write_log(wall_clock_timer.value(), total_scatter);
    return Succeeded::yes;
}

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

        for (bin.axial_pos_num() = this->proj_data_info_sptr->get_min_axial_pos_num(bin.segment_num());
             bin.axial_pos_num() <= this->proj_data_info_sptr->get_max_axial_pos_num(bin.segment_num());
             ++bin.axial_pos_num())
        {
            for (bin.tangential_pos_num() = this->proj_data_info_sptr->get_min_tangential_pos_num();
                 bin.tangential_pos_num() <= this->proj_data_info_sptr->get_max_tangential_pos_num();
                 ++bin.tangential_pos_num())
            {
                all_bins.push_back(bin);
            }
        }
    }

    // now compute scatter for all bins
    double total_scatter = 0.;
    Viewgram<float> viewgram =
            this->output_proj_data_sptr->get_empty_viewgram(vs_num.view_num(), vs_num.segment_num());
#ifdef STIR_OPENMP
#pragma omp parallel for reduction(+:total_scatter) schedule(dynamic)
#endif

    for (int i = 0; i < static_cast<int>(all_bins.size()); ++i)
    {
        const Bin bin = all_bins[i];
        const double scatter_ratio = scatter_estimate(bin);

#if defined STIR_OPENMP 
#  if _OPENMP >= 201107
#    pragma omp atomic write
#  else
#    pragma omp critical(ScatterSimulationByBin_process_data_for_view_segment_num)
#  endif
#endif
        viewgram[bin.axial_pos_num()][bin.tangential_pos_num()] =
                static_cast<float>(scatter_ratio);
        total_scatter += static_cast<double>(scatter_ratio);
    } // end loop over bins

    if (this->output_proj_data_sptr->set_viewgram(viewgram) == Succeeded::no)
        error("ScatterSimulation: error writing viewgram");

    return total_scatter;
}

void
ScatterSimulation::set_defaults()
{
    this->attenuation_threshold =  0.01f ;
    this->randomly_place_scatter_points = true;
    this->use_cache = true;
    this->zoom_xy = -1.f;
    this->zoom_z = -1.f;
    this->zoom_size_xy = -1;
    this->zoom_size_z = -1;
    this->downsample_scanner_bool = false;
    this->downsample_scanner_dets = -1;
    this->downsample_scanner_rings = -1;
    this->density_image_filename = "";
    this->activity_image_filename = "";
    this->density_image_for_scatter_points_output_filename ="";
    this->density_image_for_scatter_points_filename = "";
    this->template_proj_data_filename = "";
    this->remove_cache_for_integrals_over_activity();
    this->remove_cache_for_integrals_over_attenuation();
    this->_already_set_up = false;
}

void
ScatterSimulation::
ask_parameters()
{
    this->attenuation_threshold = ask_num("attenuation threshold(cm^-1)",0.0f, 5.0f, 0.01f);
    this->randomly_place_scatter_points = ask_num("random place scatter points?",0, 1, 1);
    this->use_cache =  ask_num(" Use cache?",0, 1, 1);
    this->density_image_filename = ask_string("density image filename", "");
    this->activity_image_filename = ask_string("activity image filename", "");
    //this->density_image_for_scatter_points_filename = ask_string("density image for scatter points filename", "");
    this->template_proj_data_filename = ask_string("Scanner ProjData filename", "");
}

void
ScatterSimulation::initialise_keymap()
{

    // this->parser.add_start_key("Scatter Simulation Parameters");
    // this->parser.add_stop_key("end Scatter Simulation Parameters");
    this->parser.add_key("template projdata filename",
                         &this->template_proj_data_filename);
    this->parser.add_key("attenuation image filename",
                         &this->density_image_filename);
    this->parser.add_key("attenuation image for scatter points filename",
                         &this->density_image_for_scatter_points_filename);
    this->parser.add_key("zoom XY for attenuation image for scatter points",
                         &this->zoom_xy);
    this->parser.add_key("zoom Z for attenuation image for scatter points",
                         &this->zoom_z);
    this->parser.add_key("XY size of downsampled image for scatter points",
                         &this->zoom_size_xy);
    this->parser.add_key("Z size of downsampled image for scatter points",
                         &this->zoom_size_z);
    this->parser.add_key("attenuation image for scatter points output filename",
                         &this->density_image_for_scatter_points_output_filename);
    this->parser.add_key("downsampled scanner number of detectors per ring",
                         &this->downsample_scanner_dets);
    this->parser.add_key("downsampled scanner number of rings",
                         &this->downsample_scanner_rings);
    this->parser.add_key("activity image filename",
                         &this->activity_image_filename);
    this->parser.add_key("attenuation threshold",
                         &this->attenuation_threshold);
    this->parser.add_key("output filename prefix",
                         &this->output_proj_data_filename);
    this->parser.add_key("downsample scanner",
                         &this->downsample_scanner_bool);
    this->parser.add_key("randomly place scatter points", &this->randomly_place_scatter_points);
    this->parser.add_key("use cache", &this->use_cache);
}


bool
ScatterSimulation::
post_processing()
{

    if (this->template_proj_data_filename.size() > 0)
        this->set_template_proj_data_info(this->template_proj_data_filename);

    if (this->activity_image_filename.size() > 0)
        this->set_activity_image(this->activity_image_filename);

    if (this->density_image_filename.size() > 0)
        this->set_density_image(this->density_image_filename);

    if(this->density_image_for_scatter_points_filename.size() > 0)
        this->set_density_image_for_scatter_points(this->density_image_for_scatter_points_filename);

    if (this->output_proj_data_filename.size() > 0)
        this->set_output_proj_data(this->output_proj_data_filename);

    return false;
}

Succeeded
ScatterSimulation::
set_up()
{
    if (is_null_ptr(proj_data_info_sptr))
        error("ScatterSimulation: projection data info not set. Aborting.");

    if (!proj_data_info_sptr->has_energy_information())
        error("ScatterSimulation: scanner energy resolution information not set. Aborting.");

    if (is_null_ptr(template_exam_info_sptr))
        error("ScatterSimulation: projection data info not set. Aborting.");

    if(!template_exam_info_sptr->has_energy_information())
        error("ScatterSimulation: template energy window information not set. Aborting.");

    if(is_null_ptr(activity_image_sptr))
        error("ScatterSimulation: activity image not set. Aborting.");

    if(is_null_ptr(density_image_sptr))
        error("ScatterSimulation: density image not set. Aborting.");

    if(downsample_scanner_bool)
      {
        if (this->_already_set_up)
          error("ScatterSimulation: set_up() called twice. This is currently not supported.");

        downsample_scanner();
      }

    if(is_null_ptr(density_image_for_scatter_points_sptr))
    {
        if (this->_already_set_up)
          error("ScatterSimulation: set_up() called twice. This is currently not supported.");
        downsample_density_image_for_scatter_points(zoom_xy, zoom_z, zoom_size_xy, zoom_size_z);
    }

//    {
//        this->output_proj_data_sptr.reset(new ProjDataInMemory(this->template_exam_info_sptr,
//                                                               this->proj_data_info_sptr->create_shared_clone()));
//        this->output_proj_data_sptr->fill(0.0);
//        info("ScatterSimulation: output projection data created.");
//    }


    // Note: horrible shift used for detection_points_vector
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
        if(dynamic_cast<ProjDataInfoCylindricalNoArcCorr*> (proj_data_info_sptr.get())){
            auto ptr = dynamic_cast<ProjDataInfoCylindricalNoArcCorr*> (proj_data_info_sptr.get());
            ptr->find_cartesian_coordinates_of_detection(detector_coord_A, detector_coord_B, Bin(0, 0, 0, 0));
        }else{
            auto ptr = dynamic_cast<ProjDataInfoBlocksOnCylindricalNoArcCorr*> (proj_data_info_sptr.get());
            ptr->find_cartesian_coordinates_of_detection(detector_coord_A, detector_coord_B, Bin(0, 0, 0, 0));
        }
        
//        if(this->proj_data_info_sptr->get_scanner_sptr()->get_scanner_geometry()=="Cylindrical"){
            assert(detector_coord_A.z() == 0);
            assert(detector_coord_B.z() == 0);
//        }
        // check that get_m refers to the middle of the scanner
        const float m_first =
                this->proj_data_info_sptr->get_m(Bin(0, 0, this->proj_data_info_sptr->get_min_axial_pos_num(0), 0));
        const float m_last =
                this->proj_data_info_sptr->get_m(Bin(0, 0, this->proj_data_info_sptr->get_max_axial_pos_num(0), 0));
//        if(this->proj_data_info_sptr->get_scanner_sptr()->get_scanner_geometry()=="Cylindrical")
        assert(fabs(m_last + m_first) < m_last * 10E-4);
    }
#endif
    if(dynamic_cast<ProjDataInfoCylindricalNoArcCorr*> (proj_data_info_sptr.get())){
            this->shift_detector_coordinates_to_origin =
            CartesianCoordinate3D<float>(this->proj_data_info_sptr->get_m(Bin(0, 0, 0, 0)), 0, 0);
    }else{
        if(dynamic_cast<ProjDataInfoBlocksOnCylindricalNoArcCorr*> (proj_data_info_sptr.get())){
            // align BlocksOnCylindrical scanner ring 0 to z=0.
            this->shift_detector_coordinates_to_origin =
            CartesianCoordinate3D<float>(this->proj_data_info_sptr->get_m(Bin(0, 0, 0, 0)), 0, 0);
        }
        // align Generic geometry here.
    }

#if 1
    // checks on image zooming to avoid getting incorrect results
    {
      check_z_to_middle_consistent(*this->activity_image_sptr, "activity");
      check_z_to_middle_consistent(*this->density_image_sptr, "attenuation");
      check_z_to_middle_consistent(*this->density_image_for_scatter_points_sptr, "scatter-point");
    }
#endif
    this->initialise_cache_for_scattpoint_det_integrals_over_attenuation();
    this->initialise_cache_for_scattpoint_det_integrals_over_activity();

    this->_already_set_up = true;

    return Succeeded::yes;
}

void
ScatterSimulation::
check_z_to_middle_consistent(const DiscretisedDensity<3,float>& _image, const std::string& name) const
{
  const VoxelsOnCartesianGrid<float> & image = dynamic_cast<VoxelsOnCartesianGrid<float> const& >(_image);
  const float z_to_middle =
    (image.get_max_index() + image.get_min_index())*image.get_voxel_size().z()/2.F;

# if 0
  const Scanner& scanner = *this->proj_data_info_sptr->get_scanner_ptr();
  const float z_to_middle_standard =
    (scanner.get_num_rings()-1) * scanner.get_ring_spacing()/2;
#endif
  const VoxelsOnCartesianGrid<float> & act_image =
    dynamic_cast<VoxelsOnCartesianGrid<float> const& >(*this->activity_image_sptr);
  const float z_to_middle_standard =
    (act_image.get_max_index() + act_image.get_min_index())*act_image.get_voxel_size().z()/2.F;

  if (abs(z_to_middle - z_to_middle_standard) > .1)
    error(boost::format("ScatterSimulation: limitation in #planes and voxel-size for the %1% image.\n"
                        "This would cause a shift of %2%mm w.r.t. the activity image.\n"
                        "(see https://github.com/UCL/STIR/issues/495.")
          % name % (z_to_middle - z_to_middle_standard));
}

void
ScatterSimulation::
set_activity_image_sptr(const shared_ptr<const DiscretisedDensity<3,float> > arg)
{
    if (is_null_ptr(arg) )
        error("ScatterSimulation: Unable to set the activity image");

    this->activity_image_sptr = arg;
    this->remove_cache_for_integrals_over_activity();
    this->_already_set_up = false;
}

void
ScatterSimulation::
set_activity_image(const std::string& filename)
{
    this->activity_image_filename = filename;
    shared_ptr<DiscretisedDensity<3,float> > sptr(read_from_file<DiscretisedDensity<3,float> >(filename));
    this->set_activity_image_sptr(sptr);
}

void
ScatterSimulation::
set_density_image_sptr(const shared_ptr<const DiscretisedDensity<3,float> > arg)
{
    if (is_null_ptr(arg) )
        error("ScatterSimulation: Unable to set the density image");
    this->density_image_sptr=arg;
    // make sure that we're not re-using a previously interpolated image for scatter points
    this->density_image_for_scatter_points_sptr.reset();
    this->remove_cache_for_integrals_over_attenuation();
    this->_already_set_up = false;
}

void
ScatterSimulation::
set_density_image(const std::string& filename)
{
    this->density_image_filename=filename;
    shared_ptr<DiscretisedDensity<3,float> > sptr(read_from_file<DiscretisedDensity<3,float> >(filename));
    this->set_density_image_sptr(sptr);
}

void
ScatterSimulation::
set_density_image_for_scatter_points_sptr(shared_ptr<const DiscretisedDensity<3,float> > arg)
{
    if (is_null_ptr(arg) )
        error("ScatterSimulation: Unable to set the density image for scatter points.");
    this->density_image_for_scatter_points_sptr.reset(
                new VoxelsOnCartesianGrid<float>(*dynamic_cast<const VoxelsOnCartesianGrid<float> *>(arg.get())));
    this->sample_scatter_points();
    this->remove_cache_for_integrals_over_attenuation();
    this->_already_set_up = false;
}

const DiscretisedDensity<3,float>&
ScatterSimulation::
get_activity_image() const
{
    return *activity_image_sptr;
}

const DiscretisedDensity<3,float>&
ScatterSimulation::
get_attenuation_image() const
{
    return *density_image_sptr;
}

const DiscretisedDensity<3,float>&
ScatterSimulation::
get_attenuation_image_for_scatter_points() const
{
    return *density_image_for_scatter_points_sptr;
}

shared_ptr<const DiscretisedDensity<3,float> >
ScatterSimulation::
get_density_image_for_scatter_points_sptr() const
{
    return density_image_for_scatter_points_sptr;
}

void
ScatterSimulation::
set_density_image_for_scatter_points(const std::string& filename)
{
    this->density_image_for_scatter_points_filename=filename;
    shared_ptr<DiscretisedDensity<3,float> > sptr(read_from_file<DiscretisedDensity<3,float> >(filename));    
    this->set_density_image_for_scatter_points_sptr(sptr);
    this->_already_set_up = false;
}

void
ScatterSimulation::
set_image_downsample_factors(float _zoom_xy, float _zoom_z,
                             int _size_zoom_xy, int _size_zoom_z)
{
    if (_zoom_xy<0.F || _zoom_z<0.F)
        error("ScatterSimulation: at least one zoom factor for the scatter-point image is negative");
    zoom_xy = _zoom_xy;
    zoom_z = _zoom_z;
    zoom_size_xy = _size_zoom_xy;
    zoom_size_z = _size_zoom_z;
    _already_set_up = false;
}

void
ScatterSimulation::
downsample_density_image_for_scatter_points(float _zoom_xy, float _zoom_z,
					    int _size_xy, int _size_z)
{
    if (is_null_ptr(this->density_image_sptr))
        error("ScatterSimulation: downsampling function called before attenuation image is set");

    const VoxelsOnCartesianGrid<float> & tmp_att = dynamic_cast<const VoxelsOnCartesianGrid<float>& >(*this->density_image_sptr);

    const int old_x = tmp_att.get_x_size();
    const int old_y = tmp_att.get_y_size();
    const int old_z = tmp_att.get_z_size();

    if (_zoom_xy < 0 || _zoom_z < 0)
    {
        VoxelsOnCartesianGrid<float> tmpl_density(this->density_image_sptr->get_exam_info_sptr(), *proj_data_info_sptr);
	info(boost::format("ScatterSimulation: template density to find zoom factors: voxel-sizes %1%, size %2%, product %3%")
	     % tmpl_density.get_voxel_size()
	     % tmpl_density.get_lengths()
	     % (tmpl_density.get_voxel_size() * BasicCoordinate<3,float>(tmpl_density.get_lengths())),
	     3);
	if (_zoom_xy < 0)
	  _zoom_xy = tmp_att.get_voxel_size().x() / tmpl_density.get_voxel_size().x();

        const float z_length =
          std::max((old_z+1)*tmp_att.get_voxel_size().z(),
                   (tmpl_density.get_z_size()+1)*tmpl_density.get_voxel_size().z());
	if (_zoom_z < 0)
          {
            if (_size_z < 0)
              _size_z = (tmpl_density.get_z_size()+1)/2;
            zoom_z = tmp_att.get_voxel_size().z() / (z_length/_size_z);
          }
        else
          zoom_z = _zoom_z;
    }
    else
      zoom_z = _zoom_z;

    set_image_downsample_factors(_zoom_xy, zoom_z, _size_xy, _size_z);

    int new_x = zoom_size_xy == -1 ? static_cast<int>(old_x * zoom_xy + 1) : zoom_size_xy;
    int new_y = zoom_size_xy == -1 ? static_cast<int>(old_y * zoom_xy + 1) : zoom_size_xy;
    const int new_z = zoom_size_z == -1 ? static_cast<int>(old_z * zoom_z + 1) : zoom_size_z;

    // make sizes odd to avoid edge effects and half-voxel shifts
    if (new_x%2 == 0)
      new_x++;
    if (new_y%2 == 0)
      new_y++;

    // adjust zoom_z to cope with ugly "shift to middle of scanner" problem
    {
      // see http://github.com/UCL/STIR/issues/495
      zoom_z = static_cast<float>(new_z-1)/(old_z-1);
      if (_zoom_z>0 && abs(zoom_z - _zoom_z)>.1)
        error(boost::format("Current limitation in ScatterSimulation: use zoom_z==-1 or %1%")
              % zoom_z);
    }

    const CartesianCoordinate3D<float> new_voxel_size =
      tmp_att.get_voxel_size() / make_coordinate(zoom_z, zoom_xy, zoom_xy);
    // create new image of appropriate size
    shared_ptr<VoxelsOnCartesianGrid<float> >
      vox_sptr(new VoxelsOnCartesianGrid<float>(tmp_att.get_exam_info_sptr(),
						IndexRange3D(0, new_z-1,
							     -new_y/2, -new_y/2+new_y-1,
							     -new_x/2, -new_x/2+new_x-1),
						tmp_att.get_origin(),
						new_voxel_size
						));
    // assign to class member
    this->density_image_for_scatter_points_sptr = vox_sptr;
    info(boost::format("ScatterSimulation: scatter-point image: voxel-sizes %1%, size %2%, total-length %3%")
	 % vox_sptr->get_voxel_size()
	 % vox_sptr->get_lengths()
	 % (vox_sptr->get_voxel_size() * (BasicCoordinate<3,float>(vox_sptr->get_lengths()+1.F))),
	 2);
    // fill values from original attenuation image
    ZoomOptions scaling(ZoomOptions::preserve_values);
    zoom_image( *vox_sptr, tmp_att, scaling);

#if 0
    // do some checks
    {
        float image_plane_spacing = dynamic_cast<VoxelsOnCartesianGrid<float> *>(density_image_for_scatter_points_sptr.get())->get_grid_spacing()[1];
        const float num_planes_per_scanner_ring_float =
                proj_data_info_sptr->get_ring_spacing() / image_plane_spacing;

        int num_planes_per_scanner_ring = round(num_planes_per_scanner_ring_float);

        if (fabs(num_planes_per_scanner_ring_float - num_planes_per_scanner_ring) > 1.E-2)
            warning(boost::format("ScatterSimulation: DataSymmetriesForBins_PET_CartesianGrid can currently only support z-grid spacing "
                                  "equal to the ring spacing of the scanner divided by an integer. This is not a problem here."
                                  "However, if you are planning to use this in an Scatter Estimation loop it might create problems."
                                  "Reconsider your z-axis downsampling."
                                  "(Image z-spacing is %1% and ring spacing is %2%)") % image_plane_spacing % proj_data_info_sptr->get_ring_spacing());
    }
#endif

    if(this->density_image_for_scatter_points_output_filename.size()>0)
      OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
	write_to_file(density_image_for_scatter_points_output_filename,
		      *this->density_image_for_scatter_points_sptr);

    this->sample_scatter_points();
    this->remove_cache_for_integrals_over_attenuation();
    this->_already_set_up = false;
}


void
ScatterSimulation::
set_output_proj_data_sptr(const shared_ptr<const ExamInfo> _exam,
                          const shared_ptr<const ProjDataInfo> _info,
                          const std::string & filename)
{
    if (filename.size() > 0 )
        this->output_proj_data_sptr.reset(new ProjDataInterfile(_exam,
                                                                _info,
                                                                filename,
                                                                std::ios::in | std::ios::out | std::ios::trunc));
    else
        this->output_proj_data_sptr.reset( new ProjDataInMemory(_exam,
                                                                _info));
}

shared_ptr<ProjData>
ScatterSimulation::
get_output_proj_data_sptr() const
{

    if(is_null_ptr(this->output_proj_data_sptr))
    {
        error("ScatterSimulation: No output ProjData set. Aborting.");
    }

    return this->output_proj_data_sptr;
}

void
ScatterSimulation::
set_output_proj_data(const std::string& filename)
{

    if(is_null_ptr(this->proj_data_info_sptr))
    {
        error("ScatterSimulation: Template ProjData has not been set. Aborting.");
    }

    this->output_proj_data_filename = filename;
    shared_ptr<ProjData> tmp_sptr;

    if (is_null_ptr(this->template_exam_info_sptr))
    {
        shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
        if (filename.empty())
        {
            tmp_sptr.reset(new ProjDataInMemory(exam_info_sptr,
                                                this->proj_data_info_sptr->create_shared_clone()));
        }
        else
        {
            tmp_sptr.reset(new ProjDataInterfile(exam_info_sptr,
                                                 this->proj_data_info_sptr->create_shared_clone(),
                                                 this->output_proj_data_filename,
                                                 std::ios::in | std::ios::out | std::ios::trunc));
        }
        
    }
    else
    {
        if (filename.empty())
        {
            tmp_sptr.reset(new ProjDataInMemory(this->template_exam_info_sptr,
                                                this->proj_data_info_sptr->create_shared_clone()));
        }
        else
        {
            tmp_sptr.reset(new ProjDataInterfile(this->template_exam_info_sptr,
                                                 this->proj_data_info_sptr->create_shared_clone(),
                                                 this->output_proj_data_filename,
                                                 std::ios::in | std::ios::out | std::ios::trunc));
        }
    }

    set_output_proj_data_sptr(tmp_sptr);
}


void
ScatterSimulation::
set_output_proj_data_sptr(shared_ptr<ProjData> arg)
{
    this->output_proj_data_sptr = arg;
}

shared_ptr<const ProjDataInfo>
ScatterSimulation::
get_template_proj_data_info_sptr() const
{
    return this->proj_data_info_sptr;
}

shared_ptr<const ExamInfo>
ScatterSimulation::get_exam_info_sptr() const
{
    return this->template_exam_info_sptr;
}

void
ScatterSimulation::
set_template_proj_data_info(const std::string& filename)
{
    this->template_proj_data_filename = filename;
    shared_ptr<ProjData> template_proj_data_sptr(ProjData::read_from_file(this->template_proj_data_filename));

    this->set_exam_info(template_proj_data_sptr->get_exam_info());

    this->set_template_proj_data_info(*template_proj_data_sptr->get_proj_data_info_sptr());
}

void
ScatterSimulation::set_template_proj_data_info(const ProjDataInfo& arg)
{
    this->_already_set_up = false;
    this->proj_data_info_sptr.reset(dynamic_cast<ProjDataInfoBlocksOnCylindricalNoArcCorr* >(arg.clone()));

    if (is_null_ptr(this->proj_data_info_sptr)){
        this->proj_data_info_sptr.reset(dynamic_cast<ProjDataInfoCylindricalNoArcCorr* >(arg.clone()));
        if (is_null_ptr(this->proj_data_info_sptr)){
            error("ScatterSimulation: Can only handle non-arccorrected data");
        }
    }

    // find final size of detection_points_vector
    this->total_detectors =
            this->proj_data_info_sptr->get_scanner_ptr()->get_num_rings()*
            this->proj_data_info_sptr->get_scanner_ptr()->get_num_detectors_per_ring ();

    // get rid of any previously stored points
    this->detection_points_vector.clear();
    // reserve space to avoid reallocation, but the actual size will grow dynamically
    this->detection_points_vector.reserve(static_cast<std::size_t>(this->total_detectors));

    // set to negative value such that this will be recomputed
    this->detector_efficiency_no_scatter = -1.F;

    // remove any cached values as they'd be incorrect if the sizes changes
    this->remove_cache_for_integrals_over_attenuation();
    this->remove_cache_for_integrals_over_activity();
}

void
ScatterSimulation::
set_exam_info(const ExamInfo& arg)
{
  this->_already_set_up = false;
  this->template_exam_info_sptr = arg.create_shared_clone();
}

void
ScatterSimulation::
set_exam_info_sptr(const shared_ptr<const ExamInfo> arg)
{
    this->_already_set_up = false;
    this->template_exam_info_sptr = arg->create_shared_clone();
}

Succeeded
ScatterSimulation::downsample_scanner(int new_num_rings, int new_num_dets)
{
    if (new_num_rings <= 0)
    {
	if(downsample_scanner_rings > 0)
            new_num_rings = downsample_scanner_rings;
        else if (!is_null_ptr(proj_data_info_sptr))
	  {
        const float total_axial_length = proj_data_info_sptr->get_scanner_sptr()->get_num_rings() *
          proj_data_info_sptr->get_scanner_sptr()->get_ring_spacing();

	    new_num_rings = round(total_axial_length / 20.F + 0.5F);
	  }
	else
            return Succeeded::no;
    }

    const Scanner *const old_scanner_ptr = this->proj_data_info_sptr->get_scanner_ptr();
    shared_ptr<Scanner> new_scanner_sptr( new Scanner(*old_scanner_ptr));

    //make a downsampled scanner with no gaps for blocksOnCylindrical
    if (new_scanner_sptr->get_scanner_geometry()!="Cylindrical")
    {
        new_num_dets=proj_data_info_sptr->get_scanner_ptr()->get_num_detectors_per_ring();
        // preserve the length of the scanner the following includes gaps
        float scanner_length_block = new_scanner_sptr->get_num_axial_buckets()*
                new_scanner_sptr->get_num_axial_blocks_per_bucket()*
                new_scanner_sptr->get_axial_block_spacing();
        new_scanner_sptr->set_num_axial_blocks_per_bucket(1);
//        new_scanner_sptr->set_num_transaxial_blocks_per_bucket(1);
        
        
        new_scanner_sptr->set_num_rings(new_num_rings);
//        float transaxial_bucket_spacing=old_scanner_ptr->get_transaxial_block_spacing()
//                *old_scanner_ptr->get_num_transaxial_blocks_per_bucket();
        float new_ring_spacing=scanner_length_block/new_scanner_sptr->get_num_rings();
//        int num_trans_buckets=old_scanner_ptr->get_num_transaxial_buckets();
// get a new number of detectors that is a multiple of the number of buckets to preserve scanner shape
//        float frac,whole;
//        frac = std::modf(float(new_num_dets/new_scanner_sptr->get_num_transaxial_buckets()), &whole);
//        int newest_num_dets=whole*new_scanner_sptr->get_num_transaxial_buckets();
//        new_scanner_sptr->set_num_detectors_per_ring(newest_num_dets);
//        int new_transaxial_dets_per_bucket=newest_num_dets/num_trans_buckets;
//        float new_det_spacing=transaxial_bucket_spacing/new_transaxial_dets_per_bucket;
        
        new_scanner_sptr->set_axial_crystal_spacing(new_ring_spacing);
        new_scanner_sptr->set_ring_spacing(new_ring_spacing);
        new_scanner_sptr->set_num_axial_crystals_per_block(new_num_rings);
        new_scanner_sptr->set_axial_block_spacing(new_ring_spacing
                    * new_scanner_sptr->get_num_axial_crystals_per_block());
        
//        new_scanner_sptr->set_num_transaxial_crystals_per_block(new_transaxial_dets_per_bucket);
//        new_scanner_sptr->set_transaxial_crystal_spacing(new_det_spacing);
//        new_scanner_sptr->set_transaxial_block_spacing(new_det_spacing
//                    * new_scanner_sptr->get_num_transaxial_crystals_per_block());
    }
    else{
        if (new_num_dets <= 0)
        {
            if(downsample_scanner_dets > 0)
                new_num_dets = downsample_scanner_dets;
            else
                new_num_dets=64;
        }
        // preserve the length of the scanner the following includes no gaps
        float scanner_length_cyl = new_scanner_sptr->get_num_rings()*
                new_scanner_sptr->get_ring_spacing();
        new_scanner_sptr->set_num_rings(new_num_rings);
        new_scanner_sptr->set_num_detectors_per_ring(new_num_dets);
        new_scanner_sptr->set_ring_spacing(static_cast<float>(scanner_length_cyl/new_scanner_sptr->get_num_rings()));
        
    }
    
    const float approx_num_non_arccorrected_bins =
      old_scanner_ptr->get_max_num_non_arccorrected_bins() * 
      (float(new_num_dets) / old_scanner_ptr->get_num_detectors_per_ring())
      + 5; // add 5 to avoid strange edge-effects, certainly with B-splines
    new_scanner_sptr->set_max_num_non_arccorrected_bins(round(approx_num_non_arccorrected_bins+.5F));
    new_scanner_sptr->set_default_bin_size(new_scanner_sptr->get_effective_ring_radius() * _PI / new_num_dets); // approx new detector size
    // Find how much is the delta ring
    // If the previous projdatainfo had max segment == 1 then should be from SSRB
    // in ScatterEstimation. Otherwise use the max possible.
    int delta_ring = proj_data_info_sptr->get_num_segments() == 1 ?  0 :
            new_scanner_sptr->get_num_rings()-1;

    new_scanner_sptr->set_up();
    shared_ptr<ProjDataInfo> templ_proj_data_info_sptr(
                                                      ProjDataInfo::ProjDataInfoCTI(new_scanner_sptr,
                                                                                    1, delta_ring,
                                                                                    new_scanner_sptr->get_num_detectors_per_ring()/2,
                                                                                    new_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                                                                    false));

    info(boost::format("ScatterSimulation: down-sampled scanner info:\n%1%")
	 % templ_proj_data_info_sptr->parameter_info(),
	 3);
    this->set_template_proj_data_info(*templ_proj_data_info_sptr);
    this->set_output_proj_data(this->output_proj_data_filename);

    return Succeeded::yes;
}

Succeeded ScatterSimulation::downsample_images_to_scanner_size()
{
    if(is_null_ptr(proj_data_info_sptr))
            return Succeeded::no;

    // Downsample the activity and attenuation images
    shared_ptr<VoxelsOnCartesianGrid<float> > tmpl_image( new VoxelsOnCartesianGrid<float>(*proj_data_info_sptr));

    if(!is_null_ptr(activity_image_sptr))
    {
        const VoxelsOnCartesianGrid<float>* tmp_act = dynamic_cast<const VoxelsOnCartesianGrid<float>* >(activity_image_sptr.get());
        VoxelsOnCartesianGrid<float>* tmp = tmpl_image->get_empty_copy();

	ZoomOptions scaling(ZoomOptions::preserve_projections);
        zoom_image(*tmp, *tmp_act, scaling);
        activity_image_sptr.reset(tmp);

        this->remove_cache_for_integrals_over_activity();
        this->_already_set_up = false;
    }

    if(!is_null_ptr(density_image_sptr))
    {
        const VoxelsOnCartesianGrid<float>* tmp_att = dynamic_cast<const VoxelsOnCartesianGrid<float>* >(density_image_sptr.get());
        VoxelsOnCartesianGrid<float>* tmp = tmpl_image->get_empty_copy();

	ZoomOptions scaling(ZoomOptions::preserve_values);
        zoom_image(*tmp, *tmp_att, scaling);
        density_image_sptr.reset(tmp);

        this->remove_cache_for_integrals_over_attenuation();
        this->_already_set_up = false;
    }

    // zooming of density_image_for_scatter_points_sptr will happen in set_up

    return Succeeded::yes;
}

void
ScatterSimulation::
set_attenuation_threshold(const float arg)
{
    attenuation_threshold = arg;
    this->_already_set_up = false;
}

void
ScatterSimulation::
set_randomly_place_scatter_points(const bool arg)
{
    randomly_place_scatter_points = arg;
    this->_already_set_up = false;
}

void
ScatterSimulation::
set_cache_enabled(const bool arg)
{
    use_cache = arg;
}

void
ScatterSimulation::
write_log(const double simulation_time, 
          const float total_scatter)
{
      if (this->output_proj_data_filename.empty())
	return;

       std::string log_filename =
                this->output_proj_data_filename + ".log";
        std::ofstream mystream(log_filename.c_str());

        if (!mystream)
        {
            warning("Cannot open log file '%s'", log_filename.c_str()) ;
            return;
        }

        int axial_bins = 0 ;

        for (int segment_num = this->output_proj_data_sptr->get_min_segment_num();
             segment_num <= this->output_proj_data_sptr->get_max_segment_num();
             ++segment_num)
            axial_bins += this->output_proj_data_sptr->get_num_axial_poss(segment_num);

        const int total_bins =
                this->output_proj_data_sptr->get_num_views() * axial_bins *
                this->output_proj_data_sptr->get_num_tangential_poss();
        mystream << this->parameter_info()
                 << "\nTotal simulation time elapsed: "
                 <<   simulation_time / 60 << "min"
		 << "\nTotal Scatter Points : " << scatt_points_vector.size()
		 << "\nTotal Scatter Counts (before upsampling and norm) : " << total_scatter
		 << "\nActivity image SIZE: "
		 << (*this->activity_image_sptr).size() << " * "
		 << (*this->activity_image_sptr)[0].size() << " * "  // TODO relies on 0 index
		 << (*this->activity_image_sptr)[0][0].size()
                << "\nAttenuation image for scatter points SIZE: "
                << (*this->density_image_for_scatter_points_sptr).size() << " * "
                << (*this->density_image_for_scatter_points_sptr)[0].size() << " * "
                << (*this->density_image_for_scatter_points_sptr)[0][0].size()
                << "\nTotal bins : " << total_bins << " = "
                << this->output_proj_data_sptr->get_num_views()
                << " view_bins * "
                << axial_bins << " axial_bins * "
                << this->output_proj_data_sptr->get_num_tangential_poss()
                << " tangential_bins\n";
}

END_NAMESPACE_STIR
