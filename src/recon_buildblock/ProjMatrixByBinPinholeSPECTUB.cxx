/*
    Copyright (C) 2022, Matthew Strugari
    Copyright (C) 2014, Biomedical Image Group (GIB), Universitat de Barcelona, Barcelona, Spain. All rights reserved.
    Copyright (C) 2014, 2021, 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
    \file
    \ingroup projection

    \brief Implementation of class stir::ProjMatrixByBinPinholeSPECTUB

    \author Matthew Strugari
    \author Carles Falcon
    \author Kris Thielemans
*/

// system libraries
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

using std::endl;
using std::nothrow;

// user defined libraries
//#include "stir/ProjDataInterfile.h"
#include "stir/recon_buildblock/ProjMatrixByBinPinholeSPECTUB.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
//#include "stir/KeyParser.h"
#include "stir/IO/read_from_file.h"
#include "stir/ProjDataInfo.h"
//#include "stir/utilities.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/Coordinate3D.h"
#include "stir/info.h"
#include "stir/format.h"
#include "stir/CPUTimer.h"
#ifdef STIR_OPENMP
#  include "stir/num_threads.h"
#endif

//#include "boost/cstdint.hpp"
//#include "boost/scoped_ptr.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

// using namespace std;
// using std::string;

//... user defined libraries .............................................................

#include "stir/recon_buildblock/PinholeSPECTUB_Weight3d.h"
#include "stir/recon_buildblock/PinholeSPECTUB_Tools.h"

//... functions from wm_SPECT.2.0............................

// void error_wm_SPECT_mph( int nerr, string txt);      //list of error messages
// void wm_inputs_mph( char ** argv, int argc );
// void read_inputs_mph( vector<string> param );

using namespace SPECTUB_mph;
using std::string;

START_NAMESPACE_STIR

const char* const ProjMatrixByBinPinholeSPECTUB::registered_name = "Pinhole SPECT UB";

ProjMatrixByBinPinholeSPECTUB::ProjMatrixByBinPinholeSPECTUB()
{
  set_defaults();
}

void
ProjMatrixByBinPinholeSPECTUB::initialise_keymap()
{
  parser.add_start_key("Projection Matrix By Bin Pinhole SPECT UB Parameters");
  ProjMatrixByBin::initialise_keymap();

  // no longer parse this
  // parser.add_key("minimum weight", &minimum_weight);
  parser.add_key("maximum number of sigmas", &maximum_number_of_sigmas);
  parser.add_key("spatial resolution PSF", &spatial_resolution_PSF);
  parser.add_key("subsampling factor PSF", &subsampling_factor_PSF);
  parser.add_key("detector file", &detector_file);
  parser.add_key("collimator file", &collimator_file);
  parser.add_key("psf correction", &psf_correction);
  parser.add_key("doi correction", &doi_correction);
  parser.add_key("attenuation type", &attenuation_type);
  parser.add_key("attenuation map", &attenuation_map);
  parser.add_key("object radius (cm)", &object_radius);
  // no longer parse this
  // parser.add_key("mask type", &mask_type);
  parser.add_key("mask file", &mask_file);
  parser.add_key("mask from attenuation map", &mask_from_attenuation_map);
  parser.add_key("keep all views in cache", &keep_all_views_in_cache);

  parser.add_stop_key("End Projection Matrix By Bin Pinhole SPECT UB Parameters");
}

void
ProjMatrixByBinPinholeSPECTUB::set_defaults()
{
  ProjMatrixByBin::set_defaults();

  this->already_setup = false;

  this->keep_all_views_in_cache = false;
  minimum_weight = 0.0;
  maximum_number_of_sigmas = 2.;
  spatial_resolution_PSF = 0.001;
  subsampling_factor_PSF = 1;
  detector_file = "";
  collimator_file = "";
  psf_correction = "no";
  doi_correction = "no";
  attenuation_type = "no";
  attenuation_map = "";
  object_radius = 0.0;
  // mask_type= "no";
  mask_file = "";
  mask_from_attenuation_map = false;
}

bool
ProjMatrixByBinPinholeSPECTUB::post_processing()
{
  if (ProjMatrixByBin::post_processing() == true)
    return true;

  this->set_attenuation_type(this->attenuation_type);

  if (!this->attenuation_map.empty())
    this->set_attenuation_image_sptr(this->attenuation_map);
  else
    this->attenuation_image_sptr.reset();

  if (!this->mask_file.empty())
    this->set_mask_image_sptr(this->mask_file);
  else
    this->mask_image_sptr.reset();

  this->already_setup = false;

  return false;
}

//******************** get/set pairs *************
/*
//! Minimum weight
float
ProjMatrixByBinPinholeSPECTUB::
get_minimum_weight() const
{
    return this->minimum_weight;
}

void
ProjMatrixByBinPinholeSPECTUB::
set_minimum_weight( const float value )
{
    if (this->minimum_weight != value)
    {
        this->minimum_weight = value;
        this->already_setup = false;
    }
}
*/

//! Maximum number of sigmas
float
ProjMatrixByBinPinholeSPECTUB::get_maximum_number_of_sigmas() const
{
  return this->maximum_number_of_sigmas;
}

void
ProjMatrixByBinPinholeSPECTUB::set_maximum_number_of_sigmas(const float value)
{
  if (this->maximum_number_of_sigmas != value)
    {
      this->maximum_number_of_sigmas = value;
      this->already_setup = false;
    }
}

//! Spatial resolution PSF
float
ProjMatrixByBinPinholeSPECTUB::get_spatial_resolution_PSF() const
{
  return this->spatial_resolution_PSF;
}

void
ProjMatrixByBinPinholeSPECTUB::set_spatial_resolution_PSF(const float value)
{
  if (this->spatial_resolution_PSF != value)
    {
      this->spatial_resolution_PSF = value;
      this->already_setup = false;
    }
}

//! Subsampling factor PSF
int
ProjMatrixByBinPinholeSPECTUB::get_subsampling_factor_PSF() const
{
  return this->subsampling_factor_PSF;
}

void
ProjMatrixByBinPinholeSPECTUB::set_subsampling_factor_PSF(const int value)
{
  if (this->subsampling_factor_PSF != value)
    {
      this->subsampling_factor_PSF = value;
      this->already_setup = false;
    }
}

//! Detector file
/*
string
ProjMatrixByBinPinholeSPECTUB::
get_detector_file() const
{
    return this->detector_file;
}
*/

void
ProjMatrixByBinPinholeSPECTUB::set_detector_file(const string& value)
{
  if (this->detector_file != value)
    {
      this->detector_file = value;
      this->already_setup = false;
    }
}

//! Collimator file
/*
string
ProjMatrixByBinPinholeSPECTUB::
get_collimator_file() const
{
    return this->collimator_file;
}
*/

void
ProjMatrixByBinPinholeSPECTUB::set_collimator_file(const string& value)
{
  if (this->collimator_file != value)
    {
      this->collimator_file = value;
      this->already_setup = false;
    }
}

//! PSF correction
string
ProjMatrixByBinPinholeSPECTUB::get_psf_correction() const
{
  return this->psf_correction;
}

void
ProjMatrixByBinPinholeSPECTUB::set_psf_correction(const string& value)
{
  if (this->psf_correction != boost::algorithm::to_lower_copy(value))
    {
      this->psf_correction = boost::algorithm::to_lower_copy(value);
      if (this->psf_correction != "yes" && this->psf_correction != "no")
        error("psf_correction has to be Yes or No");
      this->already_setup = false;
    }
}

//! Set DOI correction
string
ProjMatrixByBinPinholeSPECTUB::get_doi_correction() const
{
  return this->doi_correction;
}
void
ProjMatrixByBinPinholeSPECTUB::set_doi_correction(const string& value)
{
  if (this->doi_correction != boost::algorithm::to_lower_copy(value))
    {
      this->doi_correction = boost::algorithm::to_lower_copy(value);
      if (this->doi_correction != "yes" && this->doi_correction != "no")
        error("doi_correction has to be Yes or No");
      this->already_setup = false;
    }
}

//! Attenuation image
string
ProjMatrixByBinPinholeSPECTUB::get_attenuation_type() const
{
  return this->attenuation_type;
}

void
ProjMatrixByBinPinholeSPECTUB::set_attenuation_type(const string& value)
{
  if (this->attenuation_type != boost::algorithm::to_lower_copy(value))
    {
      this->attenuation_type = boost::algorithm::to_lower_copy(value);
      if (this->attenuation_type != "simple" && this->attenuation_type != "full" && this->attenuation_type != "no")
        error("attenuation_type has to be Simple, Full, or No");
      this->already_setup = false;
    }
}

shared_ptr<const DiscretisedDensity<3, float>>
ProjMatrixByBinPinholeSPECTUB::get_attenuation_image_sptr() const
{
  return this->attenuation_image_sptr;
}

void
ProjMatrixByBinPinholeSPECTUB::set_attenuation_image_sptr(const shared_ptr<const DiscretisedDensity<3, float>> value)
{
  this->attenuation_image_sptr = value;
  if (this->attenuation_type == "no")
    {
      info("Setting attenuation type to 'simple'.");
      this->set_attenuation_type("simple");
    }
  this->already_setup = false;
}

void
ProjMatrixByBinPinholeSPECTUB::set_attenuation_image_sptr(const string& value)
{
  this->attenuation_map = value;
  shared_ptr<const DiscretisedDensity<3, float>> im_sptr(read_from_file<DiscretisedDensity<3, float>>(this->attenuation_map));
  set_attenuation_image_sptr(im_sptr);
}

//! Object radius (cm)
float
ProjMatrixByBinPinholeSPECTUB::get_object_radius() const
{
  return this->object_radius;
}

void
ProjMatrixByBinPinholeSPECTUB::set_object_radius(const float value)
{
  if (this->object_radius != value)
    {
      this->object_radius = value;
      this->already_setup = false;
    }
}

//! Mask image
bool
ProjMatrixByBinPinholeSPECTUB::get_mask_from_attenuation_map() const
{
  return this->mask_from_attenuation_map;
}

void
ProjMatrixByBinPinholeSPECTUB::set_mask_from_attenuation_map(bool value)
{
  if (this->mask_from_attenuation_map != value)
    {
      this->mask_from_attenuation_map = value;
      this->already_setup = false;
    }
}

shared_ptr<const DiscretisedDensity<3, float>>
ProjMatrixByBinPinholeSPECTUB::get_mask_image_sptr() const
{
  return this->mask_image_sptr;
}

void
ProjMatrixByBinPinholeSPECTUB::set_mask_image_sptr(const shared_ptr<const DiscretisedDensity<3, float>> value)
{
  this->mask_image_sptr = value;
  if (this->mask_from_attenuation_map == true)
    {
      info("Setting mask from attenuation map to '0'");
      this->set_mask_from_attenuation_map(false);
    }
  this->already_setup = false;
}

void
ProjMatrixByBinPinholeSPECTUB::set_mask_image_sptr(const string& value)
{
  this->mask_file = value;
  shared_ptr<const DiscretisedDensity<3, float>> im_sptr(read_from_file<DiscretisedDensity<3, float>>(this->mask_file));
  set_mask_image_sptr(im_sptr);
}

//! Keep all views in cache
bool
ProjMatrixByBinPinholeSPECTUB::get_keep_all_views_in_cache() const
{
  return this->keep_all_views_in_cache;
}

void
ProjMatrixByBinPinholeSPECTUB::set_keep_all_views_in_cache(bool value)
{
  if (this->keep_all_views_in_cache != value)
    {
      this->keep_all_views_in_cache = value;
      this->already_setup = false;
    }
}

//******************** actual implementation *************

void
ProjMatrixByBinPinholeSPECTUB::set_up(
    const shared_ptr<const ProjDataInfo>& proj_data_info_ptr_v,
    const shared_ptr<const DiscretisedDensity<3, float>>& density_info_ptr // TODO should be Info only
)
{

  ProjMatrixByBin::set_up(proj_data_info_ptr_v, density_info_ptr);

#ifdef STIR_OPENMP
  if (!this->keep_all_views_in_cache)
    {
      warning("Pinhole SPECTUB matrix can currently only use single-threaded code unless all views are kept. Setting num_threads "
              "to 1.");
      set_num_threads(1);
    }
#endif

  std::stringstream info_stream;

  const VoxelsOnCartesianGrid<float>* image_info_ptr = dynamic_cast<const VoxelsOnCartesianGrid<float>*>(density_info_ptr.get());

  if (image_info_ptr == nullptr)
    error("ProjMatrixByBinPinholeSPECTUB set-up with a wrong type of DiscretisedDensity\n");

  if (this->already_setup)
    {
      if (this->densel_range == image_info_ptr->get_index_range() && this->voxel_size == image_info_ptr->get_voxel_size()
          && this->origin == image_info_ptr->get_origin() && *proj_data_info_ptr_v == *this->proj_data_info_ptr)
        {
          // stored matrix should be compatible, so we can just reuse it
          return;
        }
      else
        {
          this->clear_cache();
          this->delete_PinholeSPECTUB_arrays();
        }
    }

  this->proj_data_info_ptr = proj_data_info_ptr_v;
  symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data_info_ptr_v));

  this->densel_range = image_info_ptr->get_index_range();
  this->voxel_size = image_info_ptr->get_voxel_size();
  this->origin = image_info_ptr->get_origin();

  const ProjDataInfoCylindricalArcCorr* proj_Data_Info_Cylindrical
      = dynamic_cast<const ProjDataInfoCylindricalArcCorr*>(this->proj_data_info_ptr.get());

  CPUTimer timer;
  timer.start();

  //... *** code below replaces wm_inputs_mph() and read_inputs_mph()

  //.....image parameters......................

  vol.Dimx = image_info_ptr->get_x_size();               // Image: number of columns
  vol.Dimy = image_info_ptr->get_y_size();               // Image: number of rows
  vol.Dimz = image_info_ptr->get_z_size();               // Image: and projections: number of slices
  vol.szcm = image_info_ptr->get_voxel_size().x() / 10.; // Image: voxel size (cm)
  vol.thcm = image_info_ptr->get_voxel_size().z() / 10.; // Image: slice thickness (cm)

  vol.first_sl = 0;       // Image: first slice to take into account (no weight below)
  vol.last_sl = vol.Dimz; // Image: last slice to take into account (no weights above)

  // if ( wmh.vol.first_sl < 0 || wmh.vol.first_sl > wmh.vol.Dimz ) error_wm_SPECT_mph( 107, param[ 7 ] );
  // if ( wmh.vol.last_sl <= wmh.vol.first_sl || wmh.vol.last_sl > wmh.vol.Dimz ) error_wm_SPECT_mph( 108, param[ 8 ] );

  wmh.ro = object_radius; // Image: object radius (cm)

  //..... geometrical and other derived parameters of the volume structure...............

  vol.Npix = vol.Dimx * vol.Dimy;
  vol.Nvox = vol.Npix * vol.Dimz;

  vol.FOVxcmd2 = (float)vol.Dimx * vol.szcm / (float)2.; // half of the size of the image volume, dimension x (cm);
  vol.FOVcmyd2 = (float)vol.Dimy * vol.szcm / (float)2.; // half of the size of the image volume, dimension y (cm);
  vol.FOVzcmd2 = (float)vol.Dimz * vol.thcm / (float)2.; // Half of the size of the image volume, dimension z (cm);

  vol.x0 = -vol.FOVxcmd2 + (float)0.5 * vol.szcm; // x coordinate of first voxel
  vol.y0 = -vol.FOVcmyd2 + (float)0.5 * vol.szcm; // y coordinate of first voxel
  vol.z0 = -vol.FOVzcmd2 + (float)0.5 * vol.thcm; // z coordinate of first voxel

  wmh.vol = vol;

  //...ring parameters ................................................

  wmh.detector_fn = detector_file;

  //....collimator parameters ........................................

  wmh.collim_fn = collimator_file;

  //... resolution parameters ..............................................

  wmh.mn_w = minimum_weight;
  wmh.Nsigm = maximum_number_of_sigmas;
  wmh.highres = spatial_resolution_PSF;
  wmh.subsamp = subsampling_factor_PSF;

  wmh.do_subsamp = false;

  //...correction for intrinsic PSF....................................
  boost::algorithm::to_lower(psf_correction);
  if (psf_correction == "no")
    wmh.do_psfi = false;
  else
    {
      if (psf_correction == "yes")
        wmh.do_psfi = true;
      else
        error("psf_correction has to be Yes or No"); // error_wm_SPECT_mph( 116, psf_correction );
      wmh.do_subsamp = true;
    }

  //... impact depth .........................
  boost::algorithm::to_lower(doi_correction);
  if (doi_correction == "no")
    {
      wmh.do_depth = false;
    }
  else
    {
      if (doi_correction == "yes")
        wmh.do_depth = true;
      else
        error("doi_correction has to be Yes or No"); // error_wm_SPECT_mph( 117, doi_correction );
      wmh.do_subsamp = true;
    }

  //... attenuation parameters .........................
  boost::algorithm::to_lower(attenuation_type);
  if (attenuation_type == "no")
    {
      wmh.do_att = wmh.do_full_att = false;
    }
  else
    {
      wmh.do_att = true;
      if (attenuation_type == "simple")
        wmh.do_full_att = false;
      else
        {
          if (attenuation_type == "full")
            wmh.do_full_att = true;
          else
            error("attenuation_type has to be Simple, Full, or No"); // error_wm_SPECT_mph( 118, attenuation_type );
        }

      wmh.att_fn = attenuation_map;
    }

  //... masking parameters.............................
  wmh.do_msk_att = mask_from_attenuation_map;

  // no longer use mask type
  /*
  boost::algorithm::to_lower(mask_type);
  if( mask_type == "no" ) wmh.do_msk_att = wmh.do_msk_file = false;
  else {
      if( mask_type == "attenuation map" ) wmh.do_msk_att = true;
      else {
          if( mask_type == "explicit mask" ){
              wmh.do_msk_file = true;

              wmh.msk_fn = mask_file;
          }
          else error("mask_type has to be Attenuation Map, Explicit Mask, or No");    //error_wm_SPECT_mph( 120, mask_type);
      }
  }
  */

  //... initialization of do_variables to false..............

  wmh.do_round_cumsum = wmh.do_square_cumsum = false;

  //... projection parameters ...................

  wmh.prj.rad = proj_Data_Info_Cylindrical->get_ring_radius() / (float)10.; // ring radius (cm)
  wmh.prj.Nbin = proj_Data_Info_Cylindrical->get_num_tangential_poss();     // number of bins per row
  wmh.prj.Nsli = proj_Data_Info_Cylindrical->get_num_axial_poss(0);         // number of slices

  wmh.prj.szcm = proj_Data_Info_Cylindrical->get_tangential_sampling() / (float)10.; // bin size (cm)
  wmh.prj.thcm = proj_Data_Info_Cylindrical->get_axial_sampling(0) / (float)10.;     // slice thickness (cm)

  //... derived variables .......................

  wmh.prj.FOVxcmd2 = (float)wmh.prj.Nbin * wmh.prj.szcm / (float)2.; // FOVcmx divided by 2
  wmh.prj.FOVzcmd2 = (float)wmh.prj.Nsli * wmh.prj.thcm / (float)2.; // FOVcmz divided by 2

  wmh.prj.Nbd = wmh.prj.Nsli * wmh.prj.Nbin;

  wmh.prj.szcmd2 = wmh.prj.szcm / (float)2.;
  wmh.prj.thcmd2 = wmh.prj.thcm / (float)2.;

  //... files with complementary information .................

  read_prj_params_mph(wmh);
  read_coll_params_mph(wmh);

  //... precalculated functions ................

  fill_pcf(wmh, pcf);

  //... other variables .........................

  wm.Nbt = wmh.prj.Nbt;                                               // number of rows of the weight matrix
  wm.Nvox = wmh.vol.Nvox;                                             // number of columns of the weight matrix
  wmh.mndvh2 = (wmh.collim.rad - wmh.ro) * (wmh.collim.rad - wmh.ro); // reference distance ^2 for efficiency

  // variables for wm calculations by view ("UB-subset")
  wmh.prj.NOS = this->proj_data_info_ptr->get_num_views();
  wmh.prj.NdOS = wmh.prj.Ndt / wmh.prj.NOS;
  wmh.prj.NbOS = wmh.prj.Nbt / wmh.prj.NOS;

  wm.do_save_STIR = true;

  //... control of read parameters ..............
  info_stream << "Parameters of Pinhole SPECT UB matrix: (in cm)" << endl;
  info_stream << "Image. Nrow: " << wmh.vol.Dimy << "\tNcol: " << wmh.vol.Dimx << "\tvoxel_size: " << wmh.vol.szcm << endl;
  info_stream << "Number of slices: " << wmh.vol.Dimz << "\tslice_thickness: " << wmh.vol.thcm << endl;
  info_stream << "FOVxcmd2: " << wmh.vol.FOVxcmd2 << "\tFOVcmyd2: " << wmh.vol.FOVcmyd2 << "\tradius object: " << wmh.ro << endl;
  info_stream << "Minimum weight: " << wmh.mn_w << endl;

  info(info_stream.str());

  //... up to here replaces wm_inputs_mph() and read_inputs_mph()

  //... to read attenuation map ..................................................

  if (wmh.do_att)
    {
      if (is_null_ptr(attenuation_image_sptr))
        error("Attenuation image not set.");
      std::string explanation;
      if (!density_info_ptr->has_same_characteristics(*attenuation_image_sptr, explanation))
        error("Currently the attenuation map and emission image must have the same dimension, orientation, and voxel size:\n"
              + explanation);

      if ((attmap = new (nothrow) float[wmh.vol.Nvox]) == nullptr)
        error("Error allocating space to store values for attenuation map.");

      bool exist_nan = false;

      std::copy(attenuation_image_sptr->begin_all(), attenuation_image_sptr->end_all(), attmap); // read_att_map_mph( attmap );

      for (int i = 0; i < wmh.vol.Nvox; i++)
        {
          if ((boost::math::isnan)(attmap[i]))
            {
              attmap[i] = 0;
              exist_nan = true;
            }
          if (exist_nan)
            warning("attmap contains NaN values. Converted to zero.");
        }
    }
  else
    attmap = nullptr;

  //... to generate mask..........................................................

  msk_3d = new bool[wmh.vol.Nvox];
  // generate_msk_mph( msk_3d, attmap, wmh );

  // generate mask from mask file
  if (!is_null_ptr(mask_image_sptr))
    {
      if (!density_info_ptr->has_same_characteristics(*mask_image_sptr))
        error("Currently the mask image and emission image must have the same dimension, orientation, and voxel size.");

      float* mask_from_file;
      if ((mask_from_file = new (nothrow) float[wmh.vol.Nvox]) == nullptr)
        error("Error allocating space to store values for mask from file.");

      std::copy(mask_image_sptr->begin_all(), mask_image_sptr->end_all(), mask_from_file);

      // call UB generate_msk_mph pretending that this mask is an attenuation image
      // we do this to avoid using its own read_msk_file_mph
      wmh.do_msk_file = false;
      wmh.do_msk_att = true;
      generate_msk_mph(msk_3d, mask_from_file, wmh);

      delete[] mask_from_file;
    }
  // generate mask from attenuation map
  else if (wmh.do_msk_att)
    {
      if (is_null_ptr(attmap))
        error("No attenuation image set, so cannot compute the mask image from it.");
      generate_msk_mph(msk_3d, attmap, wmh);
    }
  // generate mask from object radius
  else
    {
      wmh.do_msk_file = false;
      generate_msk_mph(msk_3d, (float*)0, wmh);
    }

  //... initialize psf2d in bins ..................................................

  wmh.max_amp = (wmh.prj.rad - wmh.ro) / (wmh.collim.rad - wmh.ro);

  psf_bin.max_dimx = (int)std::floor(wmh.max_hsxcm * wmh.max_amp / wmh.prj.szcm) + 2;
  psf_bin.max_dimz = (int)std::floor(wmh.max_hszcm * wmh.max_amp / wmh.prj.thcm) + 2;

  //... distributions at mid resolution ...........................................

  if (wmh.do_subsamp)
    {

      psf_subs.max_dimx = psf_bin.max_dimx * wmh.subsamp;
      psf_subs.max_dimz = psf_bin.max_dimz * wmh.subsamp;

      if (wmh.do_depth)
        {
          psf_subs.max_dimx += (1 + (int)std::ceil(wmh.prj.crth * wmh.tmax_aix / wmh.prj.szcm)) * wmh.subsamp;
          psf_subs.max_dimz += (1 + (int)std::ceil(wmh.prj.crth * wmh.tmax_aiz / wmh.prj.thcm)) * wmh.subsamp;
        }

      if (wmh.do_psfi)
        {

          int dimx = (int)std::ceil(0.5 * wmh.prj.sgm_i * wmh.Nsigm / wmh.prj.szcm);
          int dimz = (int)std::ceil(0.5 * wmh.prj.sgm_i * wmh.Nsigm / wmh.prj.thcm);

          kern.dimx = kern.max_dimx = 2 * wmh.subsamp * dimx + 1;
          kern.dimz = kern.max_dimz = 2 * wmh.subsamp * dimz + 1;
          kern.ib0 = -dimx;
          kern.jb0 = -dimz;
          kern.lngxcmd2 = kern.lngzcmd2 = wmh.prj.sgm_i * wmh.Nsigm / (float)2.;

          kern.val = new float*[kern.max_dimz];

          for (int i = 0; i < kern.max_dimz; i++)
            kern.val[i] = new float[kern.max_dimx];

          fill_psfi(&kern, wmh);

          psf_subs.max_dimx += kern.max_dimx - 1;
          psf_subs.max_dimz += kern.max_dimz - 1;

          psf_aux.max_dimx = psf_aux.dimx = psf_subs.max_dimx;
          psf_aux.max_dimz = psf_aux.dimz = psf_subs.max_dimz;

          psf_aux.val = new float*[psf_aux.max_dimz];

          for (int i = 0; i < psf_aux.max_dimz; i++)
            psf_aux.val[i] = new float[psf_aux.max_dimx];
        }

      psf_subs.val = new float*[psf_subs.max_dimz];

      for (int i = 0; i < psf_subs.max_dimz; i++)
        psf_subs.val[i] = new float[psf_subs.max_dimx];

      psf_bin.max_dimx = psf_subs.max_dimx / wmh.subsamp + 2;
      psf_bin.max_dimz = psf_subs.max_dimz / wmh.subsamp + 2;
    }

  psf_bin.val = new float*[psf_bin.max_dimz];

  for (int i = 0; i < psf_bin.max_dimz; i++)
    psf_bin.val[i] = new float[psf_bin.max_dimx];

  //... size estimation .........................................................

  // number of non-zero elements for each weight matrix row
  Nitems = new int*[wmh.prj.NOS];
  for (int kOS = 0; kOS < wmh.prj.NOS; kOS++)
    {
      Nitems[kOS] = new int[wmh.prj.NbOS];
      for (int i = 0; i < wmh.prj.NbOS; i++)
        Nitems[kOS][i] = 1; // Nitems initializated to one
    }

  //... double array wm.val and wm.col .....................................................

  if ((wm.val = new (nothrow) float*[wmh.prj.NbOS]) == nullptr)
    error(
        "Error allocating space to store values for SPECTUB matrix"); // error_wmtools_SPECT_mph( 200, wmh.prj.NbOS, "wm.val[]" );

  if ((wm.col = new (nothrow) int*[wmh.prj.NbOS]) == nullptr)
    error("Error allocating space to store column indices for SPECTUB matrix"); // error_wmtools_SPECT_mph( 200, wmh.prj.NbOS,
                                                                                // "wm.col[]" );

  //... array wm.ne .........................................................................

  if ((wm.ne = new (nothrow) int[wmh.prj.NbOS + 1]) == 0)
    error("Error allocating space to store number of elements for SPECTUB matrix"); // error_wmtools_SPECT_mph(200, wmh.prj.NbOS +
                                                                                    // 1, "wm.ne[]");

  // allocate memory for weight matrix
  if (wm.do_save_STIR)
    {
      wm.ns = new int[wmh.prj.NbOS];
      wm.nb = new int[wmh.prj.NbOS];
      wm.na = new int[wmh.prj.NbOS];

      wm.nx = new short int[wmh.vol.Nvox];
      wm.ny = new short int[wmh.vol.Nvox];
      wm.nz = new short int[wmh.vol.Nvox];
    }

  // size estimation
  for (int kOS = 0; kOS < wmh.prj.NOS; kOS++)
    {
      wm_calculation_mph(false, kOS, &psf_bin, &psf_subs, &psf_aux, &kern, attmap, msk_3d, Nitems[kOS], wmh, wm, pcf);
    }
  info(format("Done estimating size of matrix. Execution time, CPU {} s", timer.value()), 2);

  this->already_setup = true;
}

ProjMatrixByBinPinholeSPECTUB*
ProjMatrixByBinPinholeSPECTUB::clone() const
{
  // we deleted the copy constructor as it's not safe with all those bare pointers, so cannot do this
  // return new ProjMatrixByBinPinholeSPECTUB(*this);
  error("ProjMatrixByBinPinholeSPECTUB::clone not implemented yet");
  return 0;
}

ProjMatrixByBinPinholeSPECTUB::~ProjMatrixByBinPinholeSPECTUB()
{
  delete_PinholeSPECTUB_arrays();
}

void
ProjMatrixByBinPinholeSPECTUB::delete_PinholeSPECTUB_arrays()
{
  if (!this->already_setup)
    return;

  //... freeing matrix memory....................................

  delete[] wm.val;
  delete[] wm.col;
  delete[] wm.ne;

  if (wm.do_save_STIR)
    {
      delete[] wm.ns;
      delete[] wm.nb;
      delete[] wm.na;
      delete[] wm.nx;
      delete[] wm.ny;
      delete[] wm.nz;
    }

  //... freeing pre-calculated functions ....................................

  if (wmh.do_round_cumsum)
    {
      for (int i = 0; i < pcf.round.dim; i++)
        delete[] pcf.round.val[i];
      delete[] pcf.round.val;
    }

  if (wmh.do_square_cumsum)
    {
      for (int i = 0; i < pcf.square.dim; i++)
        delete[] pcf.square.val[i];
      delete[] pcf.square.val;
    }

  if (wmh.do_depth)
    delete pcf.cr_att.val;

  //... freeing memory ....................................

  for (int i = 0; i < psf_bin.max_dimz; i++)
    delete[] psf_bin.val[i];
  delete[] psf_bin.val;

  if (wmh.do_subsamp)
    {
      for (int i = 0; i < psf_subs.max_dimz; i++)
        delete[] psf_subs.val[i];
      delete[] psf_subs.val;
    }

  if (wmh.do_psfi)
    {
      for (int i = 0; i < kern.max_dimz; i++)
        delete[] kern.val[i];
      delete[] kern.val;

      // original code did not deallocate psf_aux.val, possible memory leak
      for (int i = 0; i < psf_aux.max_dimz; i++)
        delete[] psf_aux.val[i];
      delete[] psf_aux.val;
    }

  for (int kOS = 0; kOS < wmh.prj.NOS; kOS++)
    delete[] Nitems[kOS];
  delete[] Nitems;

  if (wmh.do_att)
    delete[] attmap;

  delete[] msk_3d;
}

void
ProjMatrixByBinPinholeSPECTUB::compute_one_subset(const int kOS) const
{
  CPUTimer timer;
  timer.start();

  //... size information ..........................................................................

  unsigned int ne = 0;

  for (int i = 0; i < wmh.prj.NbOS; i++)
    ne += Nitems[kOS][i];

  info(format("Total number of non-zero weights in this view: {}, estimated size: {} MB",
              ne,
              (wm.do_save_STIR ? (ne + 10 * wmh.prj.NbOS) / 104857.6 : ne / 131072)),
       2);

  //... memory allocation for wm float arrays ....................................................

  wm_alloc(Nitems[kOS], wm, wmh);

  //... wm calculation ...............................................................................

  wm_calculation_mph(true, kOS, &psf_bin, &psf_subs, &psf_aux, &kern, attmap, msk_3d, Nitems[kOS], wmh, wm, pcf);
  info(format("Weight matrix calculation done, CPU {} s", timer.value()), 2);

  //... fill lor ..........................
  for (int j = 0; j < wmh.prj.NbOS; j++)
    {
      ProjMatrixElemsForOneBin lor;
      Bin bin;
      bin.segment_num() = 0;
      bin.view_num() = wm.na[j];
      bin.axial_pos_num() = wm.ns[j];
      bin.tangential_pos_num() = wm.nb[j];
      bin.set_bin_value(0);
      lor.set_bin(bin);

      lor.reserve(wm.ne[j]);
      for (int i = 0; i < wm.ne[j]; i++)
        {

          const ProjMatrixElemsForOneBin::value_type elem(
              Coordinate3D<int>(wm.nz[wm.col[j][i]], wm.ny[wm.col[j][i]], wm.nx[wm.col[j][i]]), wm.val[j][i]);
          lor.push_back(elem);
        }

      delete[] wm.val[j];
      delete[] wm.col[j];

      this->cache_proj_matrix_elems_for_one_bin(lor);
    }

  info(format("Total time after transfering to ProjMatrixElemsForOneBin, CPU {} s", timer.value()), 2);
}

void
ProjMatrixByBinPinholeSPECTUB::calculate_proj_matrix_elems_for_one_bin(ProjMatrixElemsForOneBin& lor) const
{
  const int view_num = lor.get_bin().view_num();

#ifdef STIR_OPENMP
#  pragma omp critical(PROJMATRIXBYBINUBONEVIEW)
#endif

  if (!this->keep_all_views_in_cache)
    this->clear_cache();

  info(format("Computing matrix elements for view {}", view_num), 2);
  compute_one_subset(view_num);

  lor.erase();
}

END_NAMESPACE_STIR
