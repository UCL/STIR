/*
  Copyright (C) 2013-2018, 2020 University College London
  Copyright (C) 2017-2019 University of Leeds

  This file contains is based on information supplied by Siemens but
  is distributed with their consent.

  This file is free software; you can redistribute that part and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation; either version 2.1 of the License, or
  (at your option) any later version.

  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.

  See STIR/LICENSE.txt for details

*/
/*!
  \file
  \ingroup recon_buildblock
  \ingroup GE

  \brief Implementation for class stir::GE:RDF_HDF5::BinNormalisationFromGEHDF5

  This file is largely a copy of the ECAT7 version, but with important changes for GEHDF5.
  \todo remove duplication

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Palak Wadhwa
*/


#include "stir/recon_buildblock/BinNormalisationFromGEHDF5.h"
#include "stir/DetectionPosition.h"
#include "stir/DetectionPositionPair.h"
#include "stir/shared_ptr.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange.h"
#include "stir/Bin.h"
#include "stir/ProjDataInMemory.h"
#include "stir/display.h"
#include "stir/IO/read_data.h"
#include "stir/IO/InterfileHeader.h"
#include "stir/ByteOrder.h"
#include "stir/is_null_ptr.h"
#include <algorithm>
#include <fstream>
#include <cctype>
#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
using std::ios;
#endif

START_NAMESPACE_STIR
namespace GE {
namespace RDF_HDF5 {

	   

const char * const 
BinNormalisationFromGEHDF5::registered_name = "From GE HDF5"; 


namespace detail
{

//
// helper functions used in this class.
//


static 
int
calc_ring1_plus_ring2(const Bin& bin, 
                      const ProjDataInfoCylindricalNoArcCorr *proj_data_cyl) {

  int segment_num = bin.segment_num();
 
  const int min_ring_diff = proj_data_cyl->get_min_ring_difference(segment_num);
  const int max_ring_diff = proj_data_cyl->get_max_ring_difference(segment_num);

  const int num_rings = proj_data_cyl->get_scanner_ptr()->get_num_rings();

  return( (2 * bin.axial_pos_num() - 
           (proj_data_cyl->get_min_axial_pos_num(segment_num) + 
            proj_data_cyl->get_max_axial_pos_num(segment_num))
           ) / (min_ring_diff != max_ring_diff ? 2 : 1) 
          + num_rings - 1 );
  
}



static
void
set_detection_tangential_coords(shared_ptr<ProjDataInfoCylindricalNoArcCorr> proj_data_cyl_uncomp,
                                const Bin& uncomp_bin, 
                                DetectionPositionPair<>& detection_position_pair) {
  int det1_num=0;
  int det2_num=0;
  
  proj_data_cyl_uncomp->get_det_num_pair_for_view_tangential_pos_num(det1_num, det2_num,
                                                                     uncomp_bin.view_num(),
                                                                     uncomp_bin.tangential_pos_num());
  detection_position_pair.pos1().tangential_coord() = det1_num;
  detection_position_pair.pos2().tangential_coord() = det2_num;
  
}



// Returns the sum of the two axial coordinates. Or -1 if the ring positions are
// out of range.
// sets axial_coord of detection_position_pair
static
int
set_detection_axial_coords(const ProjDataInfoCylindricalNoArcCorr *proj_data_info_cyl,
                           int ring1_plus_ring2, const Bin& uncomp_bin,
                           DetectionPositionPair<>& detection_position_pair) {
  
  const int num_rings = proj_data_info_cyl->get_scanner_ptr()->get_num_rings();

  const int ring_diff = uncomp_bin.segment_num();

  const int ring1 = (ring1_plus_ring2 - ring_diff)/2;
  const int ring2 = (ring1_plus_ring2 + ring_diff)/2;
  
  if (ring1<0 || ring2 < 0 || ring1>=num_rings || ring2 >= num_rings) {
    return(-1);
  }
        
  assert((ring1_plus_ring2 + ring_diff)%2 == 0);
  assert((ring1_plus_ring2 - ring_diff)%2 == 0);
  
  detection_position_pair.pos1().axial_coord() = ring1;
  detection_position_pair.pos2().axial_coord() = ring2;

  
  return(ring1 + ring2);
}



} // end of namespace detail




//
// Member functions
//



void 
BinNormalisationFromGEHDF5::set_defaults()
{
  this->normalisation_GEHDF5_filename = "";
  //this->_use_gaps = false;
  this->_use_detector_efficiencies = true;
  this->_use_dead_time = false;
  this->_use_geometric_factors = true;
}

void 
BinNormalisationFromGEHDF5::
initialise_keymap()
{
  this->parser.add_start_key("Bin Normalisation From GE HDF5");
  this->parser.add_key("normalisation_filename", &this->normalisation_GEHDF5_filename);
  //this->parser.add_parsing_key("singles rates", &this->singles_rates_ptr);
  //this->parser.add_key("use_gaps", &this->_use_gaps);
  this->parser.add_key("use_detector_efficiencies", &this->_use_detector_efficiencies);
  //this->parser.add_key("use_dead_time", &this->_use_dead_time);
  this->parser.add_key("use_geometric_factors", &this->_use_geometric_factors);
  this->parser.add_stop_key("End Bin Normalisation From GE HDF5");
}

bool 
BinNormalisationFromGEHDF5::
post_processing()
{
  read_norm_data(normalisation_GEHDF5_filename);
  return false;
}


BinNormalisationFromGEHDF5::
BinNormalisationFromGEHDF5()
{
  set_defaults();
}

BinNormalisationFromGEHDF5::
BinNormalisationFromGEHDF5(const string& filename)
{
  read_norm_data(filename);
}

Succeeded
BinNormalisationFromGEHDF5::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v)
{
  BinNormalisation::set_up(proj_data_info_ptr_v);
  proj_data_info_ptr = proj_data_info_ptr_v;
  proj_data_info_cyl_ptr =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr *>(proj_data_info_ptr.get());
  if (proj_data_info_cyl_ptr==0)
  {
    warning("BinNormalisationFromGEHDF5 can only be used on non-arccorrected data\n");
    return Succeeded::no;
  }
  if (*proj_data_info_ptr->get_scanner_ptr()  != *scanner_ptr)
  {
    warning("BinNormalisationFromGEHDF5: scanner object from proj data is different from the one "
      "from the normalisation file\n");
    return Succeeded::no;
  }

  mash = scanner_ptr->get_num_detectors_per_ring()/2/proj_data_info_ptr->get_num_views();

  return Succeeded::yes;
}

// Load all data that is needed for corrections
void
BinNormalisationFromGEHDF5::
read_norm_data(const string& filename)
{
  // If we actually do not want any correction, forget loading the data
  if(!this->use_detector_efficiencies() && !this->use_geometric_factors())
    return;

  // Build the HDF5 wrapper. This opens the file and makes sure its the correct type, plus loads all information about the scanner. 
  m_input_hdf5_sptr.reset(new GEHDF5Wrapper(filename));

  // We need the norm file to correct for geometry and efficiecies (the geometric correcction is contained inside the norm file too!)
  // But if we are not correcting for efficiencies, then we dont require the file to be a norm file, it can be geo. 
  if(this->use_detector_efficiencies() && !m_input_hdf5_sptr->is_norm_file())
    error("Norm file required, another one given (possibly geo file). Aborting");

  this->scanner_ptr = m_input_hdf5_sptr->get_scanner_sptr();

  // Generate a Projection data Info from the uncompressed scan, 
  proj_data_info_cyl_uncompressed_ptr.reset(
    dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
                    /*span=*/1, 
                    /*max_delta*/scanner_ptr->get_num_rings()-1,
                    /*num_views,=*/scanner_ptr->get_num_detectors_per_ring()/2,
                    /*num_tangential_poss=*/scanner_ptr->get_max_num_non_arccorrected_bins(),
                    /*arc_corrected =*/false) ) );

  //
  // Read data from file
  //

  // Allocate efficiency factor data from an "uncompressed scanner" (i.e. span = 1, all bins are physical bins in the scanner).
  efficiency_factors =
      Array<2,float>(IndexRange2D(0,scanner_ptr->get_num_rings()-1,
                                  0, scanner_ptr->get_num_detectors_per_ring()-1));
  // Initialize the data reading. This internally checks the file and loads required variables fo further reading. 
  m_input_hdf5_sptr->initialise_efficiency_factors();

  // Do the reading using a buffer.
  unsigned int total_size = scanner_ptr->get_num_rings()*scanner_ptr->get_num_detectors_per_ring();
  stir::Array<1, float> buffer(0, total_size-1);
  m_input_hdf5_sptr->read_efficiency_factors(buffer);
  // Aparently GE stores the normalization factor and not the "efficiency factor", so we just need to invert it. 
  // Lambda function, this just applies 1/buffer. 
  std::transform(buffer.begin(), buffer.end(),buffer.begin(), [](const float f) { return 1/f;} );
  // Copy the buffer data to the properly shaped efficiency_factors variable. 
  std::copy(buffer.begin(), buffer.end(), efficiency_factors.begin_all());

  // now read the geo factors
  {
    // somehow fill geo_norm_factors_sptr
  }

#if 1
   // to test pipe the obtained values into file
    ofstream out_eff;
    out_eff.open("eff_out.txt",ios::out);
    //geo_norm_factors_sptr->write_to_file("geo_norm.hs");
   for ( int i = efficiency_factors.get_min_index(); i<=efficiency_factors.get_max_index();i++)
   {
      for ( int j =efficiency_factors[i].get_min_index(); j <=efficiency_factors[i].get_max_index(); j++)
      {
     out_eff << efficiency_factors[i][j] << "   " ;
      }
      out_eff << std::endl<< std::endl;
  }

#endif

#if 0
   //display(geometric_factors, "geo");
  display(efficiency_factors, "eff");
#endif
}

bool 
BinNormalisationFromGEHDF5::
use_detector_efficiencies() const
{
  return this->_use_detector_efficiencies;
}

bool 
BinNormalisationFromGEHDF5::
use_dead_time() const
{
  return this->_use_dead_time;
}

bool 
BinNormalisationFromGEHDF5::
use_geometric_factors() const
{
  return this->_use_geometric_factors;
}

#if 1
float 
BinNormalisationFromGEHDF5::
get_bin_efficiency(const Bin& bin, const double start_time, const double end_time) const
{  
  float	total_efficiency = 0 ;

  /* TODO
     this loop does some complicated stuff with rings etc
     It should be possible to replace this with 

     std::vector<DetectionPositionPair<> > det_pos_pairs;
     proj_data_info_cyl_ptr->get_all_det_pos_pairs_for_bin(det_pos_pairs, bin);
     for (unsigned int i=0; i<det_pos_pairs.size(); ++i)
     {
       ...
     }
  */
  
  /* Correct dead time */
  const int start_view = bin.view_num() * mash ;
  const int end_view = start_view + mash ;
  const int min_ring_diff = proj_data_info_cyl_ptr->get_min_ring_difference(bin.segment_num());
  const int max_ring_diff = proj_data_info_cyl_ptr->get_max_ring_difference(bin.segment_num());


  /* 
     ring1_plus_ring2 is the same for any ring pair that contributes to this particular bin.segment_num(), bin.axial_pos_num().
     We determine it first here. See ProjDataInfoCylindrical for the relevant formulas
  */
  const int ring1_plus_ring2 = detail::calc_ring1_plus_ring2(bin, proj_data_info_cyl_ptr); 
                                                      
  DetectionPositionPair<> detection_position_pair;
  Bin uncompressed_bin(0,0,0,bin.tangential_pos_num());

  
  float view_efficiency = 0.;
  for(uncompressed_bin.view_num() = start_view;
      uncompressed_bin.view_num() < end_view;
      ++uncompressed_bin.view_num() ) 
  {

    detail::set_detection_tangential_coords(proj_data_info_cyl_uncompressed_ptr,
					      uncompressed_bin, detection_position_pair);
 
    float lor_efficiency= 0.;   
      
    /*
      Loop over ring differences that contribute to bin.segment_num() at the current bin.axial_pos_num().
      The ring_difference increments with 2 as the other ring differences do not give a ring pair with this axial_position. 
      This is because: ring1_plus_ring2%2 == ring_diff%2 
      (which easily follows by plugging in ring1+ring2 and ring1-ring2).

      The starting ring_diff is determined such that the above condition is satisfied. 
      You can check it by noting that the
      start_ring_diff%2
      == (min_ring_diff + (min_ring_diff+ring1_plus_ring2)%2)%2
      == (2*min_ring_diff+ring1_plus_ring2)%2
      == ring1_plus_ring2%2
    */
    for(uncompressed_bin.segment_num() = min_ring_diff + (min_ring_diff+ring1_plus_ring2)%2; 
        uncompressed_bin.segment_num() <= max_ring_diff; 
        uncompressed_bin.segment_num()+=2 ) 
    {

      // Make sure we are within the range. Just some error checking. 
      int geo_plane_num = detail::set_detection_axial_coords(proj_data_info_cyl_ptr,
                                                             ring1_plus_ring2, uncompressed_bin,
                                                             detection_position_pair);
      if ( geo_plane_num < 0 ) 
      {
        // Ring numbers out of range.
        continue;
      }

      #ifndef NDEBUG
      Bin check_bin;
      check_bin.set_bin_value(bin.get_bin_value());
      assert(proj_data_info_cyl_ptr->get_bin_for_det_pos_pair(check_bin,detection_position_pair) == Succeeded::yes);
      assert(check_bin == bin);
      #endif
      
      // Here is where the normalization is applied. Apply each of them if required. 
      float lor_efficiency_this_pair = 1.F;
      if (this->use_detector_efficiencies())
      {
        lor_efficiency_this_pair *= get_efficiency_factors(detection_position_pair);
      }
      if (this->use_dead_time())
      {
        lor_efficiency_this_pair *=get_dead_time_efficiency(detection_position_pair, start_time, end_time);
      }
      if (this->use_geometric_factors())
      {
        lor_efficiency_this_pair *=get_geometric_factors(detection_position_pair);
      }
      lor_efficiency += lor_efficiency_this_pair;
    }//endfor

    view_efficiency += lor_efficiency;
    total_efficiency += view_efficiency;
  }
  return total_efficiency;
}
#endif


float 
BinNormalisationFromGEHDF5::get_dead_time_efficiency (const DetectionPositionPair<>& detection_position_pair,
						    const double start_time,
						    const double end_time) const
{
  if (is_null_ptr(singles_rates_ptr)) {
    return 1;
  }

  return 1;  
}

float 
BinNormalisationFromGEHDF5::get_geometric_factors (const DetectionPositionPair<>& detection_position_pair) const
{

  if (is_null_ptr(geo_norm_factors_sptr))
    return 1.F;

  Bin bin;
  if (this->proj_data_info_cyl_ptr->get_bin_for_det_pos_pair(bin,detection_position_pair) == Succeeded::no)
    error("BinNormalisationFromGEHDF5 internal error");

  return this->geo_norm_factors_sptr->get_bin_value(bin);
}

float 
BinNormalisationFromGEHDF5::get_efficiency_factors (const DetectionPositionPair<>& detection_position_pair) const
{
  const DetectionPosition<>& pos1=detection_position_pair.pos1();
  const DetectionPosition<>& pos2=detection_position_pair.pos2();
  // TODO change the tangetial axis flip (scanner_ptr->get_num_detectors_per_ring()-pos1.tangential_coord()) into GEWrapper
  return (this->efficiency_factors[pos1.axial_coord()][this->scanner_ptr->get_num_detectors_per_ring()-pos1.tangential_coord()] *
          this->efficiency_factors[pos2.axial_coord()][this->scanner_ptr->get_num_detectors_per_ring()-pos2.tangential_coord()]);;  
}

} // namespace
}
END_NAMESPACE_STIR

