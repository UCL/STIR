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
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInterfile.h"
#include "stir/display.h"
#include "stir/IO/read_data.h"
#include "stir/IO/InterfileHeader.h"
#include "stir/ByteOrder.h"
#include "stir/is_null_ptr.h"
#include "stir/modulo.h"
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
set_detection_tangential_coords(shared_ptr<const ProjDataInfoCylindricalNoArcCorr> proj_data_cyl_uncomp,
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

// Returns a vetor with the segment sequence, as 0, 1, -1, 2, -2,...
static
std::vector<int> // Prefered since C++11
create_segment_sequence(shared_ptr<ProjDataInfo> const& proj_data_info_ptr)
{
  std::vector<int> segment_sequence;
  segment_sequence.resize(2*proj_data_info_ptr->get_max_segment_num()+1);
  segment_sequence[0] = 0;
  // PW Flipped the segments, segment sequence is now as: 0,1,-1 and so on.
  for (int segment_num = 1; segment_num<=proj_data_info_ptr->get_max_segment_num(); ++segment_num)
  {
    segment_sequence[2*segment_num-1] = segment_num;
    segment_sequence[2*segment_num] = -segment_num;
  }
  return segment_sequence;
}

// Returns the index in the segment sequence for a given segment number (e.g -1 returns 2)
static
unsigned int
find_segment_index_in_sequence(std::vector<int>& segment_sequence, const int segment_num)
{
  std::vector<int>::const_iterator iter = std::find(segment_sequence.begin(), segment_sequence.end(), segment_num);
  assert(iter !=  segment_sequence.end());
  return static_cast<int>(iter - segment_sequence.begin());
}
// Creates a vector that has the axial position offset for each segment. 
static
std::vector<unsigned int> // Prefered since C++11
create_ax_pos_offset(shared_ptr<ProjDataInfo> const& proj_data_info_ptr, std::vector<int>& segment_sequence)
{
  std::vector<unsigned int> seg_ax_offset;
  seg_ax_offset.resize(proj_data_info_ptr->get_num_segments());

  seg_ax_offset[0] = 0;

  unsigned int previous_value = 0;

  for (int i_seg = 1; i_seg < proj_data_info_ptr->get_num_segments(); ++i_seg)
  {
      const int segment_num = segment_sequence[i_seg-1];

      seg_ax_offset[i_seg] = static_cast<unsigned int>(proj_data_info_ptr->get_num_axial_poss(segment_num)) +
                                                       previous_value;
      previous_value = seg_ax_offset[i_seg];
  }
  return seg_ax_offset;
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
  this->set_calibration_factor(1); //TODO: read actual factor somewhere
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
set_up(const shared_ptr<const ExamInfo> &exam_info_sptr, const shared_ptr<const ProjDataInfo>& proj_data_info_ptr_v)
{
  BinNormalisation::set_up(exam_info_sptr, proj_data_info_ptr_v);
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
// This function will take a filename, that can be either a GE norm file, or a GE geo file. If you wan to do geo and norm corrections
// then you want the norm file as input, and if you only want geo files, then just the geo file is enough. The function will read from these files and
// fill the atributes with a full sinogram of efficiency factors and geometry factors. 
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
  // Read efficiency data from file
  //
  if(this->use_detector_efficiencies())
  {
    // Allocate efficiency factor data from an "uncompressed scanner" (i.e. span = 1, all bins are physical bins in the scanner).
    efficiency_factors =
        Array<2,float>(IndexRange2D(0,scanner_ptr->get_num_rings()-1,
                                    0, scanner_ptr->get_num_detectors_per_ring()-1));
    // Initialize the data reading. This internally checks the file and loads required variables fo further reading. 
    m_input_hdf5_sptr->initialise_efficiency_factors();

    // Do the reading using a buffer.
    unsigned int total_size = (scanner_ptr->get_num_rings()-1)*(scanner_ptr->get_num_detectors_per_ring()-1);
    stir::Array<1, float> buffer(0, total_size-1);
    m_input_hdf5_sptr->read_efficiency_factors(buffer);
    // Aparently GE stores the normalization factor and not the "efficiency factor", so we just need to invert it. 
    // Lambda function, this just applies 1/buffer and stores it in efficiency_factors 
    std::transform(buffer.begin(), buffer.end(),efficiency_factors.begin_all(), [](const float f) { return 1/f;} );
  }
  //
  // Read geo data from file
  //
  if(this->use_geometric_factors())
  {
    // Construct a proper ProjDataInfo to initialize geometry factors array and use it to know the boudns of the iteratios to load it.
    shared_ptr<ProjDataInfo> projInfo = ProjDataInfo::construct_proj_data_info(scanner_ptr,
                                            /*span*/ 2,  
                                            /* max_delta*/ scanner_ptr->get_num_rings()-1,
                                            /* num_views */ scanner_ptr->get_num_detectors_per_ring()/2,
                                            /* num_tangential_poss */ scanner_ptr->get_max_num_non_arccorrected_bins(),
                                            /* arc_corrected */ false
                                             );
    geo_eff_factors_sptr.reset(new ProjDataInMemory(m_input_hdf5_sptr->get_exam_info_sptr(),
                                                 projInfo,
                                                 true)); // Initialize with zeroes (always true internally...)

    // TODO: remove all these loops and "duplication", and load the entire geometric factors file. Then modify the function get_geometric_factors() 
    // so that when accessed, re-indexes the bin number to the correct geometric factor.
    // Doing this would save lots of RAM, as there are lots of symetries that are exploited in the geo file, but we are here undoing all that and duplicating data.
    
    // These arrays will help us index the data to read. Just auxiliary variables.
    std::vector<int>          segment_sequence              = detail::create_segment_sequence(projInfo);
    std::vector<unsigned int> segment_axial_position_offset = detail::create_ax_pos_offset   (projInfo, segment_sequence);

    int num_crystals_per_bucket=scanner_ptr->get_num_transaxial_crystals_per_bucket();
    // Geometric factors are related to geometry (ovbiously). This means that as the scanner has several geometric symetries itself, there is no need to
    // store all of them in a big file. This is what GE does in RDF9 files.
    // The following loops undo that. They go selecting different data pieces in the initialise_geo_factors() and reading different parts of 
    // it in read_geo_factors(), all to create a complete sinogram with all the geo factors loaded. 
    for (int i_seg = projInfo->get_min_segment_num(); i_seg <= projInfo->get_max_segment_num(); ++i_seg)
    {
      for(int i_view = 0; i_view < scanner_ptr->get_max_num_views(); ++i_view)
      {
          // Auxiliary single viewgram as a buffer
          Viewgram<float> viewgram = projInfo->get_empty_viewgram(projInfo->get_num_views()-1-i_view, i_seg);
          // AB TODO This allocates the memory. I wish I knew how to do this without continous reallocation (by reusing)
          viewgram.fill(0.0); 
          switch (m_input_hdf5_sptr->get_geo_dims())
          {
          case 3:
          {
            m_input_hdf5_sptr->initialise_geo_factors_data(modulo(i_view,num_crystals_per_bucket)+1);

            // Define which chunk of the data we are reading from. 
            std::array<hsize_t, 2> offset = {segment_axial_position_offset[detail::find_segment_index_in_sequence(segment_sequence,i_seg)], 0};
            std::array<hsize_t, 2> count  = {static_cast<hsize_t>(projInfo->get_num_axial_poss(i_seg)),
                                             static_cast<hsize_t>(projInfo->get_num_tangential_poss())};
            // Initialize buffer to store temp variables
            stir::Array<1, unsigned int> buffer(0, count[0]*count[1]-1);
            // read geo chunk
            m_input_hdf5_sptr->read_geometric_factors(buffer, offset, count);
            // copy data back
            // AB TODO: Hardcoded magic number, remove somehow (when magic is discovered)
            std::transform(buffer.begin(), buffer.end(),viewgram.begin_all(), [](const float f) { return 1/(f*2.2110049e-4);} );
            break;
          }
          case 2:
          {
            m_input_hdf5_sptr->initialise_geo_factors_data(1);

            std::array<hsize_t, 2> offset = {static_cast<hsize_t>(modulo(i_view,num_crystals_per_bucket)), 0};
            std::array<hsize_t, 2> count  = {1, static_cast<hsize_t>(projInfo->get_num_tangential_poss())};
            // Initialize buffer to store temp variables
            stir::Array<1, unsigned int> buffer(0, count[1]-1);
            // read geo chunk
            m_input_hdf5_sptr->read_geometric_factors(buffer, offset, count);
            std::vector<unsigned int> repeat_buffer;
            repeat_buffer.reserve(projInfo->get_num_axial_poss(i_seg)*count[1]-1);
            // repeat the values
            for (int i=0; i<projInfo->get_num_axial_poss(i_seg);i++)
              repeat_buffer.insert(repeat_buffer.end(),buffer.begin(),buffer.end());
            // copy data back
            // AB TODO: Hardcoded magic number, remove somehow (when magic is discovered)
            std::transform(repeat_buffer.begin(), repeat_buffer.end(),viewgram.begin_all(), [](const float f) { return 1/(f*2.2110049e-4);} );
            break;
          }
          default:
            error("BinNormalisationFromGEHDF5: Unexpected geometry type");
          }
          
          geo_eff_factors_sptr->set_viewgram(viewgram);

      }// end view for
    }// end segment for
#if 0 // Use this to store loaded geo result in an interfile format. Useful for debugging purposes. 
    shared_ptr<ProjData> output_projdata_ptr;
    const string filename="geo_debug.hs";
    output_projdata_ptr.reset(new ProjDataInterfile(m_input_hdf5_sptr->get_exam_info_sptr(),projInfo,filename));
    for (int i_seg = projInfo->get_min_segment_num(); i_seg <= projInfo->get_max_segment_num(); ++i_seg)
      for(int i_view = 0; i_view < scanner_ptr->get_max_num_views(); ++i_view)
      {
        output_projdata_ptr->set_viewgram(geo_eff_factors_sptr->get_viewgram(i_view,i_seg));
      }
#endif
  }// end loading of geo factors
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

float 
BinNormalisationFromGEHDF5::
get_uncalibrated_bin_efficiency(const Bin& bin) const
{  
    
    const float start_time=get_exam_info_sptr()->get_time_frame_definitions().get_start_time();
    const float end_time=get_exam_info_sptr()->get_time_frame_definitions().get_end_time();
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
      
      // Here is where the normalization is applied. Apply each of them if required. 
      float lor_efficiency_this_pair = 1.F;
      if (this->use_detector_efficiencies())
      {
        lor_efficiency_this_pair *= get_efficiency_factors(detection_position_pair);
      }
      if (this->use_dead_time())
      {
        lor_efficiency_this_pair *= get_dead_time_efficiency(detection_position_pair, start_time, end_time);
      }
      if (this->use_geometric_factors())
      {
        lor_efficiency_this_pair *= get_geometric_efficiency_factors(detection_position_pair);
      }
      lor_efficiency += lor_efficiency_this_pair;
    }//endfor

    view_efficiency += lor_efficiency;
    total_efficiency += view_efficiency;
  }

  return total_efficiency;
}

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
BinNormalisationFromGEHDF5::get_geometric_efficiency_factors (const DetectionPositionPair<>& detection_position_pair) const
{

  if (is_null_ptr(geo_eff_factors_sptr))
    return 1.F;

  Bin bin;
  if (this->proj_data_info_cyl_ptr->get_bin_for_det_pos_pair(bin,detection_position_pair) == Succeeded::no)
    error("BinNormalisationFromGEHDF5 internal error");

  return this->geo_eff_factors_sptr->get_bin_value(bin);
}

float 
BinNormalisationFromGEHDF5::get_efficiency_factors (const DetectionPositionPair<>& detection_position_pair) const
{
  const DetectionPosition<>& pos1=detection_position_pair.pos1();
  const DetectionPosition<>& pos2=detection_position_pair.pos2();
  return (this->efficiency_factors[pos1.axial_coord()][pos1.tangential_coord()] *
          this->efficiency_factors[pos2.axial_coord()][pos2.tangential_coord()]);  
}

} // namespace
}
END_NAMESPACE_STIR

