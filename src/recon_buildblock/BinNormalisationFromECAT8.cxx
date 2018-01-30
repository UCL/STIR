/*
  Copyright (C) 2002-2011, Hammersmith Imanet Ltd
  Copyright (C) 2013-2014 University College London

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
  \ingroup ECAT

  \brief Implementation for class stir::ecat::BinNormalisationFromECAT8

  This file is largely a copy of the ECAT7 version, but reading data via the Interfile-like header of ECAT8.
  \todo merge ECAT7 and 8 code

  \author Kris Thielemans
  \author Sanida Mustafovic
*/


#include "stir/recon_buildblock/BinNormalisationFromECAT8.h"
#include "stir/DetectionPosition.h"
#include "stir/DetectionPositionPair.h"
#include "stir/shared_ptr.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange.h"
#include "stir/Bin.h"
#include "stir/display.h"
#include "stir/IO/read_data.h"
#include "stir/IO/InterfileHeader.h"
#include "stir/ByteOrder.h"
#include "stir/is_null_ptr.h"
#include <algorithm>
#include <fstream>
#include <cctype>
#include <boost/format.hpp>
#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
using std::ios;
#endif

START_NAMESPACE_STIR
START_NAMESPACE_ECAT

	   

const char * const 
BinNormalisationFromECAT8::registered_name = "From ECAT8"; 


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
BinNormalisationFromECAT8::set_defaults()
{
  this->normalisation_ECAT8_filename = "";
  this->_use_gaps = true;
  this->_use_detector_efficiencies = true;
  this->_use_dead_time = false;
  this->_use_geometric_factors = true;
  this->_use_crystal_interference_factors = true;  
}

void 
BinNormalisationFromECAT8::
initialise_keymap()
{
  this->parser.add_start_key("Bin Normalisation From ECAT8");
  // todo remove obsolete keyword
  this->parser.add_key("normalisation_ECAT8_filename", &this->normalisation_ECAT8_filename);
  this->parser.add_key("normalisation_filename", &this->normalisation_ECAT8_filename);
  this->parser.add_parsing_key("singles rates", &this->singles_rates_ptr);
  this->parser.add_key("use_gaps", &this->_use_gaps);
  this->parser.add_key("use_detector_efficiencies", &this->_use_detector_efficiencies);
  //this->parser.add_key("use_dead_time", &this->_use_dead_time);
  this->parser.add_key("use_geometric_factors", &this->_use_geometric_factors);
  this->parser.add_key("use_crystal_interference_factors", &this->_use_crystal_interference_factors);
  this->parser.add_stop_key("End Bin Normalisation From ECAT8");
}

bool 
BinNormalisationFromECAT8::
post_processing()
{
  read_norm_data(normalisation_ECAT8_filename);
  return false;
}


BinNormalisationFromECAT8::
BinNormalisationFromECAT8()
{
  set_defaults();
}

BinNormalisationFromECAT8::
BinNormalisationFromECAT8(const string& filename)
{
  read_norm_data(filename);
}

Succeeded
BinNormalisationFromECAT8::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v)
{
  proj_data_info_ptr = proj_data_info_ptr_v;
  proj_data_info_cyl_ptr =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr *>(proj_data_info_ptr.get());
  if (proj_data_info_cyl_ptr==0)
  {
    warning("BinNormalisationFromECAT8 can only be used on non-arccorrected data\n");
    return Succeeded::no;
  }
  if (*proj_data_info_ptr->get_scanner_ptr()  != *scanner_ptr)
  {
    warning("BinNormalisationFromECAT8: scanner object from proj data is different from the one "
      "from the normalisation file\n");
    return Succeeded::no;
  }

  span = 
    proj_data_info_cyl_ptr->get_max_ring_difference(0) - 
    proj_data_info_cyl_ptr->get_min_ring_difference(0) + 1;
  // TODO insert check all other segments are the same

  mash = scanner_ptr->get_num_detectors_per_ring()/2/proj_data_info_ptr->get_num_views();

  return Succeeded::yes;
}

void
BinNormalisationFromECAT8::
read_norm_data(const string& filename)
{
  
#if 0
MatrixFile* mptr = matrix_open(filename.c_str(),  MAT_READ_ONLY, Norm3d);
  if (mptr == 0)
    error("BinNormalisationFromECAT8: error opening %s\n", filename.c_str());

  scanner_ptr.reset(
    find_scanner_from_ECAT_system_type(mptr->mhptr->system_type));
  
  MatrixData* matrix = matrix_read( mptr, mat_numcod (1, 1, 1, 0, 0), 
				    Norm3d /*= read data as well */);
  
  num_transaxial_crystals_per_block =	nrm_subheader_ptr->num_transaxial_crystals ;
#endif
#if 0
  InterfileHeader interfile_parser;
 add_key("data format", 
    KeyArgument::ASCII,	&KeyParser::do_nothing);
  interfile_parser.parse(filename.c_str());

#else
  KeyParser parser;
  std::string originating_system;
  std::string data_file_name;
  {
    parser.add_start_key("INTERFILE");
    parser.add_key("originating_system", &originating_system);
    parser.add_key("name_of_data_file", &data_file_name);
    parser.parse(filename.c_str());
  }
#endif
  // remove trailing \r
  std::string s=/*interfile_parser.*/originating_system;
  s.erase( std::remove_if( s.begin(), s.end(), isspace ), s.end() );
  /*interfile_parser.*/originating_system=s;
  s=/*interfile_parser.*/data_file_name;
  s.erase( std::remove_if( s.begin(), s.end(), isspace ), s.end() );
  /*interfile_parser.*/data_file_name=s;
  
  if (/*interfile_parser.*/originating_system == "2008")
    this->scanner_ptr.reset(new Scanner(Scanner::Siemens_mMR));
  else
    error(boost::format("Unknown originating_system '%s', when parsing file '%s'") % /*interfile_parser.*/originating_system % filename );


  const std::size_t buf_size = 344*127+9*344+504*64+837+64+64+9+837;
  Array<1,float> buffer(buf_size);
  {
    std::ifstream binary_data(/*interfile_parser.*/data_file_name.c_str(), std::ios::binary|std::ios::in);
    if (read_data(binary_data, buffer, ByteOrder::little_endian) != Succeeded::yes)
      error("failed reading '%s'",/*interfile_parser.*/data_file_name.c_str());

  }
  num_transaxial_crystals_per_block = scanner_ptr->get_num_transaxial_crystals_per_block();
  // Calculate the number of axial blocks per singles unit and 
  // total number of blocks per singles unit.
  int axial_crystals_per_singles_unit = 
    scanner_ptr->get_num_axial_crystals_per_singles_unit();
  
  int transaxial_crystals_per_singles_unit =
    scanner_ptr->get_num_transaxial_crystals_per_singles_unit();
  
  int axial_crystals_per_block = 
    scanner_ptr->get_num_axial_crystals_per_block();

  int transaxial_crystals_per_block =
    scanner_ptr->get_num_transaxial_crystals_per_block();
  
  // Axial blocks.
  num_axial_blocks_per_singles_unit = 
    axial_crystals_per_singles_unit / axial_crystals_per_block;
  
  int transaxial_blocks_per_singles_unit = 
    transaxial_crystals_per_singles_unit / transaxial_crystals_per_block;
  
  // Total blocks.
  num_blocks_per_singles_unit = 
    num_axial_blocks_per_singles_unit * transaxial_blocks_per_singles_unit;
  

#if 0
  if (scanner_ptr->get_num_rings() != nrm_subheader_ptr->num_crystal_rings)
    error("BinNormalisationFromECAT8: "
          "number of rings determined from subheader is %d, while the scanner object says it is %d\n",
           nrm_subheader_ptr->num_crystal_rings, scanner_ptr->get_num_rings());
  if (scanner_ptr->get_num_detectors_per_ring() != nrm_subheader_ptr->crystals_per_ring)
    error("BinNormalisationFromECAT8: "
          "number of detectors per ring determined from subheader is %d, while the scanner object says it is %d\n",
           nrm_subheader_ptr->crystals_per_ring, scanner_ptr->get_num_detectors_per_ring());
#endif
  proj_data_info_cyl_uncompressed_ptr.reset(
    dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr, 
                  /*span=*/1, scanner_ptr->get_num_rings()-1,
                  /*num_views,=*/scanner_ptr->get_num_detectors_per_ring()/2,
				  /*num_tangential_poss=*/344, //XXXnrm_subheader_ptr->num_r_elements, 
                  /*arc_corrected =*/false)
						     ));
  
  /*
    Extract geometrical & crystal interference, and crystal efficiencies from the
    normalisation data.    
  */

  // SM 13/02/2003 corrected number of min_tang_pos_num and max_tang_pos_num used get_max_num_non_arccorrected_bins()
   // instead of get_num_detectors_per_ring()
  //const int min_tang_pos_num = -(scanner_ptr->get_num_detectors_per_ring()/2);
  //const int max_tang_pos_num = min_tang_pos_num + scanner_ptr->get_num_detectors_per_ring() - 1;

   const int min_tang_pos_num = -(scanner_ptr->get_max_num_non_arccorrected_bins())/2;
   const int max_tang_pos_num = min_tang_pos_num +scanner_ptr->get_max_num_non_arccorrected_bins()- 1;

  /* The order of coefficients is as follows: 
  1. geometric_factors (= number_of_corr_planes * number_of_bins)
  2. crystal_interference_factors (num_transaxial_crystals_per_block * number_of_bins)
  3. efficiency_factors (number_of_rings*number_of_crystals ) */

  geometric_factors = 
    Array<2,float>(IndexRange2D(0,127-1, //XXXXnrm_subheader_ptr->num_geo_corr_planes-1,
                                min_tang_pos_num, max_tang_pos_num));
  crystal_interference_factors =
    Array<2,float>(IndexRange2D(min_tang_pos_num, max_tang_pos_num,
		    0, num_transaxial_crystals_per_block-1));
  // SM 13/02/2003 
  efficiency_factors =
    Array<2,float>(IndexRange2D(0,scanner_ptr->get_num_rings()-1,
		   0, scanner_ptr->get_num_detectors_per_ring()-1));
  

#if 0
  int geom_test = nrm_subheader_ptr->num_geo_corr_planes * (max_tang_pos_num-min_tang_pos_num +1);
  int cry_inter = num_transaxial_crystals_per_block * (max_tang_pos_num-min_tang_pos_num +1);
  int eff_test = scanner_ptr->get_num_detectors_per_ring() * scanner_ptr->get_num_rings();
#endif
  
  {
    Array<1,float>::const_iterator data_ptr = buffer.begin();
    for (Array<2,float>::full_iterator iter = geometric_factors.begin_all();
         iter != geometric_factors.end_all();
    )
      *iter++ = *data_ptr++;
    for (Array<2,float>::full_iterator iter = crystal_interference_factors.begin_all();
         iter != crystal_interference_factors.end_all();
    )
      *iter++ = *data_ptr++;
    for (Array<2,float>::full_iterator iter = efficiency_factors.begin_all();
         iter != efficiency_factors.end_all();
    )
      *iter++ = *data_ptr++;
  }

  {
    // for mMR, we need to shift the efficiencies for 1 crystal. This is probably because of where the gap is inserted
    // TODO we have no idea if this is necessary for other ECAT8 scanners.
    // The code below works for the mMR ONLY
    for (int r=0; r<scanner_ptr->get_num_rings(); ++r)
      {
	int c=scanner_ptr->get_num_detectors_per_ring()-1;
	const float save_last_eff = efficiency_factors[r][c];
	for (; c>0; --c)
	  {
	    efficiency_factors[r][c]= efficiency_factors[r][c-1];
	  }
	efficiency_factors[r][c]= save_last_eff;
      }
  }

  if (this->_use_gaps)
    {
      // TODO we really have no idea where the gaps are for every ECAT8 scanners.
      // The code below works for the mMR
      for (int r=0; r<scanner_ptr->get_num_rings(); ++r)
          for (int c=0; c<scanner_ptr->get_num_detectors_per_ring(); 
               c+=scanner_ptr->get_num_transaxial_crystals_per_block())
            {
              efficiency_factors[r][c]=0.F;
            }
    }

  // TODO mvoe dead-time stuff to a separate function
#if 0
  /* Set up equation parameters for dead_time correction */
  float *axial_t1 = nrm_subheader_ptr->ring_dtcor1 ;		/* 'Paralyzing dead_times' for each axial Xstal */
  float *axial_t2 = nrm_subheader_ptr->ring_dtcor2 ;		/* 'Non-paralyzing dead_times' for each axial Xstal */
  /* for 966
     24 entries for axial_t1 & axial_t2
     Each entry accounts for 2 crystal rings
     0 <= iRing <= 23 for ring 0 --> 47
  */
  axial_t1_array = Array<1,float>(0,scanner_ptr->get_num_rings()/num_axial_blocks_per_singles_unit-1);
  axial_t2_array = Array<1,float>(0,scanner_ptr->get_num_rings()/num_axial_blocks_per_singles_unit-1);

  for (Array<1,float>::full_iterator iter = axial_t1_array.begin_all();
         iter != axial_t1_array.end_all();)
      *iter++ = *axial_t1++;

  for (Array<1,float>::full_iterator iter = axial_t2_array.begin_all();
         iter != axial_t2_array.end_all();)
      *iter++ = *axial_t2++;
#if 0
  // this is currently not used by CTI and hence not by get_dead_time_efficiency
  float *trans_t1 = nrm_subheader_ptr->crystal_dtcor ;		/* 'Non-paralyzing dead_times' for each transaxial Xstal in block */
  trans_t1_array = Array<1,float>(0,num_transaxial_crystals_per_block-1);
  for (Array<1,float>::full_iterator iter = trans_t1_array.begin_all();
         iter != trans_t1_array.end_all();)
      *iter++ = *trans_t1++;
#endif
#endif

  
#if 1
   // to test pipe the obtained values into file
    ofstream out_geom;
    ofstream out_inter;
    ofstream out_eff;
    out_geom.open("geom_out.txt",ios::out);
    out_inter.open("inter_out.txt",ios::out);
    out_eff.open("eff_out.txt",ios::out);

    for ( int i = geometric_factors.get_min_index(); i<=geometric_factors.get_max_index();i++)
    {
      for ( int j =geometric_factors[i].get_min_index(); j <=geometric_factors[i].get_max_index(); j++)
      {
	 out_geom << geometric_factors[i][j] << "   " ;
      }
      out_geom << std::endl;
    }


   for ( int i = crystal_interference_factors.get_min_index(); i<=crystal_interference_factors.get_max_index();i++)
   {
      for ( int j =crystal_interference_factors[i].get_min_index(); j <=crystal_interference_factors[i].get_max_index(); j++)
      {
	 out_inter << crystal_interference_factors[i][j] << "   " ;
      }
      out_inter << std::endl;
   }

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
  display(geometric_factors, "geo");
  display(efficiency_factors, "eff");
  display(crystal_interference_factors, "crystal_interference_factors");
#endif
}

bool 
BinNormalisationFromECAT8::
use_detector_efficiencies() const
{
  return this->_use_detector_efficiencies;
}

bool 
BinNormalisationFromECAT8::
use_dead_time() const
{
  return this->_use_dead_time;
}

bool 
BinNormalisationFromECAT8::
use_geometric_factors() const
{
  return this->_use_geometric_factors;
}

bool 
BinNormalisationFromECAT8::
use_crystal_interference_factors() const
{
  return this->_use_crystal_interference_factors;
}

#if 1
float 
BinNormalisationFromECAT8::
get_bin_efficiency(const Bin& bin, const double start_time, const double end_time) const {


  // TODO disable when not HR+ or HR++
  /*
  Additional correction for HR+ and HR++
  ======================================
  Modification of the normalisation based on segment number
  Due to the difference in efficiency for the trues and scatter as the axial
  angle increases
  Scatter has a higher efficiency than trues when the axial angle is 0 (direct
  planes)
  As the axial angle increase the difference in efficiencies between trues and
  scatter become closer
    */
  const float geo_Z_corr = 1;

  
  float	total_efficiency = 0 ;
  
  /* Correct dead time */
  const int start_view = bin.view_num() * mash ;
  //SM removed bin.view_num() + mash ;
  const int end_view = start_view + mash ;
    //start_view +mash;
  const int min_ring_diff = proj_data_info_cyl_ptr->get_min_ring_difference(bin.segment_num());
  const int max_ring_diff = proj_data_info_cyl_ptr->get_max_ring_difference(bin.segment_num());


  /* 
     ring1_plus_ring2 is the same for any ring pair that contributes to 
     this particular bin.segment_num(), bin.axial_pos_num().
     We determine it first here. See ProjDataInfoCylindrical for the
     relevant formulas
  */
  const int ring1_plus_ring2 = detail::calc_ring1_plus_ring2(bin, proj_data_info_cyl_ptr); 
                                                      


  DetectionPositionPair<> detection_position_pair;
  Bin uncompressed_bin(0,0,0,bin.tangential_pos_num());

  {

    float view_efficiency = 0.;


    for(uncompressed_bin.view_num() = start_view;
        uncompressed_bin.view_num() < end_view;
        ++uncompressed_bin.view_num() ) {

      detail::set_detection_tangential_coords(proj_data_info_cyl_uncompressed_ptr,
					      uncompressed_bin, detection_position_pair);

      
        
      float lor_efficiency= 0.;   
      
      /*
        loop over ring differences that contribute to bin.segment_num() at the current
        bin.axial_pos_num().
        The ring_difference increments with 2 as the other ring differences do
        not give a ring pair with this axial_position. This is because
        ring1_plus_ring2%2 == ring_diff%2
        (which easily follows by plugging in ring1+ring2 and ring1-ring2).
        The starting ring_diff is determined such that the above condition
        is satisfied. You can check it by noting that the
        start_ring_diff%2
        == (min_ring_diff + (min_ring_diff+ring1_plus_ring2)%2)%2
        == (2*min_ring_diff+ring1_plus_ring2)%2
        == ring1_plus_ring2%2
      */
      for(uncompressed_bin.segment_num() = min_ring_diff + (min_ring_diff+ring1_plus_ring2)%2; 
          uncompressed_bin.segment_num() <= max_ring_diff; 
          uncompressed_bin.segment_num()+=2 ) {
        
        
        int geo_plane_num = 
	  detail::set_detection_axial_coords(proj_data_info_cyl_ptr,
					     ring1_plus_ring2, uncompressed_bin,
					     detection_position_pair);
        if ( geo_plane_num < 0 ) {
          // Ring numbers out of range.
          continue;
        }


#ifndef NDEBUG
        Bin check_bin;
        check_bin.set_bin_value(bin.get_bin_value());
        assert(proj_data_info_cyl_ptr->get_bin_for_det_pos_pair(check_bin, 
                                                                detection_position_pair) ==
               Succeeded::yes);
        assert(check_bin == bin);
#endif
        
         const DetectionPosition<>& pos1 = detection_position_pair.pos1();
        const DetectionPosition<>& pos2 = detection_position_pair.pos2();

	float lor_efficiency_this_pair = 1.F;
	if (this->use_detector_efficiencies())
	  {
	    lor_efficiency_this_pair =
	      efficiency_factors[pos1.axial_coord()][pos1.tangential_coord()] * 
	      efficiency_factors[pos2.axial_coord()][pos2.tangential_coord()];
	  }
	if (this->use_dead_time())
	  {
	    lor_efficiency_this_pair *=
	      get_dead_time_efficiency(pos1, start_time, end_time) * 
	      get_dead_time_efficiency(pos2, start_time, end_time);
	  }
	if (this->use_geometric_factors())
	  {
	    lor_efficiency_this_pair *=
#ifdef SAME_AS_PETER
              1.F;
#else	    // this is 3dbkproj (at the moment)
	    geometric_factors[geo_plane_num][uncompressed_bin.tangential_pos_num()];
#endif
	  }
	lor_efficiency += lor_efficiency_this_pair;
      }

      if (this->use_crystal_interference_factors())
	{
	  view_efficiency += lor_efficiency * 
	    crystal_interference_factors[uncompressed_bin.tangential_pos_num()][uncompressed_bin.view_num()%num_transaxial_crystals_per_block] ;
	}
      else
	{
	  view_efficiency += lor_efficiency;
	}
    }
    
    if (this->use_geometric_factors())
      {
	/* z==bin.get_axial_pos_num() only when min_axial_pos_num()==0*/
	// for oblique plaanes use the single radial profile from segment 0 
	
#ifdef SAME_AS_PETER	
	const int geo_plane_num = 0;
	
	total_efficiency += view_efficiency * 
	  geometric_factors[geo_plane_num][uncompressed_bin.tangential_pos_num()]  * 
	  geo_Z_corr;
#else
	total_efficiency += view_efficiency * geo_Z_corr;
#endif
      }
    else
      {
	total_efficiency += view_efficiency;
      }
  }
  return total_efficiency;
}
#endif


float 
BinNormalisationFromECAT8::get_dead_time_efficiency (const DetectionPosition<>& det_pos,
						    const double start_time,
						    const double end_time) const
{
  if (is_null_ptr(singles_rates_ptr)) {
    return 1;
  }

  // Get singles rate per block (rate per singles unit / blocks per singles unit).
  const float rate = singles_rates_ptr->get_singles_rate(det_pos, start_time, end_time) / 
    num_blocks_per_singles_unit;
  
  return
    ( 1.0F + axial_t1_array[ det_pos.axial_coord()/num_axial_blocks_per_singles_unit] * rate + 
      axial_t2_array[ det_pos.axial_coord()/num_axial_blocks_per_singles_unit] * rate * rate );
  
  //* ( 1. + ( trans_t1_array[ det_pos.tangential_coord() % num_transaxial_crystals_per_block ] * rate ) ) ;
  
}



END_NAMESPACE_ECAT  
END_NAMESPACE_STIR

