//
// $Id$
//
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
    Copyright (C) CTI
    This file is part of STIR.

    Some parts of this file originate in CTI code, distributed as
    part of the matrix library from Louvain-la-Neuve, and hence carries
    its restrictive license. Affected parts are the dead-time correction
    in get_deadtime_efficiency and geo_Z_corr related code.

    Most of this file is free software; you can redistribute that part and/or modify
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

  \brief Implementation for class stir::ecat::ecat7::BinNormalisationFromECAT7

  \author Kris Thielemans
  \author Sanida Mustafovic
  $Date$
  $Revision$
*/

// enable if you want results identical to Peter Bloomfield's normalisation code
// (and hence old versions of Bkproj_3d)
// #define SAME_AS_PETER

#include "stir/recon_buildblock/BinNormalisationFromECAT7.h"
#include "stir/DetectionPosition.h"
#include "stir/DetectionPositionPair.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/shared_ptr.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange.h"
#include "stir/Bin.h"
#include "stir/display.h"
#include "stir/is_null_ptr.h"
#include <algorithm>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
#endif

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

	   

const char * const 
BinNormalisationFromECAT7::registered_name = "From ECAT7"; 


namespace detail
{

//
// helper functions used in this class.
//

static 
float
calc_geo_z_correction(const Bin& bin, int span) {
  const int rtmp = abs(bin.segment_num() * span) ;
  return( 1.0F + ( ( 0.007F - ( 0.000164F * rtmp ) ) * rtmp ) );
}




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
BinNormalisationFromECAT7::set_defaults()
{
  normalisation_ECAT7_filename = "";
}

void 
BinNormalisationFromECAT7::
initialise_keymap()
{
  parser.add_start_key("Bin Normalisation From ECAT7");
  parser.add_key("normalisation_ECAT7_filename", &normalisation_ECAT7_filename);
  parser.add_parsing_key("singles rates", &singles_rates_ptr);
  parser.add_stop_key("End Bin Normalisation From ECAT7");
}

bool 
BinNormalisationFromECAT7::
post_processing()
{
  read_norm_data(normalisation_ECAT7_filename);
  return false;
}


BinNormalisationFromECAT7::
BinNormalisationFromECAT7()
{
  set_defaults();
}

BinNormalisationFromECAT7::
BinNormalisationFromECAT7(const string& filename)
{
  read_norm_data(filename);
}

Succeeded
BinNormalisationFromECAT7::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v)
{
  proj_data_info_ptr = proj_data_info_ptr_v;
  proj_data_info_cyl_ptr =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr *>(proj_data_info_ptr.get());
  if (proj_data_info_cyl_ptr==0)
  {
    warning("BinNormalisationFromECAT7 can only be used on non-arccorrected data\n");
    return Succeeded::no;
  }
  if (*proj_data_info_ptr->get_scanner_ptr()  != *scanner_ptr)
  {
    warning("BinNormalisationFromECAT7: scanner object from proj data is different from the one "
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
BinNormalisationFromECAT7::
read_norm_data(const string& filename)
{
  
  MatrixFile* mptr = matrix_open(filename.c_str(),  MAT_READ_ONLY, Norm3d);
  if (mptr == 0)
    error("BinNormalisationFromECAT7: error opening %s\n", filename.c_str());

  scanner_ptr =
    find_scanner_from_ECAT_system_type(mptr->mhptr->system_type);
  
  MatrixData* matrix = matrix_read( mptr, mat_numcod (1, 1, 1, 0, 0), 
				    Norm3d /*= read data as well */);
  if (matrix == 0)
    error("BinNormalisationFromECAT7: error reading data in  %s\n", filename.c_str());

  Norm3D_subheader * nrm_subheader_ptr =
      reinterpret_cast<Norm3D_subheader *>(matrix->shptr);
  
  num_transaxial_crystals_per_block =	nrm_subheader_ptr->num_transaxial_crystals ;


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
  


  if (scanner_ptr->get_num_rings() != nrm_subheader_ptr->num_crystal_rings)
    error("BinNormalisationFromECAT7: "
          "number of rings determined from subheader is %d, while the scanner object says it is %d\n",
           nrm_subheader_ptr->num_crystal_rings, scanner_ptr->get_num_rings());
  if (scanner_ptr->get_num_detectors_per_ring() != nrm_subheader_ptr->crystals_per_ring)
    error("BinNormalisationFromECAT7: "
          "number of detectors per ring determined from subheader is %d, while the scanner object says it is %d\n",
           nrm_subheader_ptr->crystals_per_ring, scanner_ptr->get_num_detectors_per_ring());
  
   proj_data_info_cyl_uncompressed_ptr =
    dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr, 
                  /*span=*/1, scanner_ptr->get_num_rings()-1,
                  /*num_views,=*/scanner_ptr->get_num_detectors_per_ring()/2,
                  /*num_tangential_poss=*/nrm_subheader_ptr->num_r_elements, 
                  /*arc_corrected =*/false)
                  );
  
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
    Array<2,float>(IndexRange2D(0,nrm_subheader_ptr->num_geo_corr_planes-1,
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
    float const* data_ptr = reinterpret_cast<float const *>(matrix->data_ptr);
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
  // TODO mvoe dead-time stuff to a separate function
  /* Set up equation parameters for deadtime correction */
  float *axial_t1 = nrm_subheader_ptr->ring_dtcor1 ;		/* 'Paralyzing deadtimes' for each axial Xstal */
  float *axial_t2 = nrm_subheader_ptr->ring_dtcor2 ;		/* 'Non-paralyzing deadtimes' for each axial Xstal */
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
  // this is currently not used by CTI and hence not by get_deadtime_efficiency
  float *trans_t1 = nrm_subheader_ptr->crystal_dtcor ;		/* 'Non-paralyzing deadtimes' for each transaxial Xstal in block */
  trans_t1_array = Array<1,float>(0,num_transaxial_crystals_per_block-1);
  for (Array<1,float>::full_iterator iter = trans_t1_array.begin_all();
         iter != trans_t1_array.end_all();)
      *iter++ = *trans_t1++;
#endif

  
  free_matrix_data(matrix);
  matrix_close(mptr);
#if 0
   // to test pipe the obtained values into file
  // char out_name[max_filename_length];
    char name[80];
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
      out_geom << endl;
    }


   for ( int i = crystal_interference_factors.get_min_index(); i<=crystal_interference_factors.get_max_index();i++)
   {
      for ( int j =crystal_interference_factors[i].get_min_index(); j <=crystal_interference_factors[i].get_max_index(); j++)
      {
	 out_inter << crystal_interference_factors[i][j] << "   " ;
      }
      out_inter << endl << endl;
   }

   for ( int i = efficiency_factors.get_min_index(); i<=efficiency_factors.get_max_index();i++)
   {
      for ( int j =efficiency_factors[i].get_min_index(); j <=efficiency_factors[i].get_max_index(); j++)
      {
	 out_eff << efficiency_factors[i][j] << "   " ;
      }
      out_eff << endl<< endl;
   }

#endif

#if 0
  display(geometric_factors, "geo");
  display(efficiency_factors, "eff");
  display(crystal_interference_factors, "crystal_interference_factors");
#endif
}

#if 1
float 
BinNormalisationFromECAT7::
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
  const float geo_Z_corr = detail::calc_geo_z_correction(bin, span);

  
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

        
        lor_efficiency += 
          efficiency_factors[pos1.axial_coord()][pos1.tangential_coord()] * 
          efficiency_factors[pos2.axial_coord()][pos2.tangential_coord()] * 
          get_deadtime_efficiency(pos1, start_time, end_time) * 
          get_deadtime_efficiency(pos2, start_time, end_time)
#ifdef SAME_AS_PETER
              ;
#else	    // this is 3dbkproj (at the moment)
        * geometric_factors[geo_plane_num][uncompressed_bin.tangential_pos_num()];
#endif
      }
      
      
      
      view_efficiency += lor_efficiency * 
        crystal_interference_factors[uncompressed_bin.tangential_pos_num()][uncompressed_bin.view_num()%num_transaxial_crystals_per_block] ;
      
    }
    
    /* z==bin.get_axial_pos_num() only when min_axial_pos_num()==0*/
    // for oblique plaanes use the single radial profile from segment 0 
    
#ifdef SAME_AS_PETER

    const int geo_plane_num = /* TODO septa_in ? (z=)bin.get_axial_pos_num() : */0;

    total_efficiency += view_efficiency * 
      geometric_factors[geo_plane_num][uncompressed_bin.tangential_pos_num()]  * 
      geo_Z_corr;

#else

    total_efficiency += view_efficiency * geo_Z_corr;            

#endif
    
  }
  return total_efficiency;
}
#endif


void 
BinNormalisationFromECAT7::apply(RelatedViewgrams<float>& viewgrams,
                                 const double start_time, const double end_time) const 
{

  // Generate a single set of average singles values for the interval
  // start_time to end_time.
  
  


  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
  {
 //   if (iter->get_view_num()>8)
 //     continue;

    Bin bin(iter->get_segment_num(),iter->get_view_num(), 0, 0);

    for (bin.axial_pos_num() = iter->get_min_axial_pos_num(); 
         bin.axial_pos_num() <= iter->get_max_axial_pos_num(); 
         ++bin.axial_pos_num()) {

      for (bin.tangential_pos_num() = iter->get_min_tangential_pos_num(); 
           bin.tangential_pos_num() <= iter->get_max_tangential_pos_num(); 
           ++bin.tangential_pos_num()) {

        (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] /= 
#ifndef STIR_NO_NAMESPACES
          std::
#endif
          max(1.E-20F, get_bin_efficiency(bin, start_time, end_time));
      }
    }
  }

}

void 
BinNormalisationFromECAT7::
undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)

  {
    Bin bin(iter->get_segment_num(),iter->get_view_num(), 0,0);
    for (bin.axial_pos_num()= iter->get_min_axial_pos_num(); bin.axial_pos_num()<=iter->get_max_axial_pos_num(); ++bin.axial_pos_num())
      for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num(); bin.tangential_pos_num()<=iter->get_max_tangential_pos_num(); ++bin.tangential_pos_num())
         (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] *= get_bin_efficiency(bin,start_time, end_time);
  }

}


float 
BinNormalisationFromECAT7::get_deadtime_efficiency (const DetectionPosition<>& det_pos,
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
    ( 1.0 + axial_t1_array[ det_pos.axial_coord()/num_axial_blocks_per_singles_unit] * rate + 
      axial_t2_array[ det_pos.axial_coord()/num_axial_blocks_per_singles_unit] * rate * rate );
  
  //* ( 1. + ( trans_t1_array[ det_pos.tangential_coord() % num_transaxial_crystals_per_block ] * rate ) ) ;
  
}



END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT  
END_NAMESPACE_STIR

