//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \ingroup ECAT

  \brief Implementation for class BinNormalisationFromECAT7

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "local/stir/recon_buildblock/BinNormalisationFromECAT7.h"
#include "stir/DetectionPosition.h"
#include "stir/DetectionPositionPair.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/shared_ptr.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/IndexRange2D.h"
#include "stir/Bin.h"
#include "stir/display.h"
#include <algorithm>

START_NAMESPACE_STIR
START_NAMESPACE_ECAT7

const char * const 
BinNormalisationFromECAT7::registered_name = "From ECAT7"; 

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
  scanner_ptr =
    find_scanner_from_ECAT_system_type(mptr->mhptr->system_type);
  MatrixData* matrix = matrix_read( mptr, mat_numcod (1, 1, 1, 0, 0), 0/*= read data as well */);
  Norm3D_subheader * nrm_subheader_ptr =
      reinterpret_cast<Norm3D_subheader *>(matrix->shptr);
  
  num_transaxial_crystals_per_block =	nrm_subheader_ptr->num_transaxial_crystals ;
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
  //init_deadtime(/* Arguments NOT Used	*/) ; TODO
  
  /*
    Extract geometrical & crystal interference, and crystal efficiencies from the
    normalisation data.    
  */

  const int min_tang_pos_num = -(scanner_ptr->get_num_detectors_per_ring()/2);
  const int max_tang_pos_num = min_tang_pos_num + scanner_ptr->get_num_detectors_per_ring() - 1;
  
  geometric_factors = 
    Array<2,float>(IndexRange2D(0,nrm_subheader_ptr->num_geo_corr_planes-1,
                                min_tang_pos_num, max_tang_pos_num));
  efficiency_factors =
    Array<2,float>(IndexRange2D(0, scanner_ptr->get_num_rings()-1,
                                min_tang_pos_num, max_tang_pos_num));
  crystal_interference_factors =
    Array<2,float>(IndexRange2D(min_tang_pos_num, max_tang_pos_num,
                                0, num_transaxial_crystals_per_block-1));
  
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
#if 0  
		/* Set up equation parameters for deadtime correction */
  axial_t1 = nrm_subheader_ptr->ring_dtcor1 ;		/* 'Paralyzing deadtimes' for each axial Xstal */
  axial_t2 = nrm_subheader_ptr->ring_dtcor2 ;		/* 'Non-paralyzing deadtimes' for each axial Xstal */
  trans_t1 = nrm_subheader_ptr->crystal_dtcor ;		/* 'Non-paralyzing deadtimes' for each transaxial Xstal in block */
  
  
  if ( ( load_deadtime_array( ScanHeader->uncor_singles ) ) != 1 )
		{
    errtxt( "\n\tError...in ldarray\n" ) ;
		}
#endif
  free_matrix_data(matrix);
  matrix_close(mptr);


#if 1
  display(geometric_factors, "geo");
  display(efficiency_factors, "eff");
  display(crystal_interference_factors, "crystal_interference_factors");
#endif
}



float 
BinNormalisationFromECAT7::
get_efficiency_factor(const Bin& bin) const 
{
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
  const int rtmp = bin.segment_num() * span ;
  const float geo_Z_corr = ( 1.0F + ( ( 0.007F - ( 0.000164F * rtmp ) ) * rtmp ) ) ;
  
  float	total_efficiency = 0 ;
  
  /* Correct dead time */
  const int start_view = bin.view_num() * mash ;
  const int end_view = bin.view_num() + mash ;
  const int min_ring_diff = proj_data_info_cyl_ptr->get_min_ring_difference(bin.segment_num());
  const int max_ring_diff = proj_data_info_cyl_ptr->get_max_ring_difference(bin.segment_num());
  DetectionPositionPair<> detection_position_pair;
  Bin uncompressed_bin(0,0,0, bin.tangential_pos_num());

    {
      float view_efficiency = 0. ;
      for(uncompressed_bin.view_num() = start_view; uncompressed_bin.view_num() < end_view; ++uncompressed_bin.view_num() )
      {
        float lor_efficiency = 0. ;
        
        for(uncompressed_bin.segment_num() = min_ring_diff; uncompressed_bin.segment_num() <= max_ring_diff; ++uncompressed_bin.segment_num() )
        {
          proj_data_info_cyl_uncompressed_ptr->get_det_pos_pair_for_bin(detection_position_pair,uncompressed_bin);
#ifndef NDEBUG
          Bin check_bin;
          assert(proj_data_info_cyl_ptr->get_bin_for_det_pos_pair(check_bin, detection_position_pair) ==
            Succeeded::yes);
          assert(check_bin == bin);
#endif
          const DetectionPosition<>& pos1 = detection_position_pair.pos1();
          const DetectionPosition<>& pos2 = detection_position_pair.pos2();
        
          lor_efficiency += 
            efficiency_factors[pos1.axial_coord()][pos1.tangential_coord()] * 
            efficiency_factors[pos2.axial_coord()][pos2.tangential_coord()];
          
        }
        view_efficiency += lor_efficiency * crystal_interference_factors[uncompressed_bin.tangential_pos_num()][uncompressed_bin.view_num()%num_transaxial_crystals_per_block] ;
      }
      /* z==bin.get_axial_pos_num() only when min_axial_pos_num()==0*/
      const int geo_plane_num = /* TODO septa_in ? (z=)bin.get_axial_pos_num() : */0;
      total_efficiency += view_efficiency * geometric_factors[geo_plane_num][uncompressed_bin.tangential_pos_num()]  * geo_Z_corr;            
    }
   
  return total_efficiency;
}


void 
BinNormalisationFromECAT7::apply(RelatedViewgrams<float>& viewgrams) const 
{
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
  {
    Bin bin(iter->get_segment_num(),0,iter->get_view_num(), 0);
    for (bin.axial_pos_num()= iter->get_min_axial_pos_num(); bin.axial_pos_num()<=iter->get_max_axial_pos_num(); ++bin.axial_pos_num())
      for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num(); bin.tangential_pos_num()<=iter->get_max_tangential_pos_num(); ++bin.tangential_pos_num())
         (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] /= 
#ifndef STIR_NO_NAMESPACES
         std::
#endif
         max(1.E-20F, get_efficiency_factor(bin));
  }
}

void 
BinNormalisationFromECAT7::
undo(RelatedViewgrams<float>& viewgrams) const 
{
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
  {
    Bin bin(iter->get_segment_num(),0,iter->get_view_num(), 0);
    for (bin.axial_pos_num()= iter->get_min_axial_pos_num(); bin.axial_pos_num()<=iter->get_max_axial_pos_num(); ++bin.axial_pos_num())
      for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num(); bin.tangential_pos_num()<=iter->get_max_tangential_pos_num(); ++bin.tangential_pos_num())
         (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] *= get_efficiency_factor(bin);
  }

}


END_NAMESPACE_ECAT7  
END_NAMESPACE_STIR

