/*
  Copyright (C) 2002-2011, Hammersmith Imanet Ltd
  Copyright (C) 2013-2014 University College London
  Copyright (C) 2017-2018 University of Leeds

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

  \brief Implementation for class stir::ecat::BinNormalisationFromGEHDF5

  This file is largely a copy of the ECAT7 version, but reading data via the Interfile-like header of GEHDF5.
  \todo merge ECAT7 and 8 code

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
  this->_use_gaps = false;
  this->_use_detector_efficiencies = true;
  this->_use_dead_time = false;
  this->_use_geometric_factors =false;
  this->_use_crystal_interference_factors = false;  
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
  //this->parser.add_key("use_geometric_factors", &this->_use_geometric_factors);
  //this->parser.add_key("use_crystal_interference_factors", &this->_use_crystal_interference_factors);
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

  span = 
    proj_data_info_cyl_ptr->get_max_ring_difference(0) - 
    proj_data_info_cyl_ptr->get_min_ring_difference(0) + 1;
  // TODO insert check all other segments are the same

  mash = scanner_ptr->get_num_detectors_per_ring()/2/proj_data_info_ptr->get_num_views();

  return Succeeded::yes;
}

void
BinNormalisationFromGEHDF5::
read_norm_data(const string& filename)
{
  
//  this->h5data.open(filename);
//  this->scanner_ptr = this->h5data.get_scanner_sptr();

//  num_transaxial_crystals_per_block = scanner_ptr->get_num_transaxial_crystals_per_block();
//  // Calculate the number of axial blocks per singles unit and
//  // total number of blocks per singles unit.
//  int axial_crystals_per_singles_unit =
//    scanner_ptr->get_num_axial_crystals_per_singles_unit();
  
//  int transaxial_crystals_per_singles_unit =
//    scanner_ptr->get_num_transaxial_crystals_per_singles_unit();
  
//  int axial_crystals_per_block =
//    scanner_ptr->get_num_axial_crystals_per_block();

//  int transaxial_crystals_per_block =
//    scanner_ptr->get_num_transaxial_crystals_per_block();
  
//  // Axial blocks.
//  num_axial_blocks_per_singles_unit =
//    axial_crystals_per_singles_unit / axial_crystals_per_block;
  
//  int transaxial_blocks_per_singles_unit =
//    transaxial_crystals_per_singles_unit / transaxial_crystals_per_block;
  
//  // Total blocks.
//  num_blocks_per_singles_unit =
//    num_axial_blocks_per_singles_unit * transaxial_blocks_per_singles_unit;
  

//#if 0
//  if (scanner_ptr->get_num_rings() != nrm_subheader_ptr->num_crystal_rings)
//    error("BinNormalisationFromGEHDF5: "
//          "number of rings determined from subheader is %d, while the scanner object says it is %d\n",
//           nrm_subheader_ptr->num_crystal_rings, scanner_ptr->get_num_rings());
//  if (scanner_ptr->get_num_detectors_per_ring() != nrm_subheader_ptr->crystals_per_ring)
//    error("BinNormalisationFromGEHDF5: "
//          "number of detectors per ring determined from subheader is %d, while the scanner object says it is %d\n",
//           nrm_subheader_ptr->crystals_per_ring, scanner_ptr->get_num_detectors_per_ring());
//#endif
//  proj_data_info_cyl_uncompressed_ptr.reset(
//    dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(
//    ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
//                  /*span=*/1, scanner_ptr->get_num_rings()-1,
//                  /*num_views,=*/scanner_ptr->get_num_detectors_per_ring()/2,
//				  /*num_tangential_poss=*/scanner_ptr->get_max_num_non_arccorrected_bins(),
//                  /*arc_corrected =*/false)
//						     ));
  
//  /*
//    Extract geometrical & crystal interference, and crystal efficiencies from the
//    normalisation data.
//  */

//   const int min_tang_pos_num = -(scanner_ptr->get_max_num_non_arccorrected_bins())/2;
//   const int max_tang_pos_num = min_tang_pos_num +scanner_ptr->get_max_num_non_arccorrected_bins()- 1;

//    geometric_factors =
//    Array<3,float>(IndexRange3D(0,15, 0,1981-1, //XXXXnrm_subheader_ptr->num_geo_corr_planes-1,
//                                 min_tang_pos_num, max_tang_pos_num));

//   {
//         using namespace H5;
//         using namespace std;
//         int slice = 0;

//     while ( slice < _num_time_slices) {

//       std::cout<<"Now processing view"<< slice+1 <<std::endl;

//       //PW Open the dataset from that file here.
//        DataSet dataset = this->h5data.get_file().openDataSet("/SegmentData/Segment4/3D_Norm_Correction/slice%d", slice+1);

//       /*
//         * Get dataspace of the dataset.
//         */
//      DataSpace dataspace = dataset.getSpace();
//        /*
//         * Get the number of dimensions in the dataspace.
//         */
//       int rank = dataspace.getSimpleExtentNdims();
//        /*
//         * Get the dimension size of each dimension in the dataspace and
//         * display them.
//         */
//       hsize_t dims_out[3];
//        dataspace.getSimpleExtentDims( dims_out, NULL);
//        /*
//         * Define hyperslab in the dataset; implicitly giving strike and
//         * block NULL.
//         */
//        hsize_t      offset[3];   // hyperslab offset in the file
//        hsize_t      count[3];    // size of the hyperslab in the file
//        offset[0] = 0;
//        offset[1] = 0;
//        offset[2] = 0;
//        count[0]  = dims_out[0];
//        count[1]  = dims_out[1];
//        count[2]  = dims_out[2];
//        dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );

//        /*
//         * Define the memory dataspace.
//         */
//        hsize_t     dimsm[3];
//        dimsm[0] = dims_out[0];
//        dimsm[1] = dims_out[1];
//        dimsm[2] = dims_out[2];
//        DataSpace memspace( 3, dimsm );

//       //PW Read data from hyperslab in the file into the hyperslab in memory.

//        Array<1,int> buffer(dimsm[0]*dimsm[1]*dimsm[2]);
//        dataset.read( buffer.get_data_ptr(), H5::PredType::NATIVE_INT, memspace, dataspace );
//        buffer.release_data_ptr();

//        std::copy(buffer.begin(), b7uffer.end(), tof_data.begin_all());
//        Array<1,float> data();
//        dataset.read( data[slice].get_data_ptr(), PredType::NATIVE_FLOAT, memspace, dataspace);
//        data[slice].release_data_ptr();

//        // Increment the slice index.
//           ++slice;
//  }
//           for (int i = 0; i<= 15; i++)
//           {
//              geometric_factors[i][]
//           }
//            std::copy(data[slice].begin(), data[slice].end(), geometric_factors.begin_all());
//                                  min_tang_pos_num, max_tang_pos_num));
//    }


//    efficiency_factors =
//    Array<2,float>(IndexRange2D(0,scanner_ptr->get_num_rings()-1,
//		   0, scanner_ptr->get_num_detectors_per_ring()-1));
  

//  {
//    using namespace H5;
//    using namespace std;

//    DataSet dataset = this->h5data.get_file().openDataSet("/3DCrystalEfficiency/crystalEfficiency");
//     /*
//       * Get dataspace of the dataset.
//       */
//      DataSpace dataspace = dataset.getSpace();
//      /*
//       * Get the number of dimensions in the dataspace.
//       */
//      int rank = dataspace.getSimpleExtentNdims();
//      /*
//       * Get the dimension size of each dimension in the dataspace and
//       * display them.
//       */
//      hsize_t dims_out[2];
//      dataspace.getSimpleExtentDims( dims_out, NULL);
//      cout << "rank " << rank << ", dimensions " <<
//          (unsigned long)(dims_out[0]) << " x " <<
//          (unsigned long)(dims_out[1]) << endl;
//      /*
//       * Define hyperslab in the dataset; implicitly giving strike and
//       * block NULL.
//       */
//      hsize_t      offset[2];   // hyperslab offset in the file
//      hsize_t      count[2];    // size of the hyperslab in the file
//      offset[0] = 0;
//      offset[1] = 0;
//      count[0]  = dims_out[0];
//      count[1]  = dims_out[1]/2;
//      dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );

//      /*
//       * Define the memory dataspace.
//       */
//      hsize_t     dimsm[2];              /* memory space dimensions */
//      dimsm[0] = dims_out[0];
//      dimsm[1] = dims_out[1]/2;
//      DataSpace memspace( 2, dimsm );
//      /*
//       * Read data from hyperslab in the file into the hyperslab in
//       * memory and display the data.
//       */
//      Array<1,float> data(dimsm[0]*dimsm[1]);
//      dataset.read( data.get_data_ptr(), PredType::NATIVE_FLOAT, memspace, dataspace);
//      data.release_data_ptr();
//      std::copy(data.begin(), data.end(), efficiency_factors.begin_all());
//  }

  
//#if 1
//   // to test pipe the obtained values into file
//    ofstream out_geom;
//    ofstream out_inter;
//    ofstream out_eff;
//    out_geom.open("geom_out.txt",ios::out);
//    out_inter.open("inter_out.txt",ios::out);
//    out_eff.open("eff_out.txt",ios::out);

//    for ( int i = geometric_factors.get_min_index(); i<=geometric_factors.get_max_index();i++)
//    {
//      for ( int j =geometric_factors[i].get_min_index(); j <=geometric_factors[i].get_max_index(); j++)
//      {
//	 out_geom << geometric_factors[i][j] << "   " ;
//      }
//      out_geom << std::endl;
//    }


//   for ( int i = crystal_interference_factors.get_min_index(); i<=crystal_interference_factors.get_max_index();i++)
//   {
//      for ( int j =crystal_interference_factors[i].get_min_index(); j <=crystal_interference_factors[i].get_max_index(); j++)
//      {
//	 out_inter << crystal_interference_factors[i][j] << "   " ;
//      }
//      out_inter << std::endl;
//   }

//   for ( int i = efficiency_factors.get_min_index(); i<=efficiency_factors.get_max_index();i++)
//   {
//      for ( int j =efficiency_factors[i].get_min_index(); j <=efficiency_factors[i].get_max_index(); j++)
//      {
//	 out_eff << efficiency_factors[i][j] << "   " ;
//      }
//      out_eff << std::endl<< std::endl;
//   }

//#endif

//#if 0
//  display(geometric_factors, "geo");
//  display(efficiency_factors, "eff");
//  display(crystal_interference_factors, "crystal_interference_factors");
//#endif
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

bool 
BinNormalisationFromGEHDF5::
use_crystal_interference_factors() const
{
  return this->_use_crystal_interference_factors;
}

#if 1
float 
BinNormalisationFromGEHDF5::
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
          efficiency_factors[44-pos1.axial_coord()][pos1.tangential_coord()] *
          efficiency_factors[44-pos2.axial_coord()][pos2.tangential_coord()];
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
BinNormalisationFromGEHDF5::get_dead_time_efficiency (const DetectionPosition<>& det_pos,
						    const double start_time,
						    const double end_time) const
{
  if (is_null_ptr(singles_rates_ptr)) {
    return 1;
  }

  return 1;  
}



END_NAMESPACE_STIR

