//
// $Id$
//
/*!

  \file
  \ingroup utilities

  \brief File that writes a ProjMatrixByBin in a sparse ASCII format
  (transposed) or to screen

  \todo very preliminary. Lots of useful things could be asked instead of
  hard-wired.

  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

//#define TOSCREEN

#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/ProjData.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CartesianCoordinate3D.h"

#include "stir/utilities.h"

#include <iostream>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::cout;
using std::endl;
#endif


USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  // TODO parse these?
  // set when you want to see only coords in plane 0

  bool only_plane_zero = false;
  int start_segment_num=0;
  int end_segment_num=0;
  int start_axial_pos_num=0;
  int end_axial_pos_num=45;
  if (argc!=3)
  {
    cerr << "Usage : " << argv[0] << "out_filename sample_proj_data_filename\n";
    exit(EXIT_FAILURE);
  }

#ifdef TOSCREEN
  ostream& output = cout;
#else
  ofstream output(argv[1]);
#endif
#if 0
  const shared_ptr<ProjDataInfo> proj_data_info_ptr =
    ProjDataInfo::ask_parameters();
#else
  const shared_ptr<ProjData> proj_data_ptr =
    ProjData::read_from_file(argv[2]);
  const shared_ptr<ProjDataInfo> proj_data_info_ptr =
    proj_data_ptr->get_proj_data_info_ptr()->clone();
#endif
  const float zoom = ask_num("zoom",0.F,3.F,1.F);
  int xy_size = static_cast<int>(proj_data_info_ptr->get_num_tangential_poss()*zoom);
  xy_size = ask_num("Number of x,y pixels",3,xy_size*2,xy_size);
      
  int z_size = 2*proj_data_info_ptr->get_scanner_ptr()->get_num_rings()-1;
  z_size = ask_num("Number of z pixels",1,1000,z_size);
  const shared_ptr<DiscretisedDensity<3,float> > density_info_ptr =
    new VoxelsOnCartesianGrid<float>(*proj_data_info_ptr,
				     zoom,
				     Coordinate3D<float>(0,0,0),
				     Coordinate3D<int>(z_size,xy_size,xy_size));
  const float z_origin = 
    ask_num("Shift z-origin (in pixels)", 
	    -density_info_ptr->get_length()/2,
	    density_info_ptr->get_length()/2,
             0)*
    dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*density_info_ptr).get_voxel_size().z();
  density_info_ptr->set_origin(Coordinate3D<float>(z_origin,0,0));

  shared_ptr<ProjMatrixByBin> projmatrix_sptr;
  do 
    {
      projmatrix_sptr =
	ProjMatrixByBin::ask_type_and_parameters();
    }
  while (projmatrix_sptr.use_count()==0);
  projmatrix_sptr->set_up(proj_data_info_ptr, density_info_ptr);

  CartesianCoordinate3D<int> min_range, max_range;
  density_info_ptr->get_regular_range(min_range, max_range);
  const int ysize = max_range.y()-min_range.y()+1;
  const int xsize = max_range.x()-min_range.x()+1;
  const int ymin = min_range.y();
  const int xmin = min_range.x();

#ifndef TOSCREEN
  if (start_segment_num!=end_segment_num ||
      start_axial_pos_num != end_axial_pos_num)
    warning("Matrix Size will be wrong because more than 1 segment/axial_pos\n");
  if (!only_plane_zero)
    warning("Column sizes will be wrong because more than 1 plane\n");

  output << proj_data_info_ptr->get_num_views()*proj_data_info_ptr->get_num_tangential_poss()
         << " "
         << xsize*ysize*(only_plane_zero?1:density_info_ptr->get_length())
         << endl;
#endif

  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
    for (int axial_pos_num = std::max(start_axial_pos_num,proj_data_info_ptr->get_min_axial_pos_num(segment_num));
         axial_pos_num <= std::min(end_axial_pos_num,proj_data_info_ptr->get_max_axial_pos_num(segment_num));
         ++axial_pos_num)
  for (int view_num = proj_data_info_ptr->get_min_view_num();
       view_num <= proj_data_info_ptr->get_max_view_num();
       ++view_num)
    for (int tang_pos_num = proj_data_info_ptr->get_min_tangential_pos_num();
         tang_pos_num <= proj_data_info_ptr->get_max_tangential_pos_num();
         ++tang_pos_num)
    {
#ifdef TOSCREEN
      cout << "-------------- segment " << segment_num 
	   << ", view " << view_num 
	   << ", tang_pos " <<tang_pos_num
	   << ", axial_pos " <<axial_pos_num
	   <<"----------------\n";
#endif

      ProjMatrixElemsForOneBin lor;

      projmatrix_sptr->get_proj_matrix_elems_for_one_bin(lor,Bin(segment_num,view_num, axial_pos_num, tang_pos_num));
      // cout << lor.get_number_of_elements() << endl;

#ifdef PBV3D
      // horrible trick relies on knowledge of segment 0 (checked below)
      assert(lor.size()%3==0);
      output << lor.size()/3 << endl;
#else
      output << lor.size() << endl;
#endif
      int elem_count = 0;

      for (ProjMatrixElemsForOneBin::const_iterator lor_elem_iter = lor.begin();
           lor_elem_iter != lor.end();
           ++lor_elem_iter)
      {
         if (only_plane_zero && lor_elem_iter->coord1()!=0)
           continue;
          ++elem_count;
#ifdef TOSCREEN
         cout << lor_elem_iter->coord3() << ", " << lor_elem_iter->coord2() << ", " << lor_elem_iter->coord1()
            << " : " << lor_elem_iter->get_value() << endl;
#else
	 // TODO plane offset
         const int column_num = (lor_elem_iter->coord3() - xmin)*xsize+(lor_elem_iter->coord2() - ymin);
         assert(column_num>=0);
         assert(column_num<xsize*ysize);
         output << column_num
                << " "
                << lor_elem_iter->get_value()
                << endl;
#endif
      }
#ifdef PBV3D
      if (3*elem_count != lor.size())
        warning("Problem in element count at v=%d, tp=%d, expectd %d, actual %d\n",
                view_num, tang_pos_num, lor.size(), 3*elem_count);

#else
      if (static_cast<unsigned>(elem_count) != lor.size())
        warning("Problem in element count at v=%d, tp=%d, expectd %d, actual %d\n",
                view_num, tang_pos_num, lor.size(), elem_count);
#endif
    }


   return EXIT_SUCCESS;
}

