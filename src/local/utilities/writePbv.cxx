//
// $Id$
//
/*!

  \file
  \ingroup utilities

  \brief File that writes a ProjMatrixByBin in a sparse ASCII format

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
      
  const shared_ptr<DiscretisedDensity<3,float> > density_info_ptr =
     new VoxelsOnCartesianGrid<float>(*proj_data_info_ptr,
                                         zoom,
                                         CartesianCoordinate3D<float>(0,0,0),
                                         CartesianCoordinate3D<int>(-1,xy_size,xy_size));

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
  output << proj_data_info_ptr->get_num_views()*proj_data_info_ptr->get_num_tangential_poss()
         << " "
         << xsize*ysize
         << endl;
#endif

  for (int view_num = proj_data_info_ptr->get_min_view_num();
       view_num <= proj_data_info_ptr->get_max_view_num();
       ++view_num)
    for (int tang_pos_num = proj_data_info_ptr->get_min_tangential_pos_num();
         tang_pos_num <= proj_data_info_ptr->get_max_tangential_pos_num();
         ++tang_pos_num)
    {
#ifdef TOSCREEN
      cout << "-------------- view " << view_num << ", tang_pos " <<tang_pos_num<<"----------------\n";
#endif

      ProjMatrixElemsForOneBin lor;

      projmatrix_sptr->get_proj_matrix_elems_for_one_bin(lor,Bin(0,view_num, 0, tang_pos_num));
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
         if (lor_elem_iter->coord1()!=0)
           continue;
          ++elem_count;
#ifdef TOSCREEN
         cout << lor_elem_iter->coord3() << ", " << lor_elem_iter->coord2() << ", " << lor_elem_iter->coord1()
            << " : " << lor_elem_iter->get_value() << endl;
#else
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

