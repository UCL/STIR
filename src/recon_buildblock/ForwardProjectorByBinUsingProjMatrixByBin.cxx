//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief implementations for ForwardProjectorByBinUsingProjMatrixByBin 
   
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/Viewgram.h"
#include "stir/RelatedViewgrams.h"
#include "stir/IndexRange2D.h"
#include <algorithm>
#include <vector>
#include <list>

#ifndef STIR_NO_NAMESPACE
using std::find;
using std::vector;
using std::list;
#endif

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const 
ForwardProjectorByBinUsingProjMatrixByBin::registered_name =
  "Matrix";


void
ForwardProjectorByBinUsingProjMatrixByBin::
set_defaults()
{
  proj_matrix_ptr = 0;
}

void
ForwardProjectorByBinUsingProjMatrixByBin::
initialise_keymap()
{
  parser.add_start_key("Forward Projector Using Matrix Parameters");
  parser.add_stop_key("End Forward Projector Using Matrix Parameters");
  parser.add_parsing_key("matrix type", &proj_matrix_ptr);
}

ForwardProjectorByBinUsingProjMatrixByBin::
ForwardProjectorByBinUsingProjMatrixByBin()
{
  set_defaults();
}

ForwardProjectorByBinUsingProjMatrixByBin::
ForwardProjectorByBinUsingProjMatrixByBin(  
    const shared_ptr<ProjMatrixByBin>& proj_matrix_ptr
    )
  : proj_matrix_ptr(proj_matrix_ptr)
{
  assert(proj_matrix_ptr.use_count()!=0);
}

void
ForwardProjectorByBinUsingProjMatrixByBin::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr)
{    	   
  proj_matrix_ptr->set_up(proj_data_info_ptr, image_info_ptr);
}

const DataSymmetriesForViewSegmentNumbers *
ForwardProjectorByBinUsingProjMatrixByBin::get_symmetries_used() const
{
  return proj_matrix_ptr->get_symmetries_ptr();
}

#if 0
// straightforward version which relies on ProjMatrixByBin to sort out all 
// symmetries
// will be slow if there's no caching
void 
ForwardProjectorByBinUsingProjMatrixByBin::
 actual_forward_project(RelatedViewgrams<float>& viewgrams, 
		  const DiscretisedDensity<3,float>& image,
		  const int min_axial_pos_num, const int max_axial_pos_num,
		  const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  ProjMatrixElemsForOneBin proj_matrix_row;

  RelatedViewgrams<float>::iterator r_viewgrams_iter = viewgrams.begin();

  while( r_viewgrams_iter!=viewgrams.end())
  {
    Viewgram<float>& viewgram = *r_viewgrams_iter;
    const int view_num = viewgram.get_view_num();
    const int segment_num = viewgram.get_segment_num();
    
    for ( int tang_pos = min_tangential_pos_num ;tang_pos  <= max_tangential_pos_num ;++tang_pos)  
      for ( int ax_pos = min_axial_pos_num; ax_pos <= max_axial_pos_num ;++ax_pos)
      { 
	Bin bin(segment_num, view_num, ax_pos, tang_pos, 0);
	proj_matrix_ptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, bin);
	proj_matrix_row.forward_project(bin,image);
        viewgram[ax_pos][tang_pos] = bin.get_bin_value();
      }
    ++r_viewgrams_iter; 
  }
	   
}

#else
// complicated version which handles the symmetries explicitly
// faster when no caching is performed, about just as fast when there is caching
void 
ForwardProjectorByBinUsingProjMatrixByBin::
actual_forward_project(RelatedViewgrams<float>& viewgrams, 
		  const DiscretisedDensity<3,float>& density,
		  const int min_axial_pos_num, const int max_axial_pos_num,
		  const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  ProjMatrixElemsForOneBin proj_matrix_row;
  ProjMatrixElemsForOneBin proj_matrix_row_copy;
  const DataSymmetriesForBins* symmetries = proj_matrix_ptr->get_symmetries_ptr(); 

  Array<2,int> 
    already_processed(IndexRange2D(min_axial_pos_num, max_axial_pos_num,
		                   min_tangential_pos_num, max_tangential_pos_num));

  for ( int tang_pos = min_tangential_pos_num ;tang_pos  <= max_tangential_pos_num ;++tang_pos)  
    for ( int ax_pos = min_axial_pos_num; ax_pos <= max_axial_pos_num ;++ax_pos)
    {       
      if (already_processed[ax_pos][tang_pos])
        continue;          

    Bin basic_bin(viewgrams.get_basic_segment_num(),viewgrams.get_basic_view_num(),ax_pos,tang_pos);
    symmetries->find_basic_bin(basic_bin);
    
    proj_matrix_ptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, basic_bin);

    vector<AxTangPosNumbers> r_ax_poss;
    symmetries->get_related_bins_factorised(r_ax_poss,basic_bin,
                                            min_axial_pos_num, max_axial_pos_num,
	                                    min_tangential_pos_num, max_tangential_pos_num);
    
    for (
#ifndef STIR_NO_NAMESPACES
         std::
#endif
           vector<AxTangPosNumbers>::iterator r_ax_poss_iter = r_ax_poss.begin();
         r_ax_poss_iter != r_ax_poss.end();
         ++r_ax_poss_iter)
    {
       const int axial_pos_tmp = (*r_ax_poss_iter)[1];
       const int tang_pos_tmp = (*r_ax_poss_iter)[2];

       // symmetries might take the ranges out of what the user wants
       if ( !(min_axial_pos_num <= axial_pos_tmp && axial_pos_tmp <= max_axial_pos_num &&
	      min_tangential_pos_num <=tang_pos_tmp  && tang_pos_tmp <= max_tangential_pos_num))
	  continue;

       already_processed[axial_pos_tmp][tang_pos_tmp] = 1;
       

       for (RelatedViewgrams<float>::iterator viewgram_iter = viewgrams.begin();
            viewgram_iter != viewgrams.end();
            ++viewgram_iter)
       {
         Viewgram<float>& viewgram = *viewgram_iter;
         proj_matrix_row_copy = proj_matrix_row;
         Bin bin(viewgram_iter->get_segment_num(),
           viewgram_iter->get_view_num(),
           axial_pos_tmp,
           tang_pos_tmp);
         
         auto_ptr<SymmetryOperation> symm_op_ptr = 
           symmetries->find_symmetry_operation_from_basic_bin(bin);
         assert(bin == basic_bin);
                  
         symm_op_ptr->transform_proj_matrix_elems_for_one_bin(proj_matrix_row_copy);
         proj_matrix_row_copy.forward_project(bin,density);
         
         viewgram[axial_pos_tmp][tang_pos_tmp] = bin.get_bin_value();
      }
      }  
    }      
  assert(already_processed.sum() == (
			       (max_axial_pos_num - min_axial_pos_num + 1) *
			       (max_tangential_pos_num - min_tangential_pos_num + 1)));
	   
}

#endif

END_NAMESPACE_STIR
