
//
//

/*!
  \file
  \ingroup projection

  \brief non-inline implementations for stir::BackProjectorByBinUsingProjMatrixByBin
  
  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project
    
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/* History:
   20/09/2001 KT
   - added registry and parsing 
   22/01/2002 KT
   - used new implementation for actual_backproject that takes 
     symmetries into account for faster performance. Essentially copied
     from ForwardProjectorByBinUsingProjMatrixByBin
*/
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/Viewgram.h"
#include "stir/RelatedViewgrams.h"
#include "stir/is_null_ptr.h"

using std::vector;

START_NAMESPACE_STIR

const char * const 
BackProjectorByBinUsingProjMatrixByBin::registered_name =
  "Matrix";


void
BackProjectorByBinUsingProjMatrixByBin::
set_defaults()
{
  this->proj_matrix_ptr.reset();
  //BackProjectorByBin::set_defaults();
}

void
BackProjectorByBinUsingProjMatrixByBin::
initialise_keymap()
{
  parser.add_start_key("Back Projector Using Matrix Parameters");
  parser.add_stop_key("End Back Projector Using Matrix Parameters");
  parser.add_parsing_key("matrix type", &proj_matrix_ptr);
  //BackProjectorByBin::initialise_keymap();
}


bool
BackProjectorByBinUsingProjMatrixByBin::
post_processing()
{
  //if (BackProjectorByBin::post_processing() == true)
  //  return true;
  if (is_null_ptr(proj_matrix_ptr))
  { 
    warning("BackProjectorByBinUsingProjMatrixByBin: matrix not set.\n");
    return true;
  }
  return false;
}

BackProjectorByBinUsingProjMatrixByBin::
BackProjectorByBinUsingProjMatrixByBin()
{
  set_defaults();
}

BackProjectorByBinUsingProjMatrixByBin::
BackProjectorByBinUsingProjMatrixByBin(  
    const shared_ptr<ProjMatrixByBin>& proj_matrix_ptr
    )		   
    : proj_matrix_ptr(proj_matrix_ptr)
{
  assert(!is_null_ptr(proj_matrix_ptr));	 
}

void
BackProjectorByBinUsingProjMatrixByBin::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr)

{    	   
  proj_matrix_ptr->set_up(proj_data_info_ptr, image_info_ptr);
}

const DataSymmetriesForViewSegmentNumbers *
BackProjectorByBinUsingProjMatrixByBin::get_symmetries_used() const
{
  return proj_matrix_ptr->get_symmetries_ptr();
}

void 
BackProjectorByBinUsingProjMatrixByBin::
actual_back_project(DiscretisedDensity<3,float>& image,
		    const RelatedViewgrams<float>& viewgrams,
		    const int min_axial_pos_num, const int max_axial_pos_num,
		    const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  if (proj_matrix_ptr->is_cache_enabled()/* &&
					    !proj_matrix_ptr->does_cache_store_only_basic_bins()*/)
    {
      // straightforward version which relies on ProjMatrixByBin to sort out all 
      // symmetries
      // would be slow if there's no caching at all, but is very fast if everything is cached

      ProjMatrixElemsForOneBin proj_matrix_row;
  
      RelatedViewgrams<float>::const_iterator r_viewgrams_iter = viewgrams.begin();
  
      while( r_viewgrams_iter!=viewgrams.end())
	{
	  const Viewgram<float>& viewgram = *r_viewgrams_iter;
	  const int view_num = viewgram.get_view_num();
	  const int segment_num = viewgram.get_segment_num();
    
	  for ( int tang_pos = min_tangential_pos_num ;tang_pos  <= max_tangential_pos_num ;++tang_pos)  
	    for ( int ax_pos = min_axial_pos_num; ax_pos <= max_axial_pos_num ;++ax_pos)
	      { 
		// KT 21/02/2002 added check on 0
		if (viewgram[ax_pos][tang_pos] == 0)
		  continue;
		Bin bin(segment_num, view_num, ax_pos, tang_pos, viewgram[ax_pos][tang_pos]);
		proj_matrix_ptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, bin);
		proj_matrix_row.back_project(image, bin);
	      }
	  ++r_viewgrams_iter;   
	}
    }  
  else
    {
      // complicated version which handles the symmetries explicitly
      // faster when no caching is performed, about just as fast when there is caching
      ProjMatrixElemsForOneBin proj_matrix_row;
      ProjMatrixElemsForOneBin proj_matrix_row_copy;
      const DataSymmetriesForBins* symmetries = proj_matrix_ptr->get_symmetries_ptr(); 

      Array<2,int> 
	already_processed(IndexRange2D(min_axial_pos_num, max_axial_pos_num,
				       min_tangential_pos_num, max_tangential_pos_num));

      vector<AxTangPosNumbers> related_ax_tang_poss;
      for ( int tang_pos = min_tangential_pos_num ;tang_pos  <= max_tangential_pos_num ;++tang_pos)  
	for ( int ax_pos = min_axial_pos_num; ax_pos <= max_axial_pos_num ;++ax_pos)
	  {       
	    if (already_processed[ax_pos][tang_pos])
	      continue;          

	    Bin basic_bin(viewgrams.get_basic_segment_num(),
			  viewgrams.get_basic_view_num(),
			  ax_pos,
			  tang_pos);
	    symmetries->find_basic_bin(basic_bin);
    
	    proj_matrix_ptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, basic_bin);
      
	    related_ax_tang_poss.resize(0);
	    symmetries->get_related_bins_factorised(related_ax_tang_poss,basic_bin,
						    min_axial_pos_num, max_axial_pos_num,
						    min_tangential_pos_num, max_tangential_pos_num);
    
	    for (
#ifndef STIR_NO_NAMESPACES
		 std::
#endif
		   vector<AxTangPosNumbers>::const_iterator r_ax_tang_poss_iter = related_ax_tang_poss.begin();
		 r_ax_tang_poss_iter != related_ax_tang_poss.end();
		 ++r_ax_tang_poss_iter)
	      {
		const int axial_pos_tmp = (*r_ax_tang_poss_iter)[1];
		const int tang_pos_tmp = (*r_ax_tang_poss_iter)[2];
	  
		// symmetries might take the ranges out of what the user wants
		if ( !(min_axial_pos_num <= axial_pos_tmp && axial_pos_tmp <= max_axial_pos_num &&
		       min_tangential_pos_num <=tang_pos_tmp  && tang_pos_tmp <= max_tangential_pos_num))
		  continue;
	  
		already_processed[axial_pos_tmp][tang_pos_tmp] = 1;
       
	  
		for (RelatedViewgrams<float>::const_iterator viewgram_iter = viewgrams.begin();
		     viewgram_iter != viewgrams.end();
		     ++viewgram_iter)
		  {
		    // KT 21/02/2002 added check on 0
		    if ((*viewgram_iter)[axial_pos_tmp][tang_pos_tmp] == 0)
		      continue;
		    proj_matrix_row_copy = proj_matrix_row;
		    Bin bin(viewgram_iter->get_segment_num(),
			    viewgram_iter->get_view_num(),
			    axial_pos_tmp,
			    tang_pos_tmp,
			    (*viewgram_iter)[axial_pos_tmp][tang_pos_tmp]);
	      
		    std::unique_ptr<SymmetryOperation> symm_op_ptr = 
		      symmetries->find_symmetry_operation_from_basic_bin(bin);
		    // TODO replace with Bin::compare_coordinates or so
		    assert(bin.segment_num() == basic_bin.segment_num());
		    assert(bin.view_num() == basic_bin.view_num());
		    assert(bin.axial_pos_num() == basic_bin.axial_pos_num());
		    assert(bin.tangential_pos_num() == basic_bin.tangential_pos_num());
	      
		    symm_op_ptr->transform_proj_matrix_elems_for_one_bin(proj_matrix_row_copy);
		    proj_matrix_row_copy.back_project(image, bin);
		  }
	      }  
	  }      
      assert(already_processed.sum() 
	     == (
		 (max_axial_pos_num - min_axial_pos_num + 1) *
		 (max_tangential_pos_num - min_tangential_pos_num + 1)));
    }  
}



END_NAMESPACE_STIR
