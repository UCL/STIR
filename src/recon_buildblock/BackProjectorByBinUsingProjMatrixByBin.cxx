
//
// $Id$: $Date$
//

/*!
  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for BackProjectorByBinUsingProjMatrixByBin
  
  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project
    
  \date $Date$
  \version $Revision$
*/


#include  "recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "Viewgram.h"
#include "RelatedViewgrams.h"

START_NAMESPACE_TOMO

BackProjectorByBinUsingProjMatrixByBin::
BackProjectorByBinUsingProjMatrixByBin(  
    const shared_ptr<ProjMatrixByBin>& proj_matrix_ptr
    )		   
    : proj_matrix_ptr(proj_matrix_ptr)
  {
     assert(proj_matrix_ptr.use_count()!=0);	 
    
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
	Bin bin(segment_num, view_num, ax_pos, tang_pos, viewgram[ax_pos][tang_pos]);
	proj_matrix_ptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, bin);
	proj_matrix_row.back_project(image, bin);
      }
     ++r_viewgrams_iter;   
  }
	   
}

END_NAMESPACE_TOMO
