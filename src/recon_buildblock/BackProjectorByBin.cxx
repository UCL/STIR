//
// $Id$: $Date$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for BackProjectorByBin

  \author Kris Thielemans
  \author PARAPET project
  
  \date $Date$

  \version $Revision$
*/

#include "recon_buildblock/BackProjectorByBin.h"
#include "RelatedViewgrams.h"
#include "DiscretisedDensity.h"
//#include "Viewgram.h"

START_NAMESPACE_TOMO

#if 0

BackProjectorByBin::BackProjectorByBin()
{
}


void 
BackProjectorByBin::back_project( RelatedViewgrams<float>& viewgrams, 
		     const DiscretisedDensity<3,float>& image)
{
  back_project(viewgrams,
              image,
              viewgrams.get_min_axial_pos_num(),
	      viewgrams.get_max_axial_pos_num(),
	      viewgrams.get_min_tangential_pos_num(),
	      viewgrams.get_max_tangential_pos_num());
}

void BackProjectorByBin::back_project
 ( RelatedViewgrams<float>& viewgrams, 
   const DiscretisedDensity<3,float>& image,
   const int min_axial_pos_num, 
   const int max_axial_pos_num)
{
  back_project(viewgrams, image,
             min_axial_pos_num,
	     max_axial_pos_num,
	     viewgrams.get_min_tangential_pos_num(),
	     viewgrams.get_max_tangential_pos_num());
}


void BackProjectorByBin::back_project
  (RelatedViewgrams<float>& viewgrams, 
		     const DiscretisedDensity<3,float>& density,
		     const int min_axial_pos_num, const int max_axial_pos_num,
		     const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  actual_back_project(viewgrams, density,
             min_axial_pos_num,
	     max_axial_pos_num,
	     min_tangential_pos_num,
	     max_tangential_pos_num);
}

#endif

END_NAMESPACE_TOMO
