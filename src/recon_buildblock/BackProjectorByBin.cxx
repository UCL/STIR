//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for BackProjectorByBin

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

#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/RelatedViewgrams.h"
#include "stir/DiscretisedDensity.h"
//#include "stir/Viewgram.h"

START_NAMESPACE_STIR

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

END_NAMESPACE_STIR
