//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Implementations of inline functions for class BackProjectorByBin

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
#include "stir/RelatedViewgrams.h"

START_NAMESPACE_STIR

BackProjectorByBin::BackProjectorByBin()
{
}

void 
BackProjectorByBin::back_project( DiscretisedDensity<3,float>& image,
				  const RelatedViewgrams<float>& viewgrams)
{
  back_project(image,viewgrams,
                  viewgrams.get_min_axial_pos_num(),
		  viewgrams.get_max_axial_pos_num(),
		  viewgrams.get_min_tangential_pos_num(),
		  viewgrams.get_max_tangential_pos_num());
}

void BackProjectorByBin::back_project
  (DiscretisedDensity<3,float>& image,
   const RelatedViewgrams<float>& viewgrams,    
   const int min_axial_pos_num, 
   const int max_axial_pos_num)
{
  back_project(image,viewgrams,
             min_axial_pos_num,
	     max_axial_pos_num,
	     viewgrams.get_min_tangential_pos_num(),
	     viewgrams.get_max_tangential_pos_num());
}

void 
BackProjectorByBin::
back_project(DiscretisedDensity<3,float>& density,
	     const RelatedViewgrams<float>& viewgrams, 		    
	     const int min_axial_pos_num, const int max_axial_pos_num,
	     const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  actual_back_project(density,viewgrams,
             min_axial_pos_num,
	     max_axial_pos_num,
	     min_tangential_pos_num,
	     max_tangential_pos_num);
}


END_NAMESPACE_STIR
