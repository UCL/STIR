//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Inline implementations for ForwardProjectorByBin

  \author Kris Thielemans
  \author Sanida Mustafovic
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


#if 0
ForwardProjectorByBin::ForwardProjectorByBin()
{
}

void 
ForwardProjectorByBin::forward_project(RelatedViewgrams<float>& viewgrams, 
				 const DiscretisedDensity<3,float>& image)
{
  forward_project(viewgrams, image,
                  viewgrams.get_min_axial_pos_num(),
		  viewgrams.get_max_axial_pos_num(),
		  viewgrams.get_min_tangential_pos_num(),
		  viewgrams.get_max_tangential_pos_num());
}

void ForwardProjectorByBin::forward_project
  (RelatedViewgrams<float>& viewgrams, 
   const DiscretisedDensity<3,float>& image,
   const int min_axial_pos_num, 
   const int max_axial_pos_num)
{
  forward_project(viewgrams, image,
             min_axial_pos_num,
	     max_axial_pos_num,
	     viewgrams.get_min_tangential_pos_num(),
	     viewgrams.get_max_tangential_pos_num());
}

void 
ForwardProjectorByBin::
forward_project(RelatedViewgrams<float>& viewgrams, 
		     const DiscretisedDensity<3,float>& density,
		     const int min_axial_pos_num, const int max_axial_pos_num,
		     const int min_tangential_pos_num, const int max_tangential_pos_num)
{
  actual_forward_project(viewgrams, density,
             min_axial_pos_num,
	     max_axial_pos_num,
	     min_tangential_pos_num,
	     max_tangential_pos_num);
}

#endif

END_NAMESPACE_STIR
