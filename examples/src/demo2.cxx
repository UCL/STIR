//
//
/*!
  \file
  \ingroup examples
  \brief A small modification of demo1.cxx to ask the user for the
	back projector she wants to use.

  It illustrates
	- how to ask the user for objects for which different types
	  exist (e.g. back-projector, forward-projectors, image processors 
	  etc), anything based on the RegisteredObject hierarchy.
	- that STIR is able to select basic processing units at run-time
	- how to use the (very) basic display facilities in STIR

  See README.txt in the directory where this file is located.

  \author Kris Thielemans      
*/
/*
    Copyright (C) 2004- 2012, Hammersmith Imanet Ltd

    This software is distributed under the terms 
    of the GNU General  Public Licence (GPL)
    See STIR/LICENSE.txt for details
*/
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/IO/write_to_file.h"
#include "stir/IO/read_from_file.h"
#include "stir/ProjData.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/display.h"

int main()
{
  using namespace stir;
 
  /////////////// input sinogram
  const std::string input_filename =
    ask_filename_with_extension("Input file",".hs");

  shared_ptr<ProjData> 
    proj_data_sptr(ProjData::read_from_file(input_filename));
  shared_ptr<ProjDataInfo> 
    proj_data_info_sptr(proj_data_sptr->get_proj_data_info_ptr()->clone());

  /////////////// template image (for sizes etc)
  const std::string template_filename =
    ask_filename_with_extension("Template image file",".hv");

  shared_ptr<DiscretisedDensity<3,float> > 
    density_sptr(read_from_file<DiscretisedDensity<3,float> >(template_filename));

  density_sptr->fill(0);

  /////////////// back project
  shared_ptr<BackProjectorByBin> back_projector_sptr
    (BackProjectorByBin::ask_type_and_parameters());

  back_projector_sptr->set_up(proj_data_info_sptr, density_sptr);

  back_projector_sptr->back_project(*density_sptr, *proj_data_sptr);

  /////////////// output
  write_to_file("output", *density_sptr);

  return EXIT_SUCCESS;
}
