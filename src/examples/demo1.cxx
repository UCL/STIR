//
// $Id$
//
/*!
  \file
  \ingroup examples
  \brief A simple program that backprojects some projection data.

  It illustrates
	- basic interaction with the user,
	- reading of images and projection data
	- construction of a specified type of back-projector,
	- how to use back-project all projection data
	- output of images

  See examples/README.txt

  \author Kris Thielemans      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd

    This software is distributed under the terms 
    of the GNU General  Public Licence (GPL)
    See STIR/LICENSE.txt for details
*/
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/ProjData.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"

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
  BackProjectorByBinUsingInterpolation back_projector;

  back_projector.set_up(proj_data_info_sptr, density_sptr);

  back_projector.back_project(*density_sptr, *proj_data_sptr);

  /////////////// output
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
      write_to_file("output", *density_sptr);

  return EXIT_SUCCESS;
}
