/*
    Copyright (C) 2014, University College London
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
/*!
  \file
  \ingroup mex
  \brief a Mex function for forward projection

  \author Kris Thielemans
*/

#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/IO/read_from_file.h"
#include "stir/ProjDataInMemory.h"
#include "stir/SegmentByView.h"
#include "stir/DiscretisedDensity.h"

#include "mex.h"
#include <exception>
#include <string>

using namespace stir;

/* Usage:
   out=stir_forward_project(image_filename, template_projdata_filename);

   out will be a 1D array containing all projection data
*/

void mexFunction(
		 int          nlhs,
		 mxArray      *prls[],
		 int          nrhs,
		 const mxArray *prhs[]
		 )
{

  /* Check for proper number of arguments */

  if (nrhs != 2) {
    mexErrMsgIdAndTxt("MATLAB:stir_forward_project:nargin", 
            "Usage: out=stir_forward_project(image_filename, template_projdata_filename);");
  } else if (nlhs != 1) {
    mexErrMsgIdAndTxt("MATLAB:stir_forward_project:nargout",
            "stir_forward_project requires 1 output argument.");
  }
  if ( mxIsChar(prhs[0]) != 1)
    mexErrMsgIdAndTxt( "MATLAB:stir_forward_project:inputNotString",
                       "First argument must be a string.");
  if ( mxIsChar(prhs[1]) != 1)
    mexErrMsgIdAndTxt( "MATLAB:stir_forward_project:inputNotString",
                       "Second argument must be a string.");


  char const * const image_filename = mxArrayToString(prhs[0]);
  char const * const projdata_filename = mxArrayToString(prhs[1]);

  
  try
    {
      shared_ptr <DiscretisedDensity<3,float> > 
        image_density_ptr(read_from_file<DiscretisedDensity<3,float> >(image_filename));
      
      shared_ptr<ProjData> template_proj_data_ptr = 
        ProjData::read_from_file(projdata_filename);

      shared_ptr<ForwardProjectorByBin> forw_projector_ptr;
      shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
      forw_projector_ptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM)); 

      forw_projector_ptr->set_up(template_proj_data_ptr->get_proj_data_info_ptr()->create_shared_clone(),
                                 image_density_ptr );

      ProjDataInMemory output_projdata(template_proj_data_ptr->get_exam_info_sptr(),
                                       template_proj_data_ptr->get_proj_data_info_ptr()->create_shared_clone());

      forw_projector_ptr->forward_project(output_projdata, *image_density_ptr);


      // find projdata-size (a bit awkward at the moment)
      std::size_t projdata_size(0);
      for (int segment_num=output_projdata.get_min_segment_num();
           segment_num<=output_projdata.get_max_segment_num();
           ++segment_num)
        {
          const SegmentByView<float> seg(output_projdata.get_empty_segment_by_view(segment_num));
          projdata_size += seg.size_all();
        }

      // create output array 
      mxArray * mat_array = mxCreateNumericMatrix(projdata_size, 1, mxSINGLE_CLASS,mxREAL);

	  // copy data from STIR to matlab output array
	  float *destination = (float *)mxGetPr(mat_array);
      for (int segment_num=output_projdata.get_min_segment_num();
           segment_num<=output_projdata.get_max_segment_num();
           ++segment_num)
        {
          const SegmentByView<float> seg(output_projdata.get_segment_by_view(segment_num));
          std::copy(seg.begin_all(), seg.end_all(), destination);
          destination += seg.size_all();
        }

      // finally, store mat_array into prls
      prls[0] = mat_array;

      return;
    }

  catch (std::string& error_string)
    {
      mexErrMsgIdAndTxt("MATLAB:stir_forward_project:run_time",
                        error_string.c_str());
    }
  catch (std::exception& e)
    {
      mexErrMsgIdAndTxt("MATLAB:stir_forward_project:run_time",
                        e.what());
    }
}      
