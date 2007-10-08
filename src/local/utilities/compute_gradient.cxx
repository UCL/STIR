//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
  \ingroup utilities
 
  \brief 


  \author Kris Thielemans
  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/
#include "stir/shared_ptr.h"
#include "stir/KeyParser.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"


/*
example
compute gradient parameters:=
objective function type:= PoissonLogLikelihoodWithLinearModelForMeanAndProjData
PoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters:=

input file := Utahscat600k_ca_seg4.hs
zero end planes of segment 0:= 0
projector pair type := Matrix
  Projector Pair Using Matrix Parameters :=
  Matrix type := Ray Tracing
  Ray tracing matrix parameters :=
   number of rays in tangential direction to trace for each bin := 2
  End Ray tracing matrix parameters :=
  End Projector Pair Using Matrix Parameters :=

; needed, but not used at present
; so use same as input image
sensitivity filename:= .hv

end PoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters:=

input filename:=
output filename:=
end:=

compute gradient parameters:=
objective function type:= PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin
PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin Parameters:=

  list mode filename:=
  Matrix type:= Ray Tracing
  Ray tracing matrix parameters :=
   number of rays in tangential direction to trace for each bin := 2
  End Ray tracing matrix parameters :=
  End Projector Pair Using Matrix Parameters :=
  frame definitions filename:=
  // always 1
  current time frame:=1

; needed, but not used at present
; so use same as input image
sensitivity filename:= .hv

End PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin Parameters:=


input filename:=
output filename:=
end:=

*/
int main(int argc, char **argv)
{
  USING_NAMESPACE_STIR;
  typedef DiscretisedDensity<3,float> data_type;

  shared_ptr<GeneralisedObjectiveFunction<data_type > 
    obj_function_sptr;
  std::string input_filename;
  std::string output_filename;
  shared_ptr<data_type > density_sptr, gradient_sptr;

   KeyParser parser;
   parser.add_start_key("compute gradient parameters");
   parser.add_parsing_key("objective function type", &obj_function_sptr);
   parser.add_key("input filename", &input_filename);
   parser.add_key("output filename", &output_filename);
   parser.add_stop_key("END"); 
   if (parser.parse(argv[1]) == false || is_null_ptr(obj_function_sptr))
     {
       std::cerr << "Error parsing output file format from " << argv[1]<<endl;
       exit(EXIT_FAILURE); 
     }

   density_sptr = data_type::read_from_file(input_filename);

   if (obj_function_sptr->set_up(density_sptr) != Succeeded::yes)
     {
       error();
     }

   PoissonLogLikelihoodWithLinearModelForMean<data_type >&
     obj_func =
     dynamic_cast<PoissonLogLikelihoodWithLinearModelForMean<data_type >&>
     (*obj_function_sptr);

   gradient_sptr = density_sptr->get_empty_copy();

   obj_func.compute_sub_gradient_without_penalty_plus_sensitivity(*gradient_sptr,
								  *density_sptr,
								  0);

   OutputFileFormat<data_type>::default_sptr()->
     write_to_file(*gradient_sptr, output_filename);

   return EXIT_SUCCESS;
}
