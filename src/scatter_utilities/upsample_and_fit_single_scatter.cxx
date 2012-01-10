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
  \ingroup scatter
  \brief upsample a coarse scatter estimate and fit it to emission data using a weight/mask projection data

  \see ScatterEstimationByBin::upsample_and_fit_scatter_estimate

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
  $Date$
  $Revision$
        
  \par Usage:
  \code
  [--min-scale-factor <number>]
  [--max-scale-factor <number>]
  [--remove-interleaving <1|0>]
  [--half-filter-width <number>]
  --output-filename <filename> 
  --data-to-fit <filename>
  --data-to-scale <filename>
  --weights <filename>
  \endcode
  \a remove_interleaving defaults to 1\, \a min-scale-factor to 1e-5, 
  and \a max-scale-factor to 1e5.
*/

#include <iostream>
#include <string>
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInterfile.h"
#include "stir/scatter/ScatterEstimationByBin.h"
#include "stir/Succeeded.h"
/***********************************************************/     

static void
print_usage_and_exit(const char * const prog_name)
{
  std::cerr << "\nUsage:\n" << prog_name << "\\\n"
            << "\t[--min-scale-factor <number>]\\\n"
            << "\t[--max-scale-factor <number>]\\\n"
            << "\t[--remove-interleaving <1|0>]\\\n"
            << "\t[--half-filter-width <number>]\\\n"
            << "\t--output-filename <filename>\\\n" 
            << "\t--data-to-fit <filename>\\\n"
            << "\t--data-to-scale <filename>\\\n"
            << "\t--weights <filename>\n"
            << "remove_interleaving defaults to 1\n"
            << "min-scale-factor to 1e-5, and max scale factor to 1e5\n";
  exit(EXIT_FAILURE);
}

int main(int argc, const char *argv[])                                  
{         
  const char * const prog_name = argv[0];

  float min_scale_factor = 1.E-5F;
  float max_scale_factor = 1.E5F;
  bool remove_interleaving = true;
  unsigned int half_filter_width = 1;
  stir::BSpline::BSplineType  spline_type = stir::BSpline::linear;
  std::string data_to_fit_filename;
  std::string data_to_scale_filename;
  std::string weights_filename;
  std::string output_filename;

  // option processing
  while (argc>1 && argv[1][1] == '-')
    {
      if (strcmp(argv[1], "--min-scale-factor")==0)
        {
          min_scale_factor = atof(argv[2]);
          argc-=2; argv +=2;
        }
      else if (strcmp(argv[1], "--max-scale-factor")==0)
        {
          max_scale_factor = atof(argv[2]);
          argc-=2; argv +=2;
        }
      else if (strcmp(argv[1], "--BSpline-type")==0)
        {
          spline_type = (stir::BSpline::BSplineType)atoi(argv[2]);
          argc-=2; argv +=2;
        }
      else if (strcmp(argv[1], "--half-filter-width")==0)
        {
          half_filter_width = (unsigned)atoi(argv[2]);
          argc-=2; argv +=2;
        }
      else if (strcmp(argv[1], "--remove-interleaving")==0)
        {
          remove_interleaving = atoi(argv[2]);
          argc-=2; argv +=2;
        }
      else if (strcmp(argv[1], "--output-filename")==0)
        {
          output_filename = (argv[2]);
          argc-=2; argv +=2;
        }
      else if (strcmp(argv[1], "--data-to-fit")==0)
        {
          data_to_fit_filename = argv[2];
          argc-=2; argv +=2;
        }
      else if (strcmp(argv[1], "--data-to-scale")==0)
        {
          data_to_scale_filename = argv[2];
          argc-=2; argv +=2;
        }
      else if (strcmp(argv[1], "--weights")==0)
        {
          weights_filename = argv[2];
          argc-=2; argv +=2;
        }
      else
        {
          std::cerr << "\nUnknown option: " << argv[1];
          print_usage_and_exit(prog_name);
        }
    }
    
  if (argc> 1)
    {
      std::cerr << "Command line should contain only options\n";
      print_usage_and_exit(prog_name);
    }      

  if (data_to_fit_filename.size()==0 || data_to_scale_filename.size() == 0 ||
      weights_filename.size()==0 || output_filename.size()==0)
    {
      std::cerr << "One of the required filenames has not been specified\n";
      print_usage_and_exit(prog_name);
    }

  using stir::ProjData;
  const stir::shared_ptr< ProjData > weights_proj_data_sptr = 
    ProjData::read_from_file(weights_filename);
  const stir::shared_ptr<ProjData> data_to_fit_proj_data_sptr = 
    ProjData::read_from_file(data_to_fit_filename);
  const stir::shared_ptr<ProjData> data_to_scale_proj_data_sptr = 
    ProjData::read_from_file(data_to_scale_filename);   
        
  stir::shared_ptr<stir::ProjDataInfo> data_to_fit_proj_data_info_sptr =
    data_to_fit_proj_data_sptr->get_proj_data_info_ptr()->create_shared_clone();
  
  stir::ProjDataInterfile output_proj_data(data_to_fit_proj_data_info_sptr, output_filename);
        
  stir::ScatterEstimationByBin::
    upsample_and_fit_scatter_estimate(output_proj_data,
                                      *data_to_fit_proj_data_sptr,
                                      *data_to_scale_proj_data_sptr,
                                      *weights_proj_data_sptr,
                                      min_scale_factor,
                                      max_scale_factor,
                                      half_filter_width,
                                      spline_type,
                                      remove_interleaving);
  return EXIT_SUCCESS;
}                 
