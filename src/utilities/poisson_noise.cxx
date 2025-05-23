/*!
  \file
  \ingroup utilities 
  \brief Generates a noise realisation according to Poisson statistics for  
  some projection data

  \author Kris Thielemans
  \author Sanida Mustafovic

  Usage:
  \code
  poisson_noise [-p | --preserve-mean] \
        output_filename input_projdata_filename \
        scaling_factor seed-unsigned-int
  \endcode
  The \c scaling_factor is used to multiply the input data before generating
  the Poisson random number. This means that a \c scaling_factor larger than 1
  will result in less noisy data.<br>
  The seed value for the random number generator has to be strictly positive.<br>
  Without the -p option, the mean of the output data will
  be equal to <tt>scaling_factor*mean_of_input</tt>, otherwise it
  will be equal to mean_of_input.<br>
  The options -p and --preserve-mean are identical.
*/
/*
    Copyright (C) 2000 - 2004, Hammersmith Imanet Ltd
    Copyright (C) 2017, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/GeneralisedPoissonNoiseGenerator.h"
#include "stir/ProjDataInterfile.h"

USING_NAMESPACE_STIR

void usage()
{
    using std::cerr;
    cerr <<"Usage: poisson_noise [-p | --preserve-mean] <output_filename (no extension)> <input_projdata_filename> scaling_factor seed-unsigned-int\n"
         <<"The seed value for the random number generator has to be strictly positive.\n"
         << "Without the -p option, the mean of the output data will"
	 << " be equal to\nscaling_factor*mean_of_input, otherwise it"
	 << "will be equal to mean_of_input.\n"
	 << "The options -p and --preserve-mean are identical.\n";
}

int
main (int argc,char *argv[])
{
  if(argc<5)
  {
    usage();
    return(EXIT_FAILURE);
  }  
  
  bool preserve_mean = false;

  // option processing
  if (argv[1][0] == '-')
    {
      if (strcmp(argv[1],"-p")==0 ||
	  strcmp(argv[1],"--preserve-mean")==0)
	preserve_mean = true;
      else
	{
	  usage();
	  return(EXIT_FAILURE);
	}  
      ++argv;
    }
	  
  const char *const filename = argv[1];
  const float scaling_factor = static_cast<float>(atof(argv[3]));
  shared_ptr<ProjData>  in_data = ProjData::read_from_file(argv[2]);

  unsigned int seed = atoi(argv[4]);

  GeneralisedPoissonNoiseGenerator generator(scaling_factor, preserve_mean);
  generator.seed(seed);

  ProjDataInterfile new_data(in_data->get_exam_info_sptr(),in_data->get_proj_data_info_sptr()->create_shared_clone(), filename);

  
  generator.generate_random(new_data,*in_data);
  
  return EXIT_SUCCESS;
}



