/*
    Copyright (C) 2020, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Lesser GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup utilities

  \brief Computes the spatially variant penalty strength

  \par Usage
  \verbatim
     compute_spatially_variant_penalty_strength template_proj_data
  \endverbatim

  Computes a spatially variant penalty strength, either using:

  todo: add methods and documentation
  See Tsai, Y.-J., Schramm, G., Ahn, S., Bousse, A., Arridge, S., Nuyts, J., Hutton, B. F., Stearns, C. W.,
    & Thielemans, K. (2020). Benefits of Using a Spatially-Variant Penalty Strength With Anatomical Priors
    in PET Reconstruction. IEEE Transactions on Medical Imaging, 39(1), 11–22. https://doi.org/10.1109/TMI.2019.2913889
   for more details

  \author Robert Twyman
*/

#include <stir/info.h>
#include <stir/HighResWallClockTimer.h>
#include "stir/IO/OutputFileFormat.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cerr;
#endif



START_NAMESPACE_STIR
    static void print_usage_and_exit()
    {
      //todo:update usage
      std::cerr<<"\nUsage: compute_spatially_variant_penalty_strength template_proj_data\n";
      exit(EXIT_FAILURE);
    }
END_NAMESPACE_STIR

USING_NAMESPACE_STIR
class KappaComputation: public ParsingObject
{
public:
    KappaComputation();
    void set_defaults();
    Succeeded run();
    typedef DiscretisedDensity<3,float> target_type;

protected:
    shared_ptr<GeneralisedObjectiveFunction<target_type> >  objective_function_sptr;

    // This does something to do with saving, ill come back to it
    shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format_sptr;

    Succeeded compute_kappa_at_current_image_estimate();
    Succeeded compute_kappa_with_approximate();

private:
    std::string input_image_filename;
    std::string template_image_filename;
    std::string kappa_filename;
    void initialise_keymap();
    bool post_processing();
};

KappaComputation::KappaComputation()
{
  set_defaults();
}


void
KappaComputation::set_defaults()
{
  objective_function_sptr.reset(new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type>);
}

void
KappaComputation::initialise_keymap()
{
  parser.add_start_key("Kappa Computation Parameters");
  parser.add_key("kappa filename", &kappa_filename);
  parser.add_key("input image", &input_image_filename);
  parser.add_key("template image", &template_image_filename);
  parser.add_parsing_key("objective function type", &objective_function_sptr);
  parser.add_stop_key("End");
}

bool
KappaComputation::post_processing()
{
  if (is_null_ptr(this->objective_function_sptr))
  {
    error("objective_function_sptr is null");
    return true;
  }

  if (input_image_filename.empty() && template_image_filename.empty())
  {
    error("Requires either input_image_filename or template_image_filename");
    return true;
  }
  return false;
}

Succeeded
KappaComputation::run()
{
  if (!input_image_filename.empty())
  {
    compute_kappa_at_current_image_estimate();
  }
  else if (!template_image_filename.empty())
  {
    compute_kappa_with_approximate();
  }
  return Succeeded::yes;
}


Succeeded
KappaComputation::compute_kappa_at_current_image_estimate()
{
  info("Computing the spatially variant penalty strength at the current image estimate.");
  shared_ptr<DiscretisedDensity<3,float>>
          current_image_estimate(read_from_file<DiscretisedDensity<3,float>>(input_image_filename));
  current_image_estimate->get_data_ptr();

  shared_ptr<DiscretisedDensity<3, float>> output_image;
  output_image = static_cast<const shared_ptr<DiscretisedDensity<3, float>>>(current_image_estimate->get_empty_copy());
  output_image->fill(0.);

  shared_ptr<DiscretisedDensity<3, float>> ones_image;
  ones_image = static_cast<const shared_ptr<DiscretisedDensity<3, float>>>(current_image_estimate->get_empty_copy());
  ones_image->fill(1.);

  objective_function_sptr->set_up(current_image_estimate);
  objective_function_sptr->accumulate_Hessian_times_input(*output_image, *current_image_estimate, *ones_image);
  output_file_format_sptr->write_to_file(kappa_filename, *output_image);
  return Succeeded::yes;
}

Succeeded
KappaComputation::compute_kappa_with_approximate()
{

  info("Computing the spatially variant penalty strength using approximate hessian.");
  shared_ptr<DiscretisedDensity<3,float>>
          output_image(read_from_file<DiscretisedDensity<3,float>>(input_image_filename));
  output_image->fill(0.);

  shared_ptr<DiscretisedDensity<3, float>> ones_image;
  ones_image = static_cast<const shared_ptr<DiscretisedDensity<3, float>>>(output_image->get_empty_copy());
  ones_image->fill(1.);

  objective_function_sptr->add_multiplication_with_approximate_Hessian(*output_image, *ones_image);
  output_file_format_sptr->write_to_file(kappa_filename, *output_image);
  return Succeeded::yes;
}

int
main (int argc, char * argv[])
{

  KappaComputation kappa_computer;

  if (argc!=2)
    print_usage_and_exit();
  else
    kappa_computer.parse(argv[1]);

  if (kappa_computer.run() == Succeeded::no)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
