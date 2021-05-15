/*!
  \file
  \ingroup examples
  \brief An example of a method to compute a line search evaluation in the direction of the gradient of a given image.
  All parameters can be parsed from a parameter file.

  It illustrates
    - TODO

  Note that the same functionality could be provided without deriving
  a new class from stir::ParsingObject. One could have a stir::KeyParser object
  in main() and fill it in directly.

  See README.txt in the directory where this file is located.

  \author Robert Twyman
*/
/*
    Copyright (C) 2021 University College London

    This software is distributed under the terms
    of the GNU General  Public Licence (GPL)
    See STIR/LICENSE.txt for details
*/

#include "stir/IO/OutputFileFormat.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"

namespace stir {

class LineSearcher: public ParsingObject
{
public:
    LineSearcher();
    ////// Methods
    void set_defaults();
    void setup();
    float compute_line_search_value(const float alpha);
    void perform_line_search();

    typedef DiscretisedDensity<3,float> target_type;

    ////// Class variables
    int num_evaluations;
    float alpha_min;
    float alpha_max;
    bool use_log_alphas;




protected:
    shared_ptr<GeneralisedObjectiveFunction<target_type> >  objective_function_sptr;

private:
    std::string input_filename; // data
    std::string additive_sinogram_filename;
//    std::string multiplicative_factors_filename;
    std::string image_filename; //image

    void initialise_keymap();
    bool post_processing();
};

LineSearcher::LineSearcher()
{
  set_defaults();
}

void
LineSearcher::set_defaults()
{
  objective_function_sptr.reset(new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type>);
  output_file_format_sptr = OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
  num_evaluations = 10;
  alpha_min = 0.0;
  alpha_max = 1.0;
  use_log_alphas = false;
}

void
LineSearcher::initialise_keymap()
{
  parser.add_start_key("LineSearcher parameters");
  parser.add_key("image filename", &image_filename);
  parser.add_parsing_key("objective function type", &objective_function_sptr);
  parser.add_key("number of evaluations", &num_evaluations);
  parser.add_key("alpha min", &alpha_min);
  parser.add_key("alpha max", &alpha_max);
  parser.add_stop_key("End");
}

bool LineSearcher::
post_processing()
{
  if (is_null_ptr(this->objective_function_sptr))
  {
    error("objective_function_sptr is null");
    return true;
  }
  return false;
}

void LineSearcher::
setup()
{
  /////// load initial density from file
  shared_ptr<DiscretisedDensity<3,float> > image_sptr(read_from_file<DiscretisedDensity<3,float> >(image_filename));

  /////// Select

}

float LineSearcher::
compute_line_search_value(const float alpha)
{
  return 0;
}

void LineSearcher::perform_line_search() {

}

}// end of namespace stir

int main(int argc, char **argv)
{
  using namespace stir;

  if (argc!=2)
  {
    std::cerr << "Normal usage: " << argv[0] << " parameter-file\n";
    std::cerr << "I will now ask you the questions interactively\n";
  }
  LineSearcher my_stuff;
  my_stuff.set_defaults();
  if (argc!=2)
    my_stuff.ask_parameters();
  else
    my_stuff.parse(argv[1]);
//  my_stuff.run();
  return EXIT_SUCCESS;
}