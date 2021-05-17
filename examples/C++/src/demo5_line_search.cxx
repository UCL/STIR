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


std::vector<float>
compute_linear_alphas(const float alpha_min, const float alpha_max, const float num_evaluations)
{
  std::vector<float> alphas;
  float d_alpha = (alpha_max - alpha_min) / num_evaluations;

  std::cout << "\nComputing linear alphas:"
               "\n  alpha_min =   " << alpha_min <<
               "\n  alpha_max =   " << alpha_max <<
               "\n  delta_alpha = " << d_alpha << "\n";

  /// Explicitly add alpha = 0.0 and/or alpha_min
  alphas.push_back(0.0);
  if (alpha_min != 0.0)
    alphas.push_back(alpha_min);

  /// create a vector from (alpha_min + d_alpha) to alpha_max
  for (int i = 1; i <= num_evaluations; i++)
    alphas.push_back(i * d_alpha);

  return alphas;
}


using namespace stir;

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
    bool use_exponential_alphas;

    shared_ptr<DiscretisedDensity<3,float> > image_sptr;
    shared_ptr<DiscretisedDensity<3,float> > gradient_sptr;
    shared_ptr<DiscretisedDensity<3,float> > eval_image_sptr;

protected:
    shared_ptr<GeneralisedObjectiveFunction<target_type> >  objective_function_sptr;

private:
    std::string image_filename;
    bool is_setup = false;
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
  num_evaluations = 10;
  alpha_min = 0.0;
  alpha_max = 1.0;
  use_exponential_alphas = false;
  is_setup = false;
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
  parser.add_key("use exponential alphas", &use_exponential_alphas);
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
  this->is_setup = false;
  /////// load initial density from file
  if (image_filename == "")
    error("LineSearcher setup. No image filename has been given.");
  std::cout << "Loading image: \n    " << image_filename << "\n";
  this->image_sptr  = read_from_file<DiscretisedDensity<3,float> >(image_filename);

  //////// gradient it copied Density filled with 0's
  this->gradient_sptr.reset(this->image_sptr->get_empty_copy());
  this->eval_image_sptr.reset(this->image_sptr->get_empty_copy());

  /////// setup the objective function
  objective_function_sptr->set_num_subsets(1);
  objective_function_sptr->set_up(image_sptr);


  //////// compute the gradient
  objective_function_sptr->compute_sub_gradient(*gradient_sptr, *image_sptr, 0);

  this->is_setup = true;
}


void
LineSearcher::perform_line_search() {
  if (!is_setup)
    error("LineSearcher is not setup, please run setup()");

  std::vector<float> alphas;
  std::vector<float> Phi;

  std::cout << "Computing objective function values of alphas from "  << this->alpha_min << " to "
            << this->alpha_max << " in increments of " << this->num_evaluations << "\n";


  /// get alpha values as a vector
  {
    if ( this->use_exponential_alphas )
      error("exponential alphas not yet implemented.");
    else
      alphas = compute_linear_alphas(this->alpha_min, this->alpha_max, this->num_evaluations);
  }




  std::cout << this->image_sptr->find_max();

  for (auto a = alphas.begin(); a != alphas.end(); ++a)
  {
    phi = this->compute_line_search_value(*a);
    Phi.push_back(phi);
    std::cout << "alpha = " << *a << ". Phi = " << phi << "\n";
  }

  std::cout << "\n\n"
               "====================================\n"
               "Alphas and Phi values: \n";
  for (int i = 0 ; i < alphas.size() ; ++i){
    std::cout << "  alpha = " << alphas[i] << ". Phi = " << Phi[i] << "\n";
  }
}


float
LineSearcher::compute_line_search_value(const float alpha)
{
  eval_image_sptr->fill(0.0);
  *eval_image_sptr += *this->image_sptr + *this->gradient_sptr * alpha;
  std::cout << "\nimage_max  = " <<  image_sptr->find_max()
            << "\ngrad_max = " << gradient_sptr->find_max()
            << "\neval_max = " << eval_image_sptr->find_max() << "\n";
  return objective_function_sptr->compute_objective_function(*eval_image_sptr);
}

int main(int argc, char **argv)
{
  using namespace stir;

  if (argc!=2)
  {
    std::cerr << "Normal usage: " << argv[0] << " parameter-file\n";
    std::cerr << "I will now ask you the questions interactively\n";
  }
  LineSearcher my_stuff;
  if (argc!=2)
    my_stuff.ask_parameters();
  else
    my_stuff.parse(argv[1]);
  my_stuff.setup();
  my_stuff.perform_line_search();


  return EXIT_SUCCESS;
}