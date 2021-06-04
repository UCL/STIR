/*!
  \file
  \ingroup examples
  \brief A class to compute a line search evaluation in the direction of
  the gradient of a given image and objective function.
  All parameters can be parsed from a parameter file. See `demo5_line_search.par`.

  Give an image and objective function configuration, this script will perform a line search
```suggestion
  from a minimum to a maximum step size (alpha).
  Options are included to perform this line search linearly or using exponential step
  sizes.
  Additionally, a lower positivity bound is applied to all computed images.

  The results are saved to files: `alphas.dat` contains the step size values investigated,
  and `Phis.dat` contains the objective function evaluations.
  Furthermore, the image corresponding to the maximum objective function and the gradient
  used in the line search are saved to file.

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


std::vector<double>
compute_linear_alphas(const float alpha_min, const float alpha_max, const float num_evaluations)
{
  /// This function computes a vector (of length num_evaluations) of linear values from alpha_min to alpha_max
  std::vector<double> alphas;
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
    alphas.push_back(i * d_alpha + alpha_min);

  return alphas;
}


std::vector<double>
compute_exponential_alphas(const float alpha_min, const float alpha_max, const float num_evaluations)
{
  /// This function computes a vector (of length num_evaluations) of exponential values from 10^alpha_min to 10^alpha_max
  std::vector<double> alphas;
  float d_alpha = (alpha_max - alpha_min) / num_evaluations;

  std::cout << "\nComputing exponential alphas:"
               "\n  exponential min =    " << alpha_min <<
               "\n  exponential max =    " << alpha_max <<
               "\n  exponential delta  = " << d_alpha << "\n";

  /// Explicitly add alpha = 0.0 and/or alpha_min
  alphas.push_back(0.0);

  /// create a vector from (alpha_min + d_alpha) to alpha_max
  for (int i = 1; i <= num_evaluations; i++)
    alphas.push_back(pow(10, i * d_alpha + alpha_min));

  return alphas;
}


void
save_doubles_vector_to_file(std::string filename, std::vector<double> vector)
{
  /// This function is used to save the line search results (alpha and Phi values) to separate files.
  std::ofstream myfile (filename);
  int precision = 40;
  if (myfile.is_open()){
    for (double v : vector){
      myfile << std::fixed << std::setprecision(precision) << v << std::endl;
    }
    myfile.close();
  }
}


using namespace stir;

class LineSearcher: public ParsingObject
{
public:
    LineSearcher();
    /// Methods
    void set_defaults();
    void setup();
    double compute_line_search_value(const double alpha);
    void apply_update_step(const double alpha);
    void perform_line_search();
    void save_data();
    void save_max_line_search_image();

    typedef DiscretisedDensity<3,float> target_type;

    /// Class variables
    int num_evaluations;
    float alpha_min;
    float alpha_max;
    bool use_exponential_alphas;
    double image_lower_bound;

    /// Measurements
    std::vector<double> alphas;
    std::vector<double> Phis;

    /// Image volumes
    shared_ptr<DiscretisedDensity<3,float> > image_sptr;
    shared_ptr<DiscretisedDensity<3,float> > gradient_sptr;
    shared_ptr<DiscretisedDensity<3,float> > eval_image_sptr;

protected:
    shared_ptr<GeneralisedObjectiveFunction<target_type> >  objective_function_sptr;

private:
    void initialise_keymap();
    bool post_processing();

    std::string image_filename;
    bool is_setup;
    shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format_sptr;
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
  image_lower_bound = 0.0;
  output_file_format_sptr = OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
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
  parser.add_key("line search image lower bound", &image_lower_bound);
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
  /// Setup LineSearcher
  this->is_setup = false;

  /// Load initial density from file
  if (image_filename == "")
    error("LineSearcher setup. No image filename has been given.");

  std::cout << "Loading image: \n    " << image_filename << "\n";
  this->image_sptr  = read_from_file<DiscretisedDensity<3,float> >(image_filename);

  /// Gradient it copied density filled with 0's
  this->gradient_sptr.reset(this->image_sptr->get_empty_copy());
  this->eval_image_sptr.reset(this->image_sptr->get_empty_copy());

  /// Setup the objective function
  objective_function_sptr->set_num_subsets(1);
  objective_function_sptr->set_up(image_sptr);

  /// Compute the gradient
  objective_function_sptr->compute_sub_gradient(*gradient_sptr, *image_sptr, 0);

  this->is_setup = true;
}


void
LineSearcher::perform_line_search() {
  /// Performs the line search
  /// Gets the step sizes
  /// Computes the objective function value of the image at every step size
  /// Outputs values to console

  if (!is_setup)
    error("LineSearcher is not setup, please run setup()");

  double Phi; // Used to store objective function values of each iterate
  std::cout << "Computing objective function values of alphas from "  << this->alpha_min << " to "
            << this->alpha_max << " in increments of " << this->num_evaluations << "\n";

  /// get alpha values as a vector
  {
    if ( this->use_exponential_alphas )
      alphas = compute_exponential_alphas(this->alpha_min, this->alpha_max, this->num_evaluations);
    else
      alphas = compute_linear_alphas(this->alpha_min, this->alpha_max, this->num_evaluations);
  }

  /// Iterate over each of the alphas and compute Phi
  for (auto a = alphas.begin(); a != alphas.end(); ++a)
  {
    Phi = this->compute_line_search_value(*a);
    Phis.push_back(Phi);
    std::cout << "alpha = " << *a << ". Phi = " << Phi << "\n";
  }

  /// Output alpha and Phi values to console
  std::cout << "\n\n====================================\n"
               "Alpha and Phi values: \n";
  for (int i = 0 ; i < alphas.size() ; ++i)
    std::cout << std::setprecision(20) << "  alpha = " << alphas[i] << ". Phis = " << Phis[i] << "\n";

}


double
LineSearcher::compute_line_search_value(const double alpha)
{
  /// For a given alpha, computes the objective function value at the update step
  apply_update_step(alpha);

  std::cout << "\nimage_min  = " <<  image_sptr->find_min()
            << "\ngrad_min = " << gradient_sptr->find_min()
            << "\neval_min = " << eval_image_sptr->find_min() << "\n";
  return objective_function_sptr->compute_objective_function(*eval_image_sptr);
}

void
LineSearcher::apply_update_step(const double alpha)
{
  /// Computes the update step, applies a lower threshold to the evaluation image
  this->eval_image_sptr->fill(0.0);
  *this->eval_image_sptr += *this->image_sptr + *this->gradient_sptr * alpha;
  this->eval_image_sptr->apply_lower_threshold(this->image_lower_bound);
}


void
LineSearcher::save_data()
{
  /// Saves the alpha and Phi data to separate file
  /// Saves the line search gradient to file
  if (alphas.size() != Phis.size())
    error("Length of alpha and Phi vectors is not equal.");
  save_doubles_vector_to_file("alphas.dat", this->alphas);
  save_doubles_vector_to_file("Phis.dat", this->Phis);

  /// Save gradient to file
  std::cout << "Saving LineSearchGradient.hv\n";
  output_file_format_sptr->write_to_file("LineSearchGradient.hv", *this->gradient_sptr);
}


void
LineSearcher::save_max_line_search_image()
{
  /// Finds the image with the maximum objective function in the line search and saves it to file
  double max_Phi = Phis[0];
  double max_alpha = alphas[0];

  /// Find the alpha Phi combination that is max Phi and save that image.
  for (int i = 0 ; i < alphas.size() ; ++i){
    if (Phis[i] > max_Phi){
      max_Phi = Phis[i];
      max_alpha = alphas[i];
    }
  }

  /// Check if alpha == 0 is optimal, otherwise save the image at the maximum evaluation.
  if (max_alpha != alphas[0])
  {
    apply_update_step(max_alpha);
    std::cout << "Saving max line search value image, computed at alpha = " << max_alpha << "\n"
              << "and Phi = " << max_Phi << '\n';
    output_file_format_sptr->write_to_file("MaxObjectiveFunctionImage.hv", *this->eval_image_sptr);
  } else {
    std::cout << "Max line search value image is at alpha = 0.0\n";
  }
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
  my_stuff.save_data();
  my_stuff.save_max_line_search_image();
  return EXIT_SUCCESS;
}
