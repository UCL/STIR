//
//
/*!
  \file
  \ingroup examples
  \brief A modification of demo2.cxx that parses all parameters from a parameter file.

  It illustrates
	- basic class derivation principles
	- how to use ParsingObject to have automatic capabilities of parsing
	  parameters files (and interactive questions to the user)
	- how most STIR programs parse the parameter files.

  Note that the same functionality could be provided without deriving
  a new class from ParsingObject. One could have a KeyParser object
  in main() and fill it in directly.

  See README.txt in the directory where this file is located.

  \author Kris Thielemans and Robert Twyman      
*/
/*
    Copyright (C) 2004- 2012, Hammersmith Imanet Ltd

    This software is distributed under the terms 
    of the GNU General  Public Licence (GPL)
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"

namespace stir {

class MyStuff: public ParsingObject
{
public:
  void set_defaults();
  void initialise_keymap();
  bool post_processing();
  void run();
  typedef DiscretisedDensity<3,float> target_type;

protected:
  shared_ptr<GeneralisedObjectiveFunction<target_type> >  objective_function_sptr;

private:
  std::string input_filename;
  std::string image_filename;
};

void
MyStuff::set_defaults()
{
  objective_function_sptr.reset(new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type>);
}

void 
MyStuff::initialise_keymap()
{
  parser.add_start_key("MyStuff parameters");
  parser.add_key("input file", &input_filename);
  parser.add_key("image filename", &image_filename);
  parser.add_parsing_key("objective function type", &objective_function_sptr);
  parser.add_stop_key("End");
}

bool MyStuff::
post_processing()
{
  if (is_null_ptr(this->objective_function_sptr))
  {
      error("objective_function_sptr is null");
      return true;
  }
  return false;
}

void
MyStuff::run()
{

  std::cout << "input_filename: " << input_filename << "\n";
  std::cout << "image_filename: " << image_filename << "\n";

  /////// load data from file
  shared_ptr<DiscretisedDensity<3,float> > 
    density_sptr(read_from_file<DiscretisedDensity<3,float> >(image_filename));

  // Checks density_sptr is loaded correctly.
  if (is_null_ptr(density_sptr))
  {
      error("density_sptr is null");
  }

  /////// setup objective function object
  objective_function_sptr->set_up(density_sptr);

  /////// Compute objective function value
  double my_val = objective_function_sptr->compute_objective_function(*density_sptr);

  /////////////// output
  std::cout << "The Objective Function Value of "<< image_filename << " = " << my_val << "\n";

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
  MyStuff my_stuff;
  my_stuff.set_defaults();
  if (argc!=2)
    my_stuff.ask_parameters();
  else
    my_stuff.parse(argv[1]);
  my_stuff.run();
  return EXIT_SUCCESS;
}
