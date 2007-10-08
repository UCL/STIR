//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    Internal GE use only
*/
/*!
  \file
  \ingroup motion_utilities
  \brief Utility to move an image according to average motion in the frame.

  \author Kris Thielemans
  $Date$
  $Revision$
  
  \par Usage
\verbatim
  move_image \\
     [--move-to-reference 0|1] \\
     [--frame_num_to_process number]\\
     [par_file]
\endverbatim
  See class documentation for stir::MoveImage for more info, including the format
  of the par_file. 

  Command line switches override any values in the par_file.

*/

#include "stir/stream.h"
#include "stir/ParsingObject.h"
#include "stir/shared_ptr.h"
#include "local/stir/motion/ObjectTransformation.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include <iostream>

START_NAMESPACE_STIR

/*! \ingroup motion
  \brief A class for moving an image according to average motion in the frame.

  \see transform_3d_object(DiscretisedDensity<3,float>& out_density, 
		    const DiscretisedDensity<3,float>& in_density, 
		    const RigidObject3DTransformation& rigid_object_transformation)

  \par Example par file
  \verbatim
  MoveImage Parameters:=

  END :=
\endverbatim
*/  
class MyApp : public ParsingObject
{
  typedef ParsingObject base_type;
public:
  MyApp(const char * const par_filename);
  virtual Succeeded process_data();

protected:

  
  //! parsing functions
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  //! parsing variables
  // string input_filename;
  //string output_filename_prefix;
private:
  shared_ptr<ObjectTransformation<3,float> > transformation_sptr;
};

void 
MyApp::set_defaults()
{
}


void 
MyApp::initialise_keymap()
{
  this->parser.add_start_key("Object Transformation Parameters");
  this->parser.add_parsing_key("transformation type",&this->transformation_sptr);
  this->parser.add_stop_key("END");
}

MyApp::
MyApp(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    {
      if (parse(par_filename)==false)
	exit(EXIT_FAILURE);
    }
  else
    ask_parameters();

}

bool
MyApp::
post_processing()
{
  if (base_type::post_processing() == true)
    return true;

  if (is_null_ptr(this->transformation_sptr))
    {
      warning("You have to specify a transformation");
      return true;
    }

  return false;
}


Succeeded 
MyApp::
process_data()
{
  BasicCoordinate<3,float> in_coord;
  while(true)
    {
      std::cout << "\nNext:\n";
      std::cin >> in_coord;
      std::cout << this->transformation_sptr->transform_point(in_coord)
		<< this->transformation_sptr->transform_point(in_coord) - in_coord
		<< " Jacobian " 
		<< this->transformation_sptr->jacobian(in_coord)
		<< std::endl;
    }
  return Succeeded::yes;
}


END_NAMESPACE_STIR



USING_NAMESPACE_STIR

int main(int argc, char * argv[])
{
  bool move_to_reference=true;
  bool set_move_to_reference=false;
  bool set_frame_num_to_process=false;
  int frame_num_to_process=-1;
  while (argc>=2 && argv[1][1]=='-')
    {
      if (strcmp(argv[1],"--move-to-reference")==0)
	{
	  set_move_to_reference=true;
	  move_to_reference=atoi(argv[2]);
	  argc-=2; argv+=2;
	}
      else if (strcmp(argv[1], "--frame_num_to_process")==0)
	{
	  set_frame_num_to_process=true;
	  frame_num_to_process=atoi(argv[2]);
	  argc-=2; argv+=2;
	}
      else
	{
	  warning("Wrong option\n");
	  exit(EXIT_FAILURE);
	}
    }

  if (argc!=1 && argc!=2) {
    cerr << "Usage: " << argv[0] << " \\\n"
	 << "\t[--move-to-reference 0|1] \\\n"
	 << "\t[--frame_num_to_process number]\\\n"
	 << "\t[par_file]\n";
    exit(EXIT_FAILURE);
  }
  MyApp application(argc==2 ? argv[1] : 0);
  Succeeded success =
    application.process_data();

  return success == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
