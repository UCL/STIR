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

#include "stir/DiscretisedDensity.h"
#include "stir/IO/OutputFileFormat.h"
#include "local/stir/motion/transform_3d_object.h"
#include "local/stir/motion/TimeFrameMotion.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR

/*! \ingroup motion
  \brief A class for moving an image according to average motion in the frame.

  \see transform_3d_object(DiscretisedDensity<3,float>& out_density, 
		    const DiscretisedDensity<3,float>& in_density, 
		    const RigidObject3DTransformation& rigid_object_transformation)

  \par Example par file
  \see TimeFrameMotion for other parameters
  \verbatim
  MoveImage Parameters:=
  input file:= input_filename
  ; output name
  ; filenames will be constructed by appending _f#g1d0b0 (and the extension)
  ; where # is the frame number
  output filename prefix:= output

  ; Change output file format, defaults to Interfile. See OutputFileFormat.
  ;Output file format := interfile

  ; parameters from TimeFrameMotion

  END :=
\endverbatim
*/  
class MoveImage : public TimeFrameMotion
{
private:
  typedef TimeFrameMotion base_type;
public:
  MoveImage(const char * const par_filename);


  virtual Succeeded process_data();

protected:

  
  //! parsing functions
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  //! parsing variables
  string input_filename;
  string output_filename_prefix;
private:
  shared_ptr<DiscretisedDensity<3,float> >  in_density_sptr; 
  shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > >
  output_file_format_sptr;  
};

void 
MoveImage::set_defaults()
{
  base_type::set_defaults();
  output_file_format_sptr =
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
}

void 
MoveImage::initialise_keymap()
{
  parser.add_start_key("MoveImage Parameters");

  parser.add_key("input file",&input_filename);
  parser.add_key("output filename prefix",&output_filename_prefix);
  parser.add_parsing_key("Output file format",&output_file_format_sptr);

  base_type::initialise_keymap();

  parser.add_stop_key("END");
}

MoveImage::
MoveImage(const char * const par_filename)
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
MoveImage::
post_processing()
{
  if (base_type::post_processing() == true)
    return true;

  if (output_filename_prefix.size()==0)
    {
      warning("You have to specify an output_filename_prefix");
      return true;
    }
  in_density_sptr = 
    DiscretisedDensity<3,float>::read_from_file(input_filename);

  return false;
}


Succeeded 
MoveImage::
process_data()
{
  shared_ptr< DiscretisedDensity<3,float> > out_density_sptr =
    in_density_sptr->get_empty_discretised_density();

  const unsigned int min_frame_num =
    this->get_frame_num_to_process()==-1
    ? 1 : this->get_frame_num_to_process();
  const unsigned int max_frame_num =
    this->get_frame_num_to_process()==-1 
    ? this->get_time_frame_defs().get_num_frames() 
    : this->get_frame_num_to_process();

  for (unsigned int current_frame_num = min_frame_num;
       current_frame_num<=max_frame_num;
       ++current_frame_num)
    {
      set_frame_num_to_process(current_frame_num);

      out_density_sptr->fill(0);

      transform_3d_object(*out_density_sptr, *in_density_sptr,
			  this->get_current_rigid_object_transformation());


      //*********** open output file

      {
	char rest[50];
	sprintf(rest, "_f%dg1d0b0", current_frame_num);
	const string output_filename = output_filename_prefix + rest;
	if (output_file_format_sptr->write_to_file(output_filename, *out_density_sptr)
	    == Succeeded::no)
	  {
	    warning("Error writing file %s. Exiting\n",
		  output_filename.c_str());
	    return Succeeded::no;
	  }
	}
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
  MoveImage application(argc==2 ? argv[1] : 0);
  if (set_move_to_reference)
    application.move_to_reference(move_to_reference);
  if (set_frame_num_to_process)
    application.set_frame_num_to_process(frame_num_to_process);
  Succeeded success =
    application.process_data();

  return success == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
