/*!
  \file
  \ingroup listmode
  \brief Utility to move an image in several steps according to average motion in the frame.

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/DiscretisedDensity.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/motion/RigidObject3DMotion.h"
#include "local/stir/motion/object_3d_transform_image.h"
#include "local/stir/listmode/TimeFrameDefinitions.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR


class MoveImage : public ParsingObject
{
public:
  MoveImage(const char * const par_filename);

  TimeFrameDefinitions frame_defs;

  virtual void process_data();
  
protected:

  
  //! parsing functions
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  //! parsing variables
  string input_filename;
  string output_filename_prefix;
  string frame_definition_filename;
 
     
private:
  shared_ptr<DiscretisedDensity<3,float> >  in_density_sptr; 
  shared_ptr<RigidObject3DMotion> ro3d_ptr;

  RigidObject3DTransformation move_to_scanner;
  RigidObject3DTransformation move_from_scanner;
 
  shared_ptr<OutputFileFormat> output_file_format_sptr;  
};

void 
MoveImage::set_defaults()
{
  ro3d_ptr = 0;
  output_file_format_sptr = new DefaultOutputFileFormat;

}

void 
MoveImage::initialise_keymap()
{

  parser.add_start_key("MoveImage Parameters");

  parser.add_key("input file",&input_filename);
  parser.add_key("frame_definition file",&frame_definition_filename);
  parser.add_key("output filename prefix",&output_filename_prefix);
  parser.add_parsing_key("Rigid Object 3D Motion Type", &ro3d_ptr); 
  parser.add_parsing_key("Output file format",&output_file_format_sptr);
  parser.add_stop_key("END");
}

MoveImage::
MoveImage(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    parse(par_filename) ;
  else
    ask_parameters();

}

bool
MoveImage::
post_processing()
{
   

  if (output_filename_prefix.size()==0)
    {
      warning("You have to specify an output_filename_prefix\n");
      return true;
    }
  in_density_sptr = 
    DiscretisedDensity<3,float>::read_from_file(input_filename);


  // handle time frame definitions etc

  if (frame_definition_filename.size()==0)
    {
      warning("Have to specify either 'frame_definition_filename'\n");
      return true;
    }

  frame_defs = TimeFrameDefinitions(frame_definition_filename);

  if (is_null_ptr(ro3d_ptr))
  {
    warning("Invalid Rigid Object 3D Motion object\n");
    return true;
  }


  // TODO move to RigidObject3DMotion
  if (!ro3d_ptr->is_time_offset_set())
    {
      warning("You have to specify a time_offset (or some other way to synchronise the time\n");
      return true;
    }

  move_from_scanner =
    RigidObject3DTransformation(Quaternion<float>(0.00525584F, -0.999977F, -0.00166456F, 0.0039961F),
                               CartesianCoordinate3D<float>( -1981.93F, 3.96638F, 20.1226F));
  move_to_scanner = move_from_scanner.inverse();

  return false;
}

 

void 
MoveImage::process_data()
{
  shared_ptr< DiscretisedDensity<3,float> > out_density_sptr =
    in_density_sptr->get_empty_discretised_density();

  for (unsigned int current_frame_num = 1;
       current_frame_num<=frame_defs.get_num_frames();
       ++current_frame_num)
    {
      const double start_time = frame_defs.get_start_time(current_frame_num);
      const double end_time = frame_defs.get_end_time(current_frame_num);
      cerr << "\nDoing frame " << current_frame_num
	   << ": from " << start_time << " to " << end_time << endl;


      out_density_sptr->fill(0);

      RigidObject3DTransformation rigid_object_transformation =
	ro3d_ptr->compute_average_motion_rel_time(start_time, end_time);
      
      rigid_object_transformation = 
	compose(move_to_scanner,
		compose(ro3d_ptr->get_transformation_to_reference_position(),
			compose(rigid_object_transformation,
				move_from_scanner)));
      rigid_object_transformation = 
	rigid_object_transformation.inverse();

      object_3d_transform_image(*out_density_sptr, *in_density_sptr,
				rigid_object_transformation);


      //*********** open output file

      {
	char rest[50];
	sprintf(rest, "_f%dg1b0d0", current_frame_num);
	const string output_filename = output_filename_prefix + rest;
	if (output_file_format_sptr->write_to_file(output_filename, *out_density_sptr)
	    == Succeeded::no)
	  {
	    error("Error writing file %s. Exiting\n",
		  output_filename.c_str());
	  }
	}
    }
  
}


END_NAMESPACE_STIR



USING_NAMESPACE_STIR

int main(int argc, char * argv[])
{
  
  if (argc!=1 && argc!=2) {
    cerr << "Usage: " << argv[0] << " [par_file]\n";
    exit(EXIT_FAILURE);
  }
  MoveImage application(argc==2 ? argv[1] : 0);
  application.process_data();

  return EXIT_SUCCESS;
}
