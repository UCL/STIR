/*!
  \file
  \ingroup listmode
  \brief Utility to move an image in several steps according to average motion in the frame.

  \author Kris Thielemans
  $Date$
  $Revision$
  
  \par Example par file
  \verbatim
  MoveImage Parameters:=
  input file:= input_filename
  time frame_definition filename := frame_definition_filename
  output filename prefix := output_filename_prefix
  ;move_to_reference := 1
  ; next can be set to do only 1 frame, defaults means all frames
  ;frame_num_to_process := -1
  Rigid Object 3D Motion Type := type
  ;Output file format := interfile
  END :=
\endverbatim
}
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/DiscretisedDensity.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/motion/RigidObject3DMotion.h"
#include "local/stir/motion/transform_3d_object.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include <time.h> // for localtime

START_NAMESPACE_STIR


class MoveImage : public ParsingObject
{
public:
  MoveImage(const char * const par_filename);

  TimeFrameDefinitions frame_defs;

  virtual void process_data();

  void move_to_reference(const bool);
  void set_frame_num_to_process(const int);
protected:

  
  //! parsing functions
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  //! parsing variables
  string input_filename;
  string output_filename_prefix;
  string frame_definition_filename;

  double scan_start_time;

  bool do_move_to_reference;

  int frame_num_to_process;     
private:
  shared_ptr<DiscretisedDensity<3,float> >  in_density_sptr; 
  shared_ptr<RigidObject3DMotion> ro3d_ptr;

  shared_ptr<OutputFileFormat> output_file_format_sptr;  

  int scan_start_time_secs_since_1970_UTC;
};

void 
MoveImage::set_defaults()
{
  ro3d_ptr = 0;
  frame_num_to_process = -1;
  output_file_format_sptr = new DefaultOutputFileFormat;
  do_move_to_reference = true;
  scan_start_time_secs_since_1970_UTC=-1;
}

void 
MoveImage::initialise_keymap()
{

  parser.add_start_key("MoveImage Parameters");

  parser.add_key("input file",&input_filename);
  parser.add_key("scan_start_time_secs_since_1970_UTC", 
		 &scan_start_time_secs_since_1970_UTC);
  parser.add_key("time frame definition filename",&frame_definition_filename);
  parser.add_key("output filename prefix",&output_filename_prefix);
  parser.add_key("move_to_reference", &do_move_to_reference);
  parser.add_key("frame_num_to_process", &frame_num_to_process);
  parser.add_parsing_key("Rigid Object 3D Motion Type", &ro3d_ptr); 
  parser.add_parsing_key("Output file format",&output_file_format_sptr);
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

  if (scan_start_time_secs_since_1970_UTC==-1)
    {
      warning("scan_start_time_secs_since_1970_UTC not set. Will use relative time");
      scan_start_time = 0;
    }
  else 
    {
      if (scan_start_time_secs_since_1970_UTC<1000)
	{
	  warning("scan_start_time_secs_since_1970_UTC too small");
	  return true;
	}
      {
	// convert to time_in_secs since midnight
	time_t sec_time = scan_start_time_secs_since_1970_UTC;
	
	struct tm* ScanTime = localtime( &sec_time  ) ;
	scan_start_time = 
	  ( ScanTime->tm_hour * 3600.0 ) + 
	  ( ScanTime->tm_min * 60.0 ) +
	  ScanTime->tm_sec ;
      }
    }

  
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

  if (frame_num_to_process!=-1 &&
      (frame_num_to_process<1 || 
       static_cast<unsigned>(frame_num_to_process)>frame_defs.get_num_frames()))
    {
      warning("Frame number should be between 1 and %d\n",
	      frame_defs.get_num_frames());
      return true;
    }

#if 0
  // TODO move to RigidObject3DMotion
  if (!ro3d_ptr->is_time_offset_set())
    {
      warning("You have to specify a time_offset (or some other way to synchronise the time\n");
      return true;
    }
#endif

  return false;
}

 

void 
MoveImage::
move_to_reference(const bool value)
{
  do_move_to_reference=value;
}

void
MoveImage::
set_frame_num_to_process(const int value)
{
  frame_num_to_process=value;
}

void 
MoveImage::
process_data()
{
  shared_ptr< DiscretisedDensity<3,float> > out_density_sptr =
    in_density_sptr->get_empty_discretised_density();

  const unsigned int min_frame_num =
    frame_num_to_process==-1 ? 1 : frame_num_to_process;
  const unsigned int max_frame_num =
    frame_num_to_process==-1 ? frame_defs.get_num_frames() : frame_num_to_process;

  for (unsigned int current_frame_num = min_frame_num;
       current_frame_num<=max_frame_num;
       ++current_frame_num)
    {
      const double start_time = 
	frame_defs.get_start_time(current_frame_num) + scan_start_time;
      const double end_time = 
	frame_defs.get_end_time(current_frame_num) +  scan_start_time;
      cerr << "\nDoing frame " << current_frame_num
	   << ": from " << start_time << " to " << end_time << endl;


      out_density_sptr->fill(0);

      RigidObject3DTransformation rigid_object_transformation =
	scan_start_time_secs_since_1970_UTC==-1 
	? ro3d_ptr->compute_average_motion_rel_time(start_time, end_time)
	: ro3d_ptr->compute_average_motion(start_time, end_time);
      
      rigid_object_transformation = 
	compose(ro3d_ptr->get_transformation_to_scanner_coords(),
		compose(ro3d_ptr->get_transformation_to_reference_position(),
			compose(rigid_object_transformation,
				ro3d_ptr->get_transformation_from_scanner_coords())));
      if (!do_move_to_reference)
	rigid_object_transformation = 
	  rigid_object_transformation.inverse();

      transform_3d_object(*out_density_sptr, *in_density_sptr,
				rigid_object_transformation);


      //*********** open output file

      {
	char rest[50];
	sprintf(rest, "_f%dg1d0b0", current_frame_num);
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
  application.process_data();

  return EXIT_SUCCESS;
}
