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
  \brief Utility to move projection data according to average motion in the frame.

  \author Kris Thielemans
  $Date$
  $Revision$
  
  \par Usage
\verbatim
  move_projdata \\
     [--move-to-reference 0|1] \\
     [--frame_num_to_process number]\\
     [par_file]
\endverbatim
  See class documentation for stir::MoveProjData for more info, including the format
  of the par_file. 

  Command line switches override any values in the par_file.

*/
#include "stir/ProjDataInterfile.h"
//#include "stir/IO/DefaultOutputFileFormat.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/motion/RigidObject3DMotion.h"
#include "local/stir/motion/transform_3d_object.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include <iostream>

START_NAMESPACE_STIR

/*! \ingroup motion
  \brief A class for moving projection data according to average motion in the frame.

  Output is currently always in interfile format

  \see transform_3d_object(ProjData& out_proj_data,
		    const ProjData& in_proj_data,
		    const RigidObject3DTransformation& rigid_object_transformation)
  
  \par Example par file
  \verbatim
  MoveProjData Parameters:=
  input file:= input_filename
  ; output name
  ; filenames will be constructed by appending _f#g1d0b0 (and the extension)
  ; where # is the frame number
  output filename prefix:= output
  ;; next defaults to input file
  ;output template filename:=
  ; alternative way to reduce number of segments (defaults to: use all)
  ;max_out_segment_num_to_process:=-1

  ; see TimeFrameDefinitions
  time frame_definition filename := frame_definition_filename

  ; next defines
  ;move_to_reference := 1
  ; next can be set to do only 1 frame, defaults means all frames
  ;frame_num_to_process := -1
  Rigid Object 3D Motion Type := type

  END :=
\endverbatim
*/  
class MoveProjData : public ParsingObject
{
public:
  MoveProjData(const char * const par_filename);

  TimeFrameDefinitions frame_defs;

  virtual Succeeded process_data();

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
  string output_template_filename;
  string frame_definition_filename;
 
  bool do_move_to_reference;

  int frame_num_to_process;     

  int max_in_segment_num_to_process;
  int max_out_segment_num_to_process;

private:
  shared_ptr<ProjData >  in_proj_data_sptr; 
  shared_ptr<RigidObject3DMotion> ro3d_ptr;
  shared_ptr<ProjDataInfo> proj_data_info_ptr; // template for output
  // shared_ptr<OutputFileFormat> output_file_format_sptr;  
};

void 
MoveProjData::set_defaults()
{
  ro3d_ptr = 0;
  frame_num_to_process = -1;
  //output_file_format_sptr = new DefaultOutputFileFormat;
  do_move_to_reference = true;

  max_in_segment_num_to_process=-1;
  max_out_segment_num_to_process=-1;
}

void 
MoveProjData::initialise_keymap()
{

  parser.add_start_key("MoveProjData Parameters");

  parser.add_key("input file",&input_filename);
  parser.add_key("time frame definition filename",&frame_definition_filename);
  parser.add_key("output template filename",&output_template_filename);
  parser.add_key("output filename prefix",&output_filename_prefix);
  parser.add_key("max_out_segment_num_to_process", &max_out_segment_num_to_process);
  parser.add_key("max_in_segment_num_to_process", &max_in_segment_num_to_process);

  parser.add_key("move_to_reference", &do_move_to_reference);
  parser.add_key("frame_num_to_process", &frame_num_to_process);
  parser.add_parsing_key("Rigid Object 3D Motion Type", &ro3d_ptr); 
  //parser.add_parsing_key("Output file format",&output_file_format_sptr);
  parser.add_stop_key("END");
}

MoveProjData::
MoveProjData(const char * const par_filename)
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
MoveProjData::
post_processing()
{
   

  if (output_filename_prefix.size()==0)
    {
      warning("You have to specify an output_filename_prefix\n");
      return true;
    }
  in_proj_data_sptr = 
    ProjData::read_from_file(input_filename);
  if (max_in_segment_num_to_process<0)
    max_in_segment_num_to_process = in_proj_data_sptr->get_max_segment_num();

  if (output_template_filename.size() != 0)
    {
      shared_ptr<ProjData> template_proj_data_sptr = 
	ProjData::read_from_file(output_template_filename);
      proj_data_info_ptr =
	template_proj_data_sptr->get_proj_data_info_ptr()->clone();
    }
  else
    {
      proj_data_info_ptr =
	in_proj_data_sptr->get_proj_data_info_ptr()->clone();
    }
  if (max_out_segment_num_to_process<0)
    max_out_segment_num_to_process = 
      proj_data_info_ptr->get_max_segment_num();
  else
    proj_data_info_ptr->reduce_segment_range(-max_out_segment_num_to_process,max_out_segment_num_to_process);

  // handle time frame definitions etc

  if (frame_definition_filename.size()==0)
    {
      warning("Have to specify either 'time frame_definition_filename'\n");
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
  if (!ro3d_ptr->is_synchronised())
    {
      warning("You have to specify a time_offset (or some other way to synchronise the time\n");
      return true;
    }
#endif

  return false;
}

 

void 
MoveProjData::
move_to_reference(const bool value)
{
  do_move_to_reference=value;
}

void
MoveProjData::
set_frame_num_to_process(const int value)
{
  frame_num_to_process=value;
}

Succeeded 
MoveProjData::
process_data()
{
  shared_ptr<ProjData> out_proj_data_sptr;

  const unsigned int min_frame_num =
    frame_num_to_process==-1 ? 1 : frame_num_to_process;
  const unsigned int max_frame_num =
    frame_num_to_process==-1 ? frame_defs.get_num_frames() : frame_num_to_process;

  for (unsigned int current_frame_num = min_frame_num;
       current_frame_num<=max_frame_num;
       ++current_frame_num)
    {
      const double start_time = frame_defs.get_start_time(current_frame_num);
      const double end_time = frame_defs.get_end_time(current_frame_num);
      cerr << "\nDoing frame " << current_frame_num
	   << ": from " << start_time << " to " << end_time << endl;

      {
	char rest[50];
	sprintf(rest, "_f%dg1d0b0", current_frame_num);
	const string output_filename = output_filename_prefix + rest;
	out_proj_data_sptr = 
	  new ProjDataInterfile (proj_data_info_ptr, output_filename, ios::out); 
      }

      RigidObject3DTransformation rigid_object_transformation =
	ro3d_ptr->compute_average_motion_rel_time(start_time, end_time);
      
      
      rigid_object_transformation = 
	compose(ro3d_ptr->get_transformation_to_scanner_coords(),
		compose(ro3d_ptr->get_transformation_to_reference_position(),
			compose(rigid_object_transformation,
				ro3d_ptr->get_transformation_from_scanner_coords())));
      if (!do_move_to_reference)
	rigid_object_transformation = 
	  rigid_object_transformation.inverse();

      std::cout << "Applying transformation " 
		<< rigid_object_transformation
		<< '\n';

      if (transform_3d_object(*out_proj_data_sptr, *in_proj_data_sptr,
			      rigid_object_transformation)
	  == Succeeded::no)
	return Succeeded::no;
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
  MoveProjData application(argc==2 ? argv[1] : 0);
  if (set_move_to_reference)
    application.move_to_reference(move_to_reference);
  if (set_frame_num_to_process)
    application.set_frame_num_to_process(frame_num_to_process);
  Succeeded success =
    application.process_data();

  return success == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
