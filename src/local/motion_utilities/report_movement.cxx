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
  \brief Utility to report RMSE within a frame w.r.t. reference position

  \author Kris Thielemans
  $Date$
  $Revision$
  
  \par Usage
\verbatim
  report_movement \\
     [--frame_num_to_process number]\\
     [par_file]
\endverbatim
  See class documentation for stir::ReportMovement for more info, including the format
  of the par_file. 

  Command line switches override any values in the par_file.

*/

#include "local/stir/motion/TimeFrameMotion.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR

/*! \ingroup motion
  \brief A class for reporting the movement within the frame w.r.t. to the reference position

  \par Example par file
  \see TimeFrameMotion for other parameters
  \verbatim
  ReportMovement Parameters:=

  ; parameters from TimeFrameMotion

  END :=
\endverbatim
*/  
class ReportMovement : public TimeFrameMotion
{
private:
  typedef TimeFrameMotion base_type;
public:
  ReportMovement(const char * const par_filename);

  virtual Succeeded process_data();

protected:

  
  //! parsing functions
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

};

void 
ReportMovement::set_defaults()
{
  base_type::set_defaults();
}

void 
ReportMovement::initialise_keymap()
{
  parser.add_start_key("ReportMovement Parameters");

  base_type::initialise_keymap();

  parser.add_stop_key("END");
}

ReportMovement::
ReportMovement(const char * const par_filename)
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
ReportMovement::
post_processing()
{
  if (base_type::post_processing() == true)
    return true;

  return false;
}


Succeeded 
ReportMovement::
process_data()
{

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

      transform_3d_object(*out_density_sptr, *in_density_sptr,
			  this->get_current_rigid_object_transformation());


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
  ReportMovement application(argc==2 ? argv[1] : 0);
  if (set_move_to_reference)
    application.move_to_reference(move_to_reference);
  if (set_frame_num_to_process)
    application.set_frame_num_to_process(frame_num_to_process);
  Succeeded success =
    application.process_data();

  return success == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
