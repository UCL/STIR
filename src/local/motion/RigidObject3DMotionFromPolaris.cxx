//
// $Id$
//
/*!
  \file 
  \ingroup motion

  \brief Implementation of class RigidObject3DMotionFromPolaris
 
  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#include "local/stir/motion/RigidObject3DMotionFromPolaris.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/Quaternion.h"
#include "stir/listmode/CListRecord.h"
#include "local/stir/motion/Polaris_MT_File.h"
#include "stir/VectorWithOffset.h"
#include "stir/utilities.h"
#include "stir/stream.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/linear_regression.h"
#include <fstream>
#include <ctime>

# ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::time_t; using ::tm; using ::localtime; }
#endif

START_NAMESPACE_STIR

// Find and store gating values in a vector from lm_file  
static  void 
find_and_store_gate_tag_values_from_lm(VectorWithOffset<unsigned long>& lm_times_in_millisecs, 
				       VectorWithOffset<unsigned>& lm_random_number,
				       CListModeData& listmode_data);

const char * const 
RigidObject3DMotionFromPolaris::registered_name = "Motion From Polaris"; 

RigidObject3DMotionFromPolaris::RigidObject3DMotionFromPolaris()
{
  set_defaults();
}

const RigidObject3DTransformation& 
RigidObject3DMotionFromPolaris::
get_transformation_to_scanner_coords() const
{ return move_to_scanner_coords; }


const RigidObject3DTransformation& 
RigidObject3DMotionFromPolaris::
get_transformation_from_scanner_coords() const
{ return move_from_scanner_coords; }


RigidObject3DTransformation
RigidObject3DMotionFromPolaris::
compute_average_motion(const double start_time, const double end_time) const
{
  // CartesianCoordinate3D<float> euler_angles;
  int samples = 0 ;
  float total_tx=0.F;
  float total_ty=0.F;
  float total_tz=0.F; 
  Quaternion<float> total_q(0,0,0,0);
 
  Polaris_MT_File::const_iterator iter=mt_file_ptr->begin();

  while (iter!= mt_file_ptr->end())
  {
    /* Accept motions recorded during trasnmission acquisition */
    if ((iter->sample_time >= start_time ) && ( iter->sample_time<= end_time))
    {
      const Quaternion<float> quater(iter->quat[1], iter->quat[2], iter->quat[3], iter->quat[4]);	/* Sets the quaternion matrix */
      /* Maintain running total euler angles and translations */
      total_tx += iter->trans.x();
      total_ty += iter->trans.y() ;
      total_tz += iter->trans.z() ;
      total_q +=quater;
      samples += 1;
    }
    ++iter;
  }
  /* Average quat and translation */
 
  if (samples==0)
    {
      error("RigidObject3DMotionFromPolaris::compute_average_motion:\n"
	    "\t Start-end range (%g-%g) does not seem to overlap with MT info.",
	    start_time, end_time);	     
    }
  
  total_q /=static_cast<float>(samples);
  total_q.normalise();
  total_tx /= samples; 
  total_ty /= samples;
  total_tz /= samples;
  
  return RigidObject3DTransformation(total_q,
				     CartesianCoordinate3D<float>(total_tz,total_ty,total_tx));  
}

void 
RigidObject3DMotionFromPolaris::
get_motion_rel_time(RigidObject3DTransformation& ro3dtrans, const double time) const
{
  Polaris_MT_File::const_iterator iterator_for_record_just_after_this_time =
    mt_file_ptr->begin();

  while (iterator_for_record_just_after_this_time!= mt_file_ptr->end() &&
         iterator_for_record_just_after_this_time->sample_time < time + time_offset)
    ++iterator_for_record_just_after_this_time;

  if (iterator_for_record_just_after_this_time == mt_file_ptr->end())
  {
    error("RigidObject3DMotionFromPolaris: motion asked for time %g which is "
	  "beyond the range of data (time in Polaris units: %g)\n",
	  time, time + time_offset);
  }
  else
  {
  const RigidObject3DTransformation ro3dtrans_tmp (iterator_for_record_just_after_this_time->quat,
              iterator_for_record_just_after_this_time->trans);
  ro3dtrans=ro3dtrans_tmp;
  }

}


void 
RigidObject3DMotionFromPolaris::
do_synchronisation(CListModeData& listmode_data)
{
  std::time_t sec_time = 
    listmode_data.get_scan_start_time_in_secs_since_1970();

  if (sec_time==std::time_t(-1))
    {
      warning("Scan start time could not be found from list mode data");
    }
  else
    {
      // Polaris times are currently in localtime since midnight
      // This relies on TZ though: bad! (TODO)
      struct std::tm* lm_start_time_tm = std::localtime( &sec_time  ) ;
      const unsigned long lm_start_time = 
	( lm_start_time_tm->tm_hour * 3600L ) + 
	( lm_start_time_tm->tm_min * 60 ) + 
	lm_start_time_tm->tm_sec ;

      cerr << "\nListmode file says that listmode start time is " 
	   << lm_start_time 
	   << " in secs after midnight local time"<< endl;
    }

  VectorWithOffset<unsigned long> lm_times_in_millisecs;
  VectorWithOffset<unsigned> lm_random_numbers;
  find_and_store_gate_tag_values_from_lm(lm_times_in_millisecs,lm_random_numbers,listmode_data); 
  cerr << "done find and store gate tag values" << endl;
  const VectorWithOffset<unsigned>::size_type num_lm_tags = lm_random_numbers.size() ;
  if (num_lm_tags==0)
    error("RigidObject3DMotionFromPolaris: no time data in list mode file");

  const unsigned long num_mt_tags = mt_file_ptr->num_tags();
  if (num_mt_tags==0)
    error("RigidObject3DMotionFromPolaris: no data in polaris file");


  /* Determine location of LM random numbers in Motion Tracking list 

    WARNING: assumes that mt is started BEFORE lm
  */
  for (long int mt_offset = 0; mt_offset + num_lm_tags <= num_mt_tags; ++mt_offset )
  {
    // check if tags match from current position
    Polaris_MT_File::const_iterator iterator_for_random_num =
      mt_file_ptr->begin_all_tags() + mt_offset;
    // check if first tag matches
    if (iterator_for_random_num->rand_num != lm_random_numbers[0])
      continue; 

    unsigned long int num_matched_tags = 1;

    float previous_mt_tag_time = iterator_for_random_num->sample_time;
    ++iterator_for_random_num;
    unsigned int lm_tag_num = 1;
    while (iterator_for_random_num!= mt_file_ptr->end_all_tags() &&
	   lm_tag_num < num_lm_tags)
      { 
	if (iterator_for_random_num->rand_num != lm_random_numbers[lm_tag_num])
	  {
	    // no match
	    num_matched_tags = 0;
	    break; // get out of loop over tags
	  }

	++num_matched_tags;	
	previous_mt_tag_time = iterator_for_random_num->sample_time;
	++iterator_for_random_num;
	++lm_tag_num;
      } // end of loop that checks current offset
    
    if (num_matched_tags!=0)
    {
      // yes, they match
      cerr << "\n\tFound " << num_matched_tags << " matching tags between mt file and list mode data\n";
      cerr << "\tEntry " << mt_offset << " in .mt file corresponds to start of list mode data \n";
      time_offset = 
	(mt_file_ptr->begin_all_tags()+mt_offset)->sample_time;

      // fit
      // note: initialise to 0 to avoid compiler warnings
      double constant = 0; double scale = 0;
      {
	VectorWithOffset<float> weights(num_matched_tags);
	weights.fill(1.F);
	// ignore first data point
	// TODO explain why
	weights[0]=0;

	// copy mt times into a vector
	VectorWithOffset<double> mt_times(num_matched_tags);
	{
	  Polaris_MT_File::const_iterator mt_iter =
	       mt_file_ptr->begin_all_tags() + mt_offset;
	  VectorWithOffset<double>::iterator mt_time_iter = mt_times.begin();
	  for (;
	       mt_time_iter != mt_times.end();
	       ++mt_iter, ++mt_time_iter)
	    *mt_time_iter =mt_iter->sample_time - time_offset;
	}
	// note: initialise to 0 to avoid compiler warnings
	double chi_square = 0;
	double variance_of_constant = 0;
	double variance_of_scale = 0;
	double covariance_of_constant_with_scale = 0;
	linear_regression(constant, scale,
			  chi_square,
			  variance_of_constant,
			  variance_of_scale,
			  covariance_of_constant_with_scale,
			  mt_times.begin(), mt_times.end(),
			  lm_times_in_millisecs.begin(),
			  weights.begin(),
			  /* use_estimated_variance = */true
                       );

	std::cout << "scale = " << scale << " +- " << sqrt(variance_of_scale)
		  << ", cst = " << constant << " +- " << sqrt(variance_of_constant)
		  << "\nchi_square = " << chi_square
		  << "\ncovariance = " << covariance_of_constant_with_scale
       << endl;

      }
      constant += time_offset;

      // do some reporting of time discrepancies 
      {
	// Find average period between Polaris samples. 
	// Used for warning about anomalous differences between sample times.
	const double expected_tag_period = 
	  ((mt_file_ptr->end_all_tags()-1)->sample_time -
	   mt_file_ptr->begin_all_tags()->sample_time)/
	  ((mt_file_ptr->end_all_tags()-1) -
	   mt_file_ptr->begin_all_tags());

	double max_deviation = 0;
	double time_of_max_deviation = 0;
	Polaris_MT_File::const_iterator mt_iter =
	  mt_file_ptr->begin_all_tags() + mt_offset;	
	unsigned int lm_tag_num = 0;
	double previous_mt_tag_time = mt_iter->sample_time;
	// skip first
	++mt_iter; ++lm_tag_num;
	while (mt_iter!= mt_file_ptr->end_all_tags() && 
	       lm_tag_num < num_lm_tags)
	{
	  const float mt_tag_time =  mt_iter->sample_time;
	  ++mt_iter;
	  const float elapsed_mt_tag_time = 
	    (mt_tag_time - previous_mt_tag_time);	 
	  if (elapsed_mt_tag_time > 1.3F * expected_tag_period)
	    {
	      warning("MT file contains a time interval (%g) that is larger than expected after time %g\n",
		      elapsed_mt_tag_time, previous_mt_tag_time);
	    }
	  previous_mt_tag_time = mt_tag_time;

	  const double lm_tag_time_in_millisecs = lm_times_in_millisecs[lm_tag_num];
	  ++lm_tag_num;

	  const double deviation = fabs(mt_tag_time - (lm_tag_time_in_millisecs*scale+constant));
	  if (deviation>max_deviation)
	    {
	      max_deviation = deviation;
	      time_of_max_deviation = lm_tag_time_in_millisecs/1000.;
	    }
	}
	warning("Max deviation between Polaris and listmode is:\n"
		"\t%g, at %g secs (in list mode time)",
		max_deviation, time_of_max_deviation);
      }
      // check if times match (old)
      {
	double max_deviation = 0;
	double time_of_max_deviation = 0;
	Polaris_MT_File::const_iterator mt_iter =
	  mt_file_ptr->begin_all_tags() + mt_offset;	
	unsigned int lm_tag_num = 0;
	while (mt_iter!= mt_file_ptr->end_all_tags() && 
	       lm_tag_num < num_lm_tags)
	{
	  const float mt_tag_time = 
	    mt_iter->sample_time;
	  ++mt_iter;
	  const double lm_tag_time_in_millisecs = lm_times_in_millisecs[lm_tag_num];
	  ++lm_tag_num;

	  const double deviation = fabs(mt_tag_time-time_offset - lm_tag_time_in_millisecs/1000.);
	  if (deviation>max_deviation)
	    {
	      max_deviation = deviation;
	      time_of_max_deviation = lm_tag_time_in_millisecs/1000.;
	    }
	}
	warning("Original Max deviation between Polaris and listmode is:\n"
		"\t%g, at %g secs (in list mode time)",
		max_deviation, time_of_max_deviation);
      }
      return;
    }
  }

  // if we get here, we didn't find a match
  error( "\n\n\t\tNo matching data found\n" ) ;  
  
}


Succeeded 
RigidObject3DMotionFromPolaris::synchronise(CListModeData& listmode_data)
{
  // TODO add CListModeData::get_filename() or so to avoid the problem below
  if (list_mode_filename.size() == 0)
    {
      cerr << "\nCould not open synchronisation file as listmode filename missing."
	   << "\nSynchronising..." << endl;
      
      do_synchronisation(listmode_data);
    }
  else
    {      
      warning("Looking for synchronisation file: Assuming that listmode corresponds to  \"%s\".\n",
	      list_mode_filename.c_str());

      const string sync_filename =
	list_mode_filename + "_" + 
	find_filename(mt_filename.c_str()) +
	".sync";

      std::ifstream sync_file(sync_filename.c_str());
      if (sync_file)
	{
	  char line[1000];
	  sync_file.getline(line, 1000);
	  double time_offset_read;
	  if (!sync_file || 
	      sscanf(line, "time offset := %lf", &time_offset_read)!=1)
	    {
	      warning("Error while reading synchronisation file \"%s\".\n"
		      "Remove file and start again\n",
		      sync_filename.c_str());
	      return Succeeded::no;
	    }
	  set_time_offset(time_offset_read);
	  cerr << "\nsynchronisation time offset read from sync file: " << get_time_offset() << endl;
	}
      else
	{
	  cerr << "\nCould not open synchronisation file " << sync_filename
	       << " for reading.\nSynchronising..." << endl;
  
	  do_synchronisation(listmode_data);
	  cerr << "\nsynchronisation time offset  " << get_time_offset() << endl;
	  std::ofstream out_sync_file(sync_filename.c_str());
	  if (!out_sync_file)
	    warning("Could not open synchronisation file %s for writing. Proceeding...\n",
		    sync_filename.c_str());
	  else
	    {
	      out_sync_file << "time offset := " << get_time_offset() << endl;
	      cerr << "\n(written to file " << sync_filename << ")\n";
	    }
	}
    }
  return Succeeded::yes;

}

template <class T>
static inline 
void 
push_back(VectorWithOffset<T>& v, const T& elem)
{
  if (v.capacity()==v.size())
    v.reserve(v.capacity()*2);

  v.resize(v.size()+1);
  v[v.size()-1] = elem;
}

void
find_and_store_gate_tag_values_from_lm(VectorWithOffset<unsigned long>& lm_time, 
				       VectorWithOffset<unsigned>& lm_random_number, 
				       CListModeData& listmode_data/*const string& lm_filename*/)
{
  
  unsigned  LastChannelState=0;
  unsigned  ChState;
  int PulseWidth = 0 ;
  unsigned long StartPulseTime=0;
 
  
  // TODO make sure that enough events are read for synchronisation
  unsigned long max_num_events = 1UL << (8*sizeof(unsigned long)-1);
  //unsigned long max_num_events = 100000;
  long more_events = max_num_events;
  
  // reset listmode to the beginning 
  listmode_data.reset();
  
  shared_ptr <CListRecord> record_sptr = listmode_data.get_empty_record_sptr();
  CListRecord& record = *record_sptr;
  while (more_events)
  {

    if (listmode_data.get_next_record(record) == Succeeded::no) 
    {
       break; //get out of while loop
    }
    if (record.is_time())
    {
      unsigned CurrentChannelState =  record.time().get_gating();
      unsigned long CurrentTime = record.time().get_time_in_millisecs();
      
      if ( LastChannelState != CurrentChannelState && CurrentChannelState )
      {
	if ( PulseWidth > 5 ) //TODO get rid of number 5
	{
	  push_back(lm_random_number,ChState);
	  push_back(lm_time,StartPulseTime);
	}
	LastChannelState = CurrentChannelState ;
	PulseWidth = 0 ;
      }
      else if ( LastChannelState == CurrentChannelState && CurrentChannelState )
      {
	if ( !PulseWidth ) StartPulseTime = CurrentTime ;
	ChState = LastChannelState ;
	PulseWidth += 1 ;
      }
    }
    more_events-=1;
  }

  int s = lm_random_number.size();

  //for ( int i = 1; i<= lm_random_number.size(); i++)
    //cerr << lm_random_number[i] << "  ";
 
  if (s <=1)
    error("RigidObject3DMotionFromPolaris: No random numbers stored from lm file \n");

  //cerr << " LM random number" << endl; 

  //for ( int i = 0;i<=10;i++)
   //cerr << lm_random_number[i] << "  " ;

  // reset listmode to the beginning 
  listmode_data.reset();
 
}

void 
RigidObject3DMotionFromPolaris::set_defaults()
{
  RigidObject3DMotion::set_defaults();
  mt_filename = "";
  transformation_from_scanner_coordinates_filename = "";
}


void 
RigidObject3DMotionFromPolaris::initialise_keymap()
{
  RigidObject3DMotion::initialise_keymap();
  parser.add_start_key("Rigid Object 3D Motion From Polaris Parameters");
  parser.add_key("mt filename", &mt_filename);
  parser.add_key("transformation_from_scanner_coordinates_filename",
		 &transformation_from_scanner_coordinates_filename);
  parser.add_stop_key("End Rigid Object 3D Motion From Polaris");
}

bool RigidObject3DMotionFromPolaris::post_processing()
{
  mt_file_ptr = new Polaris_MT_File(mt_filename);
  if (is_null_ptr(mt_file_ptr))
    {
      warning("Error initialising mt file \n");
      return true;
    }
  {
    std::ifstream move_from_scanner_file(transformation_from_scanner_coordinates_filename.c_str());
    if (!move_from_scanner_file.good())
      {
	warning("Error reading transformation_from_scanner_coordinates_filename: '%s'",
		transformation_from_scanner_coordinates_filename.c_str());
	return true;
      }
    Quaternion<float> quat;
    CartesianCoordinate3D<float> trans;
    move_from_scanner_file >> quat;
    
    move_from_scanner_file >> trans;
    if (!move_from_scanner_file.good())
      {
	warning("Error reading transformation_from_scanner_coordinates_filename: '%s'",
		transformation_from_scanner_coordinates_filename.c_str());
	return true;
      }

    move_from_scanner_coords = 
      RigidObject3DTransformation(quat, trans);
    cerr << "'Move_from_scanner' quaternion  " << move_from_scanner_coords.get_quaternion()<<endl;
    cerr << "'Move_from_Scanner' translation  " << move_from_scanner_coords.get_translation()<<endl;
    move_to_scanner_coords = move_from_scanner_coords.inverse();
  }


  if (RigidObject3DMotion::post_processing()==true)
    return true;
  return false;
}

END_NAMESPACE_STIR


