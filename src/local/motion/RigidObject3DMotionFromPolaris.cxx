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

#include <fstream>
#include <ctime>

# ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::time_t; using ::tm; using ::localtime; }
#endif


#ifndef STIR_NO_NAMESPACES
using std::iostream;
using std::streampos;
#endif

START_NAMESPACE_STIR
#define  NEWOFFSET

// Find and store gating values in a vector from lm_file  
static  void 
find_and_store_gate_tag_values_from_lm(VectorWithOffset<unsigned long>& lm_times_in_millisecs, 
				       VectorWithOffset<unsigned>& lm_random_number,
				       CListModeData& listmode_data);
#ifndef NEWOFFSET
// Find and store random numbers from mt_file
static void 
find_and_store_random_numbers_from_mt_file(VectorWithOffset<unsigned>& mt_random_numbers,
					   Polaris_MT_File& mt_file);
#endif

const char * const 
RigidObject3DMotionFromPolaris::registered_name = "Motion From Polaris"; 

RigidObject3DMotionFromPolaris::RigidObject3DMotionFromPolaris()
{
  set_defaults();
}

RigidObject3DMotionFromPolaris::
RigidObject3DMotionFromPolaris(const string mt_filename_v,
			       shared_ptr<Polaris_MT_File> mt_file_ptr_v)
{
  mt_file_ptr = mt_file_ptr_v;
  mt_filename = mt_filename_v;
  // TODO
  error("constructor does not work yet");

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
    iter++;
  }
  /* Average quat and translation */
 
  if (samples==0)
    {
      warning("Start-end range does not seem to overlap with MT info.\n"
	      "Reference transformation set to identity, but this is WRONG.\n");
      return RigidObject3DTransformation(Quaternion<float>(1,0,0,0), CartesianCoordinate3D<float>(0,0,0));
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
RigidObject3DMotionFromPolaris::find_offset(CListModeData& listmode_data)
{
  std::time_t sec_time = 
    listmode_data.get_scan_start_time_in_secs_since_1970();
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
  

#ifdef NEWOFFSET
  VectorWithOffset<unsigned long> lm_times_in_millisecs;
  VectorWithOffset<unsigned> lm_random_numbers;
  find_and_store_gate_tag_values_from_lm(lm_times_in_millisecs,lm_random_numbers,listmode_data); 
  cerr << "done find and store gate tag values" << endl;
  // TODO remove
  {
    std::ofstream lmtimes("lmtimes.txt");
    lmtimes << lm_times_in_millisecs;
    std::ofstream lmtags("lmtags.txt");
    lmtags << lm_random_numbers;
  }
  const VectorWithOffset<unsigned>::size_type num_lm_tags = lm_random_numbers.size() ;
  // Peter has size-1 for some reason

  const unsigned long num_mt_tags = mt_file_ptr->num_tags();
  
  const float expected_tag_period = .2F; // TODO move to Polaris_MT_File
  /* Determine location of LM random numbers in Motion Tracking list 

    WARNING: assumes that mt is started BEFORE lm, and stopped AFTER 
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
    unsigned long previous_lm_tag_time_in_millisecs = lm_times_in_millisecs[0];
    unsigned int lm_tag_num = 1;
    while (iterator_for_random_num!= mt_file_ptr->end())
      {
	const float elapsed_mt_tag_time = 
	  (iterator_for_random_num->sample_time - previous_mt_tag_time);
	if (elapsed_mt_tag_time > 1.3F * expected_tag_period)
	  {
#if 0
	    cerr << "skipping 'missing data' after MT time " << previous_mt_tag_time << '\n' ;
	    while (lm_tag_num < num_lm_tags &&
		   lm_times_in_millisecs[lm_tag_num] - previous_lm_tag_time_in_millisecs < elapsed_mt_tag_time)
	      ++lm_tag_num;
#else
	    warning("MT file contains a too large time interval (%g) after time %g\n",
		    elapsed_mt_tag_time, previous_mt_tag_time);
#endif
	  }
	if (lm_tag_num >= num_lm_tags)
	  break; // get out of while loop

	if (iterator_for_random_num->rand_num != lm_random_numbers[lm_tag_num])
	  {
	    // no match
	    num_matched_tags = 0;
	    break; // get out of loop over tags
	  }

	++num_matched_tags;	
	previous_mt_tag_time = iterator_for_random_num->sample_time;
	++iterator_for_random_num;
	previous_lm_tag_time_in_millisecs = lm_times_in_millisecs[lm_tag_num];
	++lm_tag_num;
      } // end of loop that checks current offset
    
    if (num_matched_tags!=0)
    {
      // yes, they match
      cerr << "\n\tFound " << num_matched_tags << " matching tags between mt file and listmode data\n";
      cerr << "\tEntry " << mt_offset << " (not counting missing data) in .mt file corresponds to Time 0 \n";
      time_offset = 
	(mt_file_ptr->begin_all_tags()+mt_offset)->sample_time;

      // check if times match 
      {
	double max_deviation = 0;
	double time_of_max_deviation = 0;
	Polaris_MT_File::const_iterator iterator_for_random_num =
	  mt_file_ptr->begin_all_tags() + mt_offset;	
	unsigned int lm_tag_num = 0;
	while (iterator_for_random_num!= mt_file_ptr->end() && 
	       lm_tag_num < num_lm_tags)
	{
	  const float mt_tag_time = 
	    iterator_for_random_num->sample_time;
	  ++iterator_for_random_num;
	  const double lm_tag_time_in_millisecs = lm_times_in_millisecs[lm_tag_num];
	  ++lm_tag_num;

	  const double deviation = fabs(mt_tag_time-time_offset - lm_tag_time_in_millisecs/1000.);
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
      return;
    }
  }

  // if we get here, we didn't find a match
  error( "\n\n\t\tNo matching data found\n" ) ;  
#else 
  int Total;
  int i1;
  
  long int nTags = 0;
  long int nMT_Rand = 0;
  
  VectorWithOffset<float> lm_times_in_millisecs;
  VectorWithOffset<unsigned> lm_random_numbers;
  VectorWithOffset<unsigned> mt_random_numbers;
  // LM_file tags + times
  find_and_store_gate_tag_values_from_lm(lm_times_in_millisecs,lm_random_numbers,listmode_data); 
  cerr << "done find and store gate tag values" << endl;
  nTags = lm_random_numbers.size();
  // to be consistent with Peter's code
  nTags -=1;
  //MT_file random numbers
  //cerr << " Reading mt file" << endl;    
  find_and_store_random_numbers_from_mt_file(mt_random_numbers, *mt_file_ptr);
  //cerr << " Done reading mt file" << endl;
  nMT_Rand = mt_random_numbers.size();
   //cerr << "Random_num" << nMT_Rand;
  
  /* Determine location of LM random numbers in Motion Tracking list */
  long int OffSet = 0 ;
  int ZeroOffSet = -1;
  while ( OffSet + nTags < nMT_Rand )
  {
    for ( Total = 0 , i1 = 0 ; i1 < nTags ; i1++ )
    {
      if (mt_random_numbers[i1 + OffSet]!= lm_random_numbers[i1])
      {
	Total = 1 ;
	break; // get out: no match
      }
    }
    if ( !Total )
    {
      ZeroOffSet = OffSet ;
      OffSet += nMT_Rand ;
    }
    OffSet += 1 ;
  }
  if ( ZeroOffSet < 0 )
  {
    error( "\n\n\t\tNo matching data found\n" ) ;
  }
 
  
  int mt_offset = ZeroOffSet;
  time_offset = 
    (mt_file_ptr->begin_all_tags()+mt_offset)->sample_time;
  cerr << "\t\tEntry " << mt_offset << "(ignoring missing data lines) in .mt file Corresponds to Time 0 \n"
       << "\t\tTime offset:= " << time_offset << endl;
#endif

#if 0
  cerr << endl;
  cerr << "MT random numbers" << "   ";
  for ( int i = mt_offset; i<=mt_offset+10; i++)
   cerr << (*mt_file_ptr)[i].rand_num << "   ";
#endif
  
}


Succeeded 
RigidObject3DMotionFromPolaris::synchronise(CListModeData& listmode_data)
{
  // TODO add CListModeData::get_filename() or so to avoid the problem below
  if (list_mode_filename.size() == 0)
    {
      cerr << "\nCould not open synchronisation file as listmode filename missing."
	   << "\nSynchronising..." << endl;
      
      find_offset(listmode_data);
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
  
	  find_offset(listmode_data);
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

#ifdef NEWOFFSET  
void 
find_and_store_random_numbers_from_mt_file(VectorWithOffset<unsigned>& mt_random_numbers,
					   Polaris_MT_File& mt_file)
{
  Polaris_MT_File::const_iterator iterator_for_random_num =
    mt_file.begin_all_tags();
  while (iterator_for_random_num!= mt_file.end_all_tags())
  {
    push_back(mt_random_numbers,iterator_for_random_num->rand_num);
    ++iterator_for_random_num;
  }
  
}
#endif

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


