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
#include "local/stir/listmode/CListRecord.h"
#include "stir/KeyParser.h"
#include "local/stir/motion/Polaris_MT_File.h"

#include <iostream>
#define MAX_STRING_LENGTH 512

#ifndef STIR_NO_NAMESPACES
using std::iostream;
using std::streampos;
#endif

START_NAMESPACE_STIR

const char * const 
RigidObject3DMotionFromPolaris::registered_name = "Motion From Polaris"; 

RigidObject3DMotionFromPolaris::RigidObject3DMotionFromPolaris()
{}

RigidObject3DMotionFromPolaris::RigidObject3DMotionFromPolaris(const string mt_filename_v,shared_ptr<Polaris_MT_File> mt_file_ptr_v)
{
  mt_file_ptr = mt_file_ptr_v;
  mt_filename = mt_filename_v;

}

RigidObject3DTransformation
RigidObject3DMotionFromPolaris::compute_average_motion(const float start_time, const float end_time) const
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
  
  total_q /=samples;
  total_q.normalise();
  total_tx /= samples; 
  total_ty /= samples;
  total_tz /= samples;
  
  return RigidObject3DTransformation(total_q,
				     CartesianCoordinate3D<float>(total_tz,total_ty,total_tx));  
}

void 
RigidObject3DMotionFromPolaris::
get_motion(RigidObject3DTransformation& ro3dtrans, const float time) const
{
   Polaris_MT_File::const_iterator iterator_for_record_just_after_this_time =
    mt_file_ptr->begin();

  while (iterator_for_record_just_after_this_time!= mt_file_ptr->end() &&
         iterator_for_record_just_after_this_time->sample_time < time + Polaris_time_offset)
    ++iterator_for_record_just_after_this_time;

  if (iterator_for_record_just_after_this_time == mt_file_ptr->end())
  {
    error("RigidObject3DMotionFromPolaris: reached the end of the file");
  }
  else
  {
  RigidObject3DTransformation ro3dtrans_tmp (iterator_for_record_just_after_this_time->quat,
              iterator_for_record_just_after_this_time->trans);
  ro3dtrans=ro3dtrans_tmp;
  }

}


void 
RigidObject3DMotionFromPolaris::find_offset(CListModeData& listmode_data)
{
#ifdef NEWOFFSET
  vector<float> lm_times;
  vector<unsigned> lm_random_numbers;
  find_and_store_gate_tag_values_from_lm(lm_times,lm_random_numbers,listmode_data); 
  cerr << "done find and store gate tag values" << endl;
  const vector<unsigned>::size_type num_lm_tags = lm_random_numbers.size() ;
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
    float previous_lm_tag_time = lm_times[0];
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
		   lm_times[lm_tag_num] - previous_lm_tag_time < elapsed_mt_tag_time)
	      ++lm_tag_num;
#else
	    warning("MT file contains a too large time interval (%g) after time %g\n",
		    elapsed_mt_tag_time, previous_mt_tag_time);
#endif
	  }
	if (lm_tag_num >= num_lm_tags)
	  break; // get out of while loop

	// check time consistency
	const float elapsed_lm_tag_time =
	  lm_times[lm_tag_num] - previous_lm_tag_time;

	if (fabs((elapsed_lm_tag_time - elapsed_mt_tag_time)/expected_tag_period) >= 1.F)
	  error ("Time desynchronisation between mt file and lm file after MT time %g\n",
		 previous_mt_tag_time);

	if (iterator_for_random_num->rand_num != lm_random_numbers[lm_tag_num])
	  {
	    // no match
	    num_matched_tags = 0;
	    break; // get out of loop over tags
	  }

	++num_matched_tags;	
	previous_mt_tag_time = iterator_for_random_num->sample_time;
	++iterator_for_random_num;
	previous_lm_tag_time = lm_times[lm_tag_num];
	++lm_tag_num;
      } // end of loop that checks current offset
    
    if (num_matched_tags!=0)
    {
      // yes, they match
      cerr << "\n\tFound " << num_matched_tags << " matching tags between mt file and listmode data\n";
      cerr << "\tEntry " << mt_offset << " in .mt file corresponds to Time 0 \n";
      Polaris_time_offset = (*mt_file_ptr)[mt_offset].sample_time;
      cerr<< "\tPolaris time offset is:  " <<  Polaris_time_offset << endl;
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
  
  vector<float> lm_times;
  vector<unsigned> lm_random_numbers;
  vector<unsigned> mt_random_numbers;
  // LM_file tags + times
  find_and_store_gate_tag_values_from_lm(lm_times,lm_random_numbers,listmode_data); 
  cerr << "done find and store gate tag values" << endl;
  nTags = lm_random_numbers.size();
  // to be consistent with Peter's code
  nTags -=1;
  //MT_file random numbers
  //cerr << " Reading mt file" << endl;    
  find_and_store_random_numbers_from_mt_file(mt_random_numbers);
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
  else
  {
   printf( "\t\tEntry %6d in .mt file Corresponds to Time 0 \n", ZeroOffSet) ;
  }
  
  
  int mt_offset = ZeroOffSet;
  Polaris_time_offset = (*mt_file_ptr)[mt_offset].sample_time;
  cerr<< "\tPolaris time offset is:  " <<  Polaris_time_offset << endl;

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
  
  find_offset(listmode_data);
  return Succeeded::yes;

}


void
RigidObject3DMotionFromPolaris::find_and_store_gate_tag_values_from_lm(vector<float>& lm_time, 
								       vector<unsigned>& lm_random_number, 
								       CListModeData& listmode_data/*const string& lm_filename*/)
{
  
  unsigned  LastChannelState=0;
  unsigned  ChState;
  int PulseWidth = 0 ;
  //long int CurrentTime;
  //long int StartPulseTime;
  double StartPulseTime=0;
  //int NumTag = 0 ;
 
  
  // TODO make sure that enough events is read for synchronisation
  unsigned long max_num_events = 1UL << 8*sizeof(unsigned long)-1;
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
      unsigned CurrentChannelState =  record.time().get_gating() ;
      double CurrentTime = record.time().get_time_in_secs();
      
      if ( LastChannelState != CurrentChannelState && CurrentChannelState )
      {
	if ( PulseWidth > 5 ) //TODO get rid of number 5
	{
	  lm_random_number.push_back(ChState);
	  lm_time.push_back(StartPulseTime);
	  //NumTag += 1 ;
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
RigidObject3DMotionFromPolaris::find_and_store_random_numbers_from_mt_file(vector<unsigned>& mt_random_numbers)
{
  //Polaris_MT_File::Record mt_record;
  Polaris_MT_File::const_iterator iterator_for_random_num =
    mt_file_ptr->begin_all_tags();
  while (iterator_for_random_num!= mt_file_ptr->end_all_tags())
  {
    mt_random_numbers.push_back(iterator_for_random_num->rand_num);
    ++iterator_for_random_num;
  }
  
}

#if 1
void 
RigidObject3DMotionFromPolaris::set_defaults()
{
  mt_filename = "";
}


void 
RigidObject3DMotionFromPolaris::initialise_keymap()
{
  parser.add_start_key("Start Rigid Object3D Motion From Polaris");
  parser.add_key("mt filename", &mt_filename);
  parser.add_stop_key("End Rigid Object3D Motion From Polaris");

}

bool RigidObject3DMotionFromPolaris::post_processing()
{

  mt_file_ptr = new Polaris_MT_File(mt_filename);
  return false;
}


Succeeded 
RigidObject3DMotionFromPolaris::set_polaris_time_offset(float mt_offset)
{
  float time_offset = (*mt_file_ptr)[mt_offset].sample_time;
  Polaris_time_offset=time_offset;
  return Succeeded::yes;
}

#endif

END_NAMESPACE_STIR


