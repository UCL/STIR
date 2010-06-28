//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet
    For internal GE use only.
*/
/*!
  \file 
  \ingroup motion

  \brief Implementation of class stir::RigidObject3DMotionFromPolaris
 
  \author Sanida Mustafovic
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/

#include "local/stir/motion/RigidObject3DMotionFromPolaris.h"
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/Quaternion.h"
#include "stir/listmode/CListRecord.h"
#include "local/stir/motion/Polaris_MT_File.h"
#include "stir/VectorWithOffset.h"
#include "stir/utilities.h"
#include "stir/stream.h"
#include "stir/is_null_ptr.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/linear_regression.h"
#include <fstream>
#include <ctime>

#ifndef BOOST_NO_STRINGSTREAM
#include <sstream>
#else
#include <strstream>
#endif

# ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::time_t; using ::tm; using ::localtime; }
#endif

START_NAMESPACE_STIR

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


static const double time_not_yet_determined=-1234567.8;

/*! Convert from Polaris transformation to STIR conventions

   The Polaris records info as q0 qx qy qz tx ty tz, where the
   coordinate system is right-handed (see the Polaris manual).
   The STIR coordinate system is left-handed. We do take
   a coordinate transformation into account when applying 
   Polaris transformations to scanner coordinates. However,
   that coordinate transformation uses RigidObject3DTransformation,
   and hence cannot swap right-handed to left-handed.
   So, we convert the Polaris transformation to a left-handed
   transformation by swapping x and y.

   The exact form then depends heavily on conventions in RigidObject3DTransformation,
   which currently takes its info as
   \code
   Quaternion<float>(q0,qZ,qY,qX), CartesianCoordinate3D<float>(tZ,tY,tX)
   \endcode
   where I've used capitals to denote the left-handed coordinate system:
   \code
    x=Y,y=X,z=Z.
   \endcode
   Some careful checks in e.g. Mathematica then show that
   \code
    qx=qY,qy=qX,qz=qZ.
   \endcode
   This not so obvious as it sounds as there is a potential sign here. 
   For instance, the Polaris convention uses a quaternion 
   <code>(0,x,y,z)</code> for a point,
   while RigidObject3DTransformation uses </code>(0,z,y,x)</code> 
   (see the point2quat function in RigidObject3DTransformation.cxx ).
   So, effectively there are 2 coordinate swaps between Polaris conventions and 
   RigidObject3DTransformation:
   \code
   Quaternion<float>(q0,qZ,qY,qX) = Quaternion<float>(q0,qz,qx,qy)
   \endcode
   and <code>qz,qx,qy</code> is an even permutation of <code>qx,qy,qz</code>
*/
RigidObject3DTransformation
RigidObject3DMotionFromPolaris::
make_transformation_from_polaris_data(Polaris_MT_File::Record const& record)
{
  /* WARNING:
     This depends conventions in RigidObject3DTransformation.

     Note that Polaris_MT_File already sets the record.trans as (tz,ty,tx).

     Some other examples:

     point2quat(z,y,x)->(0,x,y,z) and DO_XY_SWAP (i.e. explicit xy swap in transform_point)
     return RigidObject3DTransformation(record.quat, record.trans);

     point2quat(z,y,x)->(0,z,y,x) and no DO_XY_SWAP
     return 
        RigidObject3DTransformation(Quaternion<float>(record.quat[1],
	                                              -record.quat[3],
						      -record.quat[2],
						      -record.quat[4]),
				CartesianCoordinate3D<float>(record.trans.z(),
							     record.trans.x(),
							     record.trans.y()));
  */ 
  return
    RigidObject3DTransformation(Quaternion<float>(record.quat[1],
						  record.quat[4],
						  record.quat[2],
						  record.quat[3]),
				CartesianCoordinate3D<float>(record.trans.z(),
							     record.trans.x(),
							     record.trans.y()));
}

// Find and store gating values in a vector from lm_file  
static  void 
find_and_store_gate_tag_values_from_lm(VectorWithOffset<unsigned long>& lm_times_in_millisecs, 
				       VectorWithOffset<unsigned>& lm_random_number,
				       CListModeData& listmode_data,
				       const unsigned int mask_for_tags);

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


double 
RigidObject3DMotionFromPolaris::
rel_time_to_polaris_time(const double time) const
{
  return time*time_drift + time_offset;
}

double 
RigidObject3DMotionFromPolaris::
polaris_time_to_rel_time(const double time) const
{
  return (time - time_offset)/time_drift;
}

std::vector<double>
RigidObject3DMotionFromPolaris::
get_rel_time_of_samples(const double start_time, const double end_time) const
{
  const double polaris_start_time = this->rel_time_to_polaris_time(start_time);
  const double polaris_end_time = this->rel_time_to_polaris_time(end_time);
  std::vector<double> result;
  Polaris_MT_File::const_iterator iter=this->mt_file_ptr->begin();

  while (iter!= this->mt_file_ptr->end() && iter->sample_time< polaris_start_time)
    ++iter;

  while (iter!= this->mt_file_ptr->end() && iter->sample_time<= polaris_end_time)
  {
    result.push_back(this->polaris_time_to_rel_time(iter->sample_time));
    ++iter;
  }
  return result;
}


RigidObject3DTransformation
RigidObject3DMotionFromPolaris::
compute_average_motion_polaris_time(const double start_time, const double end_time) const
{
  // CartesianCoordinate3D<float> euler_angles;
  int samples = 0 ;
  CartesianCoordinate3D<float> total_t(0,0,0);
  Quaternion<float> total_q(0,0,0,0);
 
  Polaris_MT_File::const_iterator iter=mt_file_ptr->begin();

  while (iter!= mt_file_ptr->end())
  {
    /* Accept motions recorded during time interval */
    if ((iter->sample_time >= start_time ) && ( iter->sample_time<= end_time))
    {
     RigidObject3DTransformation transf =
       make_transformation_from_polaris_data(*iter);
     Quaternion<float> quater = transf.get_quaternion();
     const CartesianCoordinate3D<float> trans= transf.get_translation();
      // make sure that all quaternions use a fixed sign choice, otherwise adding them up does not make a lot of sense
      if (quater[1]<0)
	quater *= -1;
      /* Maintain running total quaternions and translations */
      total_t += trans;
      total_q += quater;
      samples += 1;
    }
    ++iter;
  }
  /* Average quat and translation */
 
  if (samples==0)
    {
      error("RigidObject3DMotionFromPolaris::compute_average_motion_polaris_time:\n"
	    "\t Start-end range (%g-%g) does not seem to overlap with MT info.",
	    start_time, end_time);	     
    }
  
  total_q /=static_cast<float>(samples);
  if (norm(total_q)<.9)
    warning("RigidObject3DMotionFromPolaris::compute_average_motion_polaris_time:\n"
	    "\taveraged quaternion has norm %g which is very different from 1.\n"
	    "\tThis indicates large movement in the range (%g-%g).",
	    norm(total_q), start_time, end_time);
  total_q.normalise();
  total_t /= samples; 
  
  return RigidObject3DTransformation(total_q, total_t);
}

RigidObject3DTransformation 
RigidObject3DMotionFromPolaris::
compute_average_motion_in_tracker_coords_rel_time(const double start_time, const double end_time) const
{
  return compute_average_motion_polaris_time(rel_time_to_polaris_time(start_time),
					     rel_time_to_polaris_time(end_time));
}

RigidObject3DTransformation 
RigidObject3DMotionFromPolaris::
get_motion_in_tracker_coords_rel_time(const double time) const
{
  const double polaris_time =
    rel_time_to_polaris_time(time);

  Polaris_MT_File::const_iterator iterator_for_record_just_after_this_time =
    mt_file_ptr->begin();

  while (iterator_for_record_just_after_this_time!= mt_file_ptr->end() &&
         iterator_for_record_just_after_this_time->sample_time < polaris_time)
    ++iterator_for_record_just_after_this_time;

  if (iterator_for_record_just_after_this_time == mt_file_ptr->end())
  {
    error("RigidObject3DMotionFromPolaris: motion asked for time %g which is "
	  "beyond the range of data (time in Polaris units: %g)\n",
	  time, polaris_time);
    // statement to avoid compiler warning
    return RigidObject3DTransformation();
  }
  else
  {
    return
      make_transformation_from_polaris_data(*iterator_for_record_just_after_this_time);
  }

}


void 
RigidObject3DMotionFromPolaris::
do_synchronisation(CListModeData& listmode_data)
{
  VectorWithOffset<unsigned long> lm_times_in_millisecs;
  VectorWithOffset<unsigned> lm_random_numbers;
  find_and_store_gate_tag_values_from_lm(lm_times_in_millisecs,lm_random_numbers,listmode_data, this->_mask_for_tags); 
  cout << "done find and store gate tag values" << endl;
  const VectorWithOffset<unsigned>::size_type num_lm_tags = lm_random_numbers.size() ;
  if (num_lm_tags==0)
    error("RigidObject3DMotionFromPolaris: no time data in list mode file");

  const unsigned long num_mt_tags = mt_file_ptr->num_tags();
  if (num_mt_tags==0)
    error("RigidObject3DMotionFromPolaris: no data in polaris file");


  /* Determine location of LM random numbers in Motion Tracking list 

    WARNING: assumes that mt is started BEFORE lm
  */
  double start_MT_time_of_matching_sequence = 0;
  // copy mt times into a vector
  Array<1,double> mt_match_times;
  Array<1,double> lm_match_times;
  mt_match_times.reserve(lm_times_in_millisecs.size());
  lm_match_times.reserve(lm_times_in_millisecs.size());

  for (long int mt_offset = 0; mt_offset + num_lm_tags <= num_mt_tags; ++mt_offset )
  {
    // check if tags match from current position
    Polaris_MT_File::const_iterator iterator_for_random_num =
      mt_file_ptr->begin_all_tags() + mt_offset;
    // check if first tag matches
    if ((iterator_for_random_num->rand_num & this->_mask_for_tags) != lm_random_numbers[0])
      continue; 

    unsigned long int num_matched_tags = 1;
    start_MT_time_of_matching_sequence = (double)iterator_for_random_num->sample_time;

    mt_match_times.resize(0);
    lm_match_times.resize(0);
    push_back(mt_match_times, (double)start_MT_time_of_matching_sequence);
    push_back(lm_match_times, (double)lm_times_in_millisecs[0]);
    ++iterator_for_random_num;
    unsigned int lm_tag_num = 1;
    while (iterator_for_random_num!= mt_file_ptr->end_all_tags() &&
	   lm_tag_num < num_lm_tags)
      { 
	if ((iterator_for_random_num->rand_num & this->_mask_for_tags) != lm_random_numbers[lm_tag_num])
	  {
	    // no match
	    if (num_matched_tags > 10)
	      {
		warning("Matching sequence of length %d (starting at MT time %g) breaks at MT time %g",
			num_matched_tags, start_MT_time_of_matching_sequence, 
			(double)iterator_for_random_num->sample_time);
	      }
	    num_matched_tags = 0;
	    break; // get out of loop over tags
	  }
	push_back(mt_match_times, (double)iterator_for_random_num->sample_time);
	push_back(lm_match_times, (double)lm_times_in_millisecs[lm_tag_num]);

	++num_matched_tags;	

	// The code that finds the list mode tags only stores a new tag when the channels change value,
	// assuming that a different random number will be used every time. This assumption is no
	// no longer valid when a cable isn't connected.
	// We get around this problem by deleting repeated occurences in both data streams.
	// In addition, we'll ignore 0 tags (which might occur because we're masking out a channel).
	const unsigned int current_rand_num = iterator_for_random_num->rand_num & this->_mask_for_tags;
	++iterator_for_random_num;
	while (iterator_for_random_num!= mt_file_ptr->end_all_tags() &&
	       ((iterator_for_random_num->rand_num & this->_mask_for_tags) == current_rand_num ||
		(iterator_for_random_num->rand_num & this->_mask_for_tags) == 0))
	  {
	    ++iterator_for_random_num;
	  }
	const unsigned int current_lm_rand_num = lm_random_numbers[lm_tag_num]  & this->_mask_for_tags;
	++lm_tag_num;
	while (lm_tag_num < num_lm_tags &&
	       ((lm_random_numbers[lm_tag_num] & this->_mask_for_tags) == current_lm_rand_num ||
		(lm_random_numbers[lm_tag_num] & this->_mask_for_tags) == 0))
	  {
	    ++lm_tag_num;
	  }
      } // end of loop that checks current offset
    
    if (num_matched_tags!=0)
    {
      // yes, they match
      cout << "\n\tFound " << num_matched_tags << " matching tags between mt file and list mode data\n";
      cout << "\tEntry " << mt_offset << " in .mt file (MT time "
	   << start_MT_time_of_matching_sequence << ") corresponds to start of list mode data \n";
      this->time_offset = start_MT_time_of_matching_sequence;

      // fit
      {
	// note: initialise to 0 to avoid compiler warnings
	double constant = 0; double scale = 0;
	// first shift mt times according to our initial estimate
	// and scale lm_times to secs.
	// This will make the fit a bit more stable.
	mt_match_times -= this->time_offset;
	lm_match_times /= 1000.;

	VectorWithOffset<float> weights(num_matched_tags);
	weights.fill(1.F);
	// ignore first data point
	// TODO explain why
	weights[0]=0;

	// copy mt times into a vector
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
			  mt_match_times.begin(), mt_match_times.end(),
			  lm_match_times.begin(),
			  weights.begin(),
			  /* use_estimated_variance = */true
                       );

	std::cout << "\tscale = " << scale << " +- " << sqrt(variance_of_scale)
		  << ", cst = " << constant << " +- " << sqrt(variance_of_constant)
		  << "\n\tchi_square = " << chi_square
		  << "\n\tcovariance = " << covariance_of_constant_with_scale
		  << endl;
	this->time_offset += constant;
	this->time_drift = scale;

	// report max-difference
	{
	  Array<1,double> diff = lm_match_times * scale + constant - mt_match_times;
	  // ignore first
	  diff.resize(1,diff.size()-1);
	  std::cout << "\nDeviation between Polaris and listmode time is between "
		    << diff.find_min() << " and " << diff.find_max() << '\n';
	}
      } // end of fit

      std::cout << "\n\tTime offset " <<  time_offset << " drift " << time_drift << '\n';
      // do some reporting of time discrepancies 
      {
	// Find average period between Polaris samples. 
	// Used for warning about anomalous differences between sample times.
	const double expected_tag_period = 
	  ((mt_file_ptr->end_all_tags()-1)->sample_time -
	   mt_file_ptr->begin_all_tags()->sample_time)/
	  ((mt_file_ptr->end_all_tags()-1) -
	   mt_file_ptr->begin_all_tags());

	Polaris_MT_File::const_iterator mt_iter =
	  mt_file_ptr->begin_all_tags() + mt_offset;	
	double previous_mt_tag_time = mt_iter->sample_time;
	while (mt_iter!= mt_file_ptr->end_all_tags() && 
	       lm_tag_num < num_lm_tags)
	{
	  const float mt_tag_time =  mt_iter->sample_time;
	  ++mt_iter;
	  const float elapsed_mt_tag_time = 
	    (mt_tag_time - previous_mt_tag_time);	 
	  if (elapsed_mt_tag_time > 1.3F * expected_tag_period)
	    {
	      warning("MT file contains a time interval (%g) that is larger than expected after time %g",
		      elapsed_mt_tag_time, previous_mt_tag_time);
	    }
	  previous_mt_tag_time = mt_tag_time;
	}
      }
      return;
    }
  }

  // if we get here, we didn't find a match
  warning( "No matching data found" );
  std::cerr << "Some diagnostics\n";
  {
    const std::time_t listmode_data_start_time_in_secs = 
      listmode_data.get_scan_start_time_in_secs_since_1970();
    if (listmode_data_start_time_in_secs!=std::time_t(-1))
      {
	std::cerr << "List mode data started at " 
		  << listmode_data_start_time_in_secs
		  << " secs since 1970\n";
	// Polaris times are currently in localtime since midnight
	// This relies on TZ though: bad! (TODO)
	struct std::tm* lm_start_time_tm = std::localtime( &listmode_data_start_time_in_secs  ) ;
	const double lm_start_time = 
	  ( lm_start_time_tm->tm_hour * 3600. ) + 
	  ( lm_start_time_tm->tm_min * 60. ) + 
	  lm_start_time_tm->tm_sec ;
	
	std::cerr <<"Listmode file says that listmode start time is " 
		  << lm_start_time 
		  << " in secs after midnight local time"<< endl;
      }
    else
      {
	std::cerr <<"Listmode file has scan_start_time not filled in\n";
      }
    std::cerr << "Polaris tracking started at " 
	      << mt_file_ptr->get_start_time_in_secs_since_1970()
	      << " secs since 1970\n";
    std::cerr << "Polaris first and last tags are at " 
	      << mt_file_ptr->begin_all_tags()->sample_time << ", "
	      << (mt_file_ptr->end_all_tags()-1)->sample_time 
	      << " in secs after midnight local time\n";

    std::cerr << "\nFirst 50 list mode tags:\n";
    for (unsigned int lm_tag_num = 1; lm_tag_num <= std::min(std::size_t(50),num_lm_tags); ++lm_tag_num)
      { 
	std::cerr << lm_random_numbers[lm_tag_num] << ", ";
      }
    std::cerr << "... \n";
  }      
  error( "\n\t\tNo matching data found" ) ;  
  
}

void 
RigidObject3DMotionFromPolaris::
set_mask_for_tags(const unsigned int mask_for_tags)
{
  this->_mask_for_tags = mask_for_tags;
}


Succeeded 
RigidObject3DMotionFromPolaris::synchronise()
{
  if (this->list_mode_filename.size() == 0)
    {
      warning("Could not synchronise as listmode filename missing.");
      return Succeeded::no;
    }

  this->listmode_data_start_time_in_secs=std::time_t(-1);

  try
    {
      shared_ptr<CListModeData> lm_data_ptr =
	CListModeData::read_from_file(list_mode_filename);

      this->listmode_data_start_time_in_secs = 
	lm_data_ptr->get_scan_start_time_in_secs_since_1970();

      if (this->listmode_data_start_time_in_secs==std::time_t(-1))
	{
	  warning("Scan start time could not be found from list mode data");
	}
    }
  catch (...)
    {
      warning("List mode file \"%s\" not found or in incorrect format", this->list_mode_filename.c_str());
    }
    

  {
    // TODO warning->info
    warning("Looking for synchronisation file: Assuming that listmode corresponds to  \"%s\".",
	    this->list_mode_filename.c_str());

    const string sync_filename =
      this->list_mode_filename + "_" + 
      get_filename(this->mt_filename) +
      ".sync";

    // define parser for .sync file format
    const int current_sync_version = 2;
    int sync_version=-1;
    KeyParser parser;
    parser.add_start_key("Polaris vs. list mode synchronisation file");
    parser.add_stop_key("end");
    parser.add_key("version", &sync_version);
    parser.add_key("time offset", &this->time_offset);
    parser.add_key("time drift", &this->time_drift);
      
    this->time_offset =time_not_yet_determined;
    this->time_drift = 1;

    std::ifstream sync_file(sync_filename.c_str());
    if (sync_file)
      {
	if (parser.parse(sync_file) == false || 
	    sync_version!=current_sync_version ||
	    this->time_offset == time_not_yet_determined)
	  {
	    warning("RigidObject3DMotionFromPolaris: Error while reading synchronisation file \"%s\".\n"
		    "Remove file and start again.",
		    sync_filename.c_str());
	    if (sync_version!=current_sync_version)
	      warning("Reason: version should be %d", current_sync_version);
	    if (this->time_offset == time_not_yet_determined)
	      warning("Reason: time_offset not set");
	    return Succeeded::no;
	  }
      }
    else
      {
	// TODO warning-<info
	warning("Could not open synchronisation file %s  for reading.\nSynchronising...",
		sync_filename.c_str());

	shared_ptr<CListModeData> lm_data_ptr =
	  CListModeData::read_from_file(this->list_mode_filename);
      
	this->do_synchronisation(*lm_data_ptr);

	// write info to .sync file
	std::ofstream out_sync_file(sync_filename.c_str());
	if (!out_sync_file)
	  {
	    warning("Could not open synchronisation file %s for writing. Proceeding...\n",
		    sync_filename.c_str());
	  }
	else
	  {
	    // set variable such that the correct number will be written to file
	    sync_version=current_sync_version;
	    out_sync_file << parser.parameter_info();
	    warning("Synchronisation written to file %s", sync_filename.c_str());
	  }	
      }  
    
    // write sync info to cerr
    std::cerr << parser.parameter_info();
  }

  if (fabs(time_drift-1) > max_time_drift_deviation)
    {
      warning("RigidObject3DMotionFromPolaris: time_drift %g is too large.\n"
	      "You could change the tolerance using the 'maximum time drift deviation' keyword.",
	      time_drift);
      return Succeeded::no;
    }
  if (this->listmode_data_start_time_in_secs!=std::time_t(-1))
    {
      // Polaris times are currently in localtime since midnight
      // This relies on TZ though: bad! (TODO)
      struct std::tm* lm_start_time_tm = std::localtime( &this->listmode_data_start_time_in_secs  ) ;
      const double lm_start_time = 
	( lm_start_time_tm->tm_hour * 3600. ) + 
	( lm_start_time_tm->tm_min * 60. ) + 
	lm_start_time_tm->tm_sec ;

      cout << "\nListmode file says that listmode start time is " 
	   << lm_start_time 
	   << " in secs after midnight local time"<< endl;

      if (fabs(time_offset - lm_start_time) > max_time_offset_deviation)
	{
	  warning("RigidObject3DMotionFromPolaris: max_time_offset deviation %g is too large.\n"
	      "You could change the tolerance using the 'maximum time offset deviation' keyword.",
	      time_offset - lm_start_time);
	  return Succeeded::no;
	}
    }
  else
    {
      this->listmode_data_start_time_in_secs =
	mt_file_ptr->get_start_time_in_secs_since_1970() +
	round(time_offset - mt_file_ptr->begin_all_tags()->sample_time);
      warning("Used first line of Polaris to get absolute time info of\n"
	      "start of list mode data (ignoring time-drift):\n"
	      "estimated at %ld secs since 1970 UTC",
	      this->listmode_data_start_time_in_secs);
    }

  if (fabs(this->secs_since_1970_to_rel_time(this->listmode_data_start_time_in_secs))>.1)
    {
      warning("RigidObject3DMotionFromPolaris: internal problem with time_offsets. Sorry");
       return Succeeded::no;
    } 
  if (fabs(rel_time_to_polaris_time(this->secs_since_1970_to_rel_time(mt_file_ptr->get_start_time_in_secs_since_1970())) - 
	   mt_file_ptr->begin_all_tags()->sample_time) > max_time_offset_deviation)
    {
      warning("Polaris start of data (%g secs since midnight) does not seem to match \n"
	      "with its first time tag (%g),\n" 
	      "or there's a very large time drift. I'm stopping anyway.\n",
	      rel_time_to_polaris_time(this->secs_since_1970_to_rel_time(mt_file_ptr->get_start_time_in_secs_since_1970())),
	   mt_file_ptr->begin_all_tags()->sample_time);
      return Succeeded::no;
    }
	   
  return Succeeded::yes;

}

double 
RigidObject3DMotionFromPolaris::
secs_since_1970_to_rel_time(std::time_t secs) const
{
  // TODO WARNING assumes that list mode data starts at rel_time 0 (which is ok for 962 and 966)

  // somewhat tricky as time_t might be an unsigned type, and potentially longer than 'long'
  if (secs>this->listmode_data_start_time_in_secs)
    return
      static_cast<double>(secs-this->listmode_data_start_time_in_secs);
  else
    return
      -static_cast<double>(this->listmode_data_start_time_in_secs-secs);
}
  
void
find_and_store_gate_tag_values_from_lm(VectorWithOffset<unsigned long>& lm_time, 
				       VectorWithOffset<unsigned>& lm_random_number, 
				       CListModeData& listmode_data/*const string& lm_filename*/,
				       const unsigned int mask_for_tags)
{
  
  unsigned  LastChannelState=0;
  unsigned  ChState=0;;
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
      unsigned CurrentChannelState =  record.time().get_gating() & mask_for_tags;
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


bool 
RigidObject3DMotionFromPolaris::
is_synchronised() const
{
  return time_offset!=time_not_yet_determined;
}  

void 
RigidObject3DMotionFromPolaris::set_defaults()
{
  RigidObject3DMotion::set_defaults();
  list_mode_filename="";
  mt_filename = "";
  transformation_from_scanner_coordinates_filename = "";
  time_offset = time_not_yet_determined;
  max_time_drift_deviation = .01;
  max_time_offset_deviation = 3.;
  this->_mask_for_tags= 0xffffffff;
}


void 
RigidObject3DMotionFromPolaris::initialise_keymap()
{
  RigidObject3DMotion::initialise_keymap();
  parser.add_start_key("Rigid Object 3D Motion From Polaris Parameters");
  parser.add_key("mt filename", &mt_filename);
  parser.add_key("list_mode_filename",&list_mode_filename);

  parser.add_key("transformation_from_scanner_coordinates_filename",
		 &transformation_from_scanner_coordinates_filename);
  parser.add_key("maximum time_drift deviation",
		 &max_time_drift_deviation);
  parser.add_key("maximum time offset deviation",
		 &max_time_offset_deviation);
  parser.add_key("mask for tags",
		 &this->_mask_for_tags);
  parser.add_stop_key("End Rigid Object 3D Motion From Polaris");
}

bool RigidObject3DMotionFromPolaris::post_processing()
{
  if (set_mt_file(mt_filename) == Succeeded::no)
    {
      warning("Error initialising mt file \n");
      return true;
    }
  {
#if 0
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
#else
    {
      std::string conventions;
      std::string transformation_as_string;
      KeyParser parser;
      parser.add_start_key("Move from scanner to tracker coordinates");
      parser.add_key("conventions", &conventions);
      parser.add_key("transformation",&transformation_as_string);
      parser.add_stop_key("END"); 
      if (parser.parse(transformation_from_scanner_coordinates_filename.c_str()) == false)
	{
	  warning("Error reading transformation_from_scanner_coordinates_filename:\n'%s'",
		  transformation_from_scanner_coordinates_filename.c_str());
	  return true;
	}
      if (conventions != "q0qzqyqx and left-handed")
	{
	  warning("Error reading transformation_from_scanner_coordinates_filename:\n'%s'"
		  "\nvalue for 'conventions' keyword has to be 'q0qzqyqx and left-handed'",
		  "\nbut is '%s'",
		  transformation_from_scanner_coordinates_filename.c_str(),
		  conventions.c_str());
	  return true;
	}
      std::stringstream transformation_as_stream(transformation_as_string);
      transformation_as_stream >> move_from_scanner_coords;
      if (!transformation_as_stream.good())
	{
	  warning("Error reading transformation_from_scanner_coordinates_filename:\n'%s'"
		  "\nvalue for 'transformation' keyword is invalid."
		  "\nIt should be something like '{{q0,qz,qy,qx},{tz,ty,tx}}'",
		  transformation_from_scanner_coordinates_filename.c_str());
	  return true;
	}
      if (std::fabs(norm(move_from_scanner_coords.get_quaternion())-1)>.01)
      {
#ifdef BOOST_NO_STRINGSTREAM
	// dangerous for out-of-range, but 'old-style' ostrstream seems to need this
	char str[100000];
	ostrstream s(str, 100000);
#else
	std::ostringstream s;
#endif
	s << move_from_scanner_coords;
	warning("Error reading transformation_from_scanner_coordinates_filename:\n'%s'"
		"\nvalue for 'transformation' keyword is invalid:"
		"\nquaternion should be normalised to 1."
		"\ntransformation read:\n%s",
		transformation_from_scanner_coordinates_filename.c_str(),
		s.str().c_str());
	return true;
	}
    }
#endif
    cerr << "'Move_from_scanner' quaternion  " << move_from_scanner_coords.get_quaternion()<<endl;
    cerr << "'Move_from_Scanner' translation  " << move_from_scanner_coords.get_translation()<<endl;
    move_to_scanner_coords = move_from_scanner_coords.inverse();
  }


  if (max_time_drift_deviation<0 || max_time_drift_deviation>.9)
    {
      warning("Polaris: Invalid max_time_drift_deviation %g",  max_time_drift_deviation);
      return true;
    }
  if (max_time_offset_deviation<0 || max_time_offset_deviation>100000)
    {
      warning("Polaris: Invalid max_time_offset_deviation %g",  max_time_offset_deviation);
      return true;
    }

  synchronise();
  
  if (RigidObject3DMotion::post_processing()==true)
    return true;
  return false;
}

Succeeded 
RigidObject3DMotionFromPolaris::
set_mt_file(const string& mt_filename_v)
{
  mt_filename = mt_filename_v;
  mt_file_ptr = new Polaris_MT_File(mt_filename);
  return is_null_ptr(mt_file_ptr) ? Succeeded::no : Succeeded::yes;
}

Succeeded 
RigidObject3DMotionFromPolaris::
set_list_mode_data_file(const string& lm_filename)
{
  this->list_mode_filename = lm_filename;
  return Succeeded::yes;
}



END_NAMESPACE_STIR


