//
//
/*
    Copyright (C) 2010- 2010 , Hammersmith Imanet Ltd
    For internal GE use only
*/
#ifndef __stir_AbsTimeIntervalFromDynamicData__H__
#define __stir_AbsTimeIntervalFromDynamicData__H__
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::AbsTimeIntervalFromDynamicData

  \author Kris Thielemans
*/

#include "stir/RegisteredParsingObject.h"
#include "local/stir/AbsTimeInterval.h"
#include "stir/TimeFrameDefinitions.h"
#include <string>

START_NAMESPACE_STIR
class Succeeded;

/*! \ingroup buildblock

  \brief class for specifying a time interval via a dynamic scan

  The dynamic scan can be either a sinogram of an image. It is read via DynamicProjData or
  DynamicDiscretisedDensity.
  
  \par Example .par file
  \verbatim
    ; example keyword in some .par file
     time interval type:= from dynamic data
       Absolute Time Interval from dynamic data :=
         filename:=scandata.S
         start time frame:=1
         end time frame:=1
       end Absolute Time Interval from dynamic data :=
  \endverbatim

*/
class AbsTimeIntervalFromDynamicData
: public RegisteredParsingObject<AbsTimeIntervalFromDynamicData,
				 AbsTimeInterval,
				 AbsTimeInterval>
{

public:
  //! Name which will be used when parsing a AbsTimeInterval object 
  static const char * const registered_name; 

  virtual ~AbsTimeIntervalFromDynamicData() {}
  //! default constructor gives ill-defined values
  AbsTimeIntervalFromDynamicData();
  //! read info from file
  /*! will call error() if something goes wrong */
  AbsTimeIntervalFromDynamicData(const std::string& filename, 
                                 const unsigned int start_time_frame_num,
                                 const unsigned int end_time_frame_num);
  
 private:
  std::string _filename; 
  //TimeFrameDefinitions _time_frame_defs;
  double _scan_start_time_in_secs_since_1970; // start of scan, not the time-interval
  unsigned int _start_time_frame_num;
  unsigned int _end_time_frame_num;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();


  Succeeded set_times();

};

END_NAMESPACE_STIR
#endif
