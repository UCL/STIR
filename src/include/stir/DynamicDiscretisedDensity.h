//
//
/*!
  \file
  \ingroup densitydata
  \brief Declaration of class stir::DynamicDiscretisedDensity
  \author Kris Thielemans
  \author Charalampos Tsoumpas
  \author Richard Brown
  
*/
/*
    Copyright (C) 2005 - 2011-01-12, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    Copyright (C) 2018-2020 University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_DynamicDiscretisedDensity_H__
#define __stir_DynamicDiscretisedDensity_H__

#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/Scanner.h"
#include "stir/NestedIterator.h"
#include "stir/is_null_ptr.h"
#include <vector>
#include <string>

START_NAMESPACE_STIR

class Succeeded;

/*! \ingroup buildblock
  \brief Class of multiple image frames, one for each time frame
  Each time frame is a DiscretisedDensity<3,float>

  \todo template in \c elemT and numDimensions?
 */
class DynamicDiscretisedDensity: public ExamData
{
 public:
  //! A typedef that can be used what the base of the hierarchy is
  /*! This typedef is used in write_to_file().
  */
  typedef DynamicDiscretisedDensity hierarchy_base_type;

  typedef DiscretisedDensity<3,float> singleDiscDensT;
 private:
  //! typedef for the private member that stores the densities (one for each time frame)
  typedef std::vector<shared_ptr<singleDiscDensT > > DensitiesT;
 public:
  //! typedef for iterator that runs over all densels in all frames
  typedef NestedIterator<DensitiesT::iterator, PtrBeginEndAllFunction<DensitiesT::iterator> > full_iterator;
  //! typedef for const iterator that runs over all densels in all frames
  typedef NestedIterator<DensitiesT::const_iterator, ConstPtrBeginEndAllFunction<DensitiesT::const_iterator> > const_full_iterator;

  static
    DynamicDiscretisedDensity*
    read_from_file(const std::string& filename);

  DynamicDiscretisedDensity() {}

  DynamicDiscretisedDensity(const DynamicDiscretisedDensity&argument);

  DynamicDiscretisedDensity(const TimeFrameDefinitions& time_frame_definitions, 
                            const double scan_start_time_in_secs_since_1970,
                            const shared_ptr<Scanner>& scanner_sptr)
    {
      _densities.resize(time_frame_definitions.get_num_frames());
      shared_ptr<ExamInfo> _exam_info_sptr(new ExamInfo);
      _exam_info_sptr->set_time_frame_definitions(time_frame_definitions);
      _exam_info_sptr->start_time_in_secs_since_1970=scan_start_time_in_secs_since_1970;
      this->exam_info_sptr=_exam_info_sptr;
      _calibration_factor=-1.F;
      _isotope_halflife=-1.F;
      _scanner_sptr=scanner_sptr;
    }
  //!  Construct an empty DynamicDiscretisedDensity based on a shared_ptr<DiscretisedDensity<3,float> >
  DynamicDiscretisedDensity(const TimeFrameDefinitions& time_frame_definitions,
                            const double scan_start_time_in_secs_since_1970,
                            const shared_ptr<Scanner>& scanner_sptr,
                            const shared_ptr<singleDiscDensT >& density_sptr)
    {  
      _densities.resize(time_frame_definitions.get_num_frames());
      shared_ptr<ExamInfo> _exam_info_sptr;
      if (is_null_ptr(density_sptr->get_exam_info_sptr()))
        _exam_info_sptr.reset(new ExamInfo);
      else
        _exam_info_sptr = density_sptr->get_exam_info_sptr()->create_shared_clone();
      _exam_info_sptr->set_time_frame_definitions(time_frame_definitions);
      _exam_info_sptr->start_time_in_secs_since_1970=scan_start_time_in_secs_since_1970;
      this->exam_info_sptr = _exam_info_sptr;
      _calibration_factor=-1.F;
      _isotope_halflife=-1.F;
      _scanner_sptr=scanner_sptr;
    
      for (unsigned int frame_num=1; frame_num<=time_frame_definitions.get_num_frames(); ++frame_num)
        {
          shared_ptr<singleDiscDensT> density_frame_sptr(density_sptr->get_empty_copy());
          ExamInfo this_exam_info(*exam_info_sptr);
          this_exam_info.set_time_frame_definitions(TimeFrameDefinitions(time_frame_definitions, frame_num));
          density_frame_sptr->set_exam_info(this_exam_info);
          this->_densities[frame_num-1] = density_frame_sptr;
      }
    }  

  DynamicDiscretisedDensity&
    operator=(const DynamicDiscretisedDensity& argument);

  /*! @name functions returning full_iterators 
    These return iterators that run through all elements in all time frames.
  */
  //@{
  inline full_iterator begin_all();
  inline const_full_iterator begin_all() const;
  inline const_full_iterator begin_all_const() const;
  inline full_iterator end_all();
  inline const_full_iterator end_all() const;
  inline const_full_iterator end_all_const() const;
  //@}

  /*! \name get/set the densities
    \warning The frame_num starts from 1
  */
  //@{
  /*!
    \warning This method replaced the set_density_sptr as it was unsafe
  */
  void 
    set_density(const singleDiscDensT& density,
                     const unsigned int frame_num);
  /*
    DynamicDiscretisedDensity(  TimeFrameDefinitions time_frame_defintions,shared_ptr<Scanner>,
    std::vector<shared_ptr<DiscretiseDensity<3,float> > _densities);
  */

  const std::vector<shared_ptr<singleDiscDensT> > &
    get_densities() const ;

  const singleDiscDensT & 
    get_density(const unsigned int frame_num) const ;

  const singleDiscDensT & 
    operator[](const unsigned int frame_num) const 
    { return this->get_density(frame_num); }

  //! Avoid using this as it's unsafe
  singleDiscDensT & 
    get_density(const unsigned int frame_num);

  singleDiscDensT & 
    operator[](const unsigned int frame_num)  
    { return this->get_density(frame_num); }
  //@}

  const float get_isotope_halflife() const;

  const float get_calibration_factor() const;

  //! at method
  const singleDiscDensT & at(const unsigned int frame_num) const
  { return this->get_density(frame_num); }

  //! at method
  singleDiscDensT & at(const unsigned int frame_num)
  { return this->get_density(frame_num); }

  //! Return time of start of scan
  /*! \return the time in seconds since 1 Jan 1970 00:00 UTC, i.e. independent
    of your local time zone.

    Note that the return type is a \c double. This allows for enough accuracy
    for a long time to come. It also means that the start time can have fractional 
    seconds.

    The time frame definitions should be relative to this time.
  */
  const double get_start_time_in_secs_since_1970() const;

  const float get_scanner_default_bin_size() const;

  void set_time_frame_definitions(const TimeFrameDefinitions& time_frame_definitions) 
  {
    shared_ptr<ExamInfo> sptr = this->exam_info_sptr->create_shared_clone();
    sptr->set_time_frame_definitions(time_frame_definitions);
    this->exam_info_sptr = sptr;
  }

  void set_scanner(const Scanner& scanner)
  { this->_scanner_sptr.reset(new Scanner(scanner)); }

  const TimeFrameDefinitions & 
    get_time_frame_definitions() const ;

  unsigned get_num_time_frames() const
  {
    return this->get_time_frame_definitions().get_num_time_frames();
  }

  /*! \brief write data to file
    Currently only in ECAT7 format.
    \warning write_time_frame_definitions() is not yet implemented, so time information is missing.
  */
  Succeeded   
    write_to_ecat7(const std::string& filename) const;

  void calibrate_frames() const ;
  /*!
    \warning This function should be used only if the _decay_corrected is false. Time of a frame is taken as the mean time for each frame which is an accurate approximation only if frame_duration <<< isotope_halflife.
  */
  void decay_correct_frames()  ;
  void set_if_decay_corrected(const bool is_decay_corrected)  ;
  void set_isotope_halflife(const float isotope_halflife);
  void set_calibration_factor(const float calibration_factor) ;
  void set_num_densities(const int num_densities)
  { _densities.resize(num_densities); }
 private:
  // warning: if adding any new members, you have to change the copy constructor as well.
  //TimeFrameDefinitions _time_frame_definitions;
  DensitiesT _densities;
  shared_ptr<Scanner> _scanner_sptr;
  float _calibration_factor;
  float _isotope_halflife;
  bool _is_decay_corrected; 
  //double _start_time_in_secs_since_1970;
};

END_NAMESPACE_STIR

#include "stir/DynamicDiscretisedDensity.inl"

#endif //__stir_DynamicDiscretisedDensity_H__
