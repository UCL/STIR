//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    For internal GE use only.
*/
/*!
  \file
  \ingroup motion

  \brief Declaration of class stir::Polaris_MT_File

  \author Sanida Mustafovic
  \author Kris Thielemans
  $Date$
  $Revision$ 
*/

#ifndef __stir_Polaris_MT_File__
#define __stir_Polaris_MT_File__

#include "local/stir/Quaternion.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Succeeded.h"

#include <vector>
#include <string>
#include <ctime>

# ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::time_t; using ::tm; using ::localtime; }
#endif


START_NAMESPACE_STIR

/*!\ingroup motion
   \brief a class for parsing .mt files output by the Polaris software

  At present, the acquired motion tracking data is formatted as
  \verbatim
    23/5/2003 18:18:32 - 11 1 966_IRSL_II
    record
    record
    ...
  \endverbatim
  where each record is a line containing the following numbers
  \verbatim
      Sample Time (secs since midnight in local time)
      Random #
      Tool Number (as a character)
      Q0, Qx, Qy, Qz
      Tx, Ty, Tz
      RMS Error 
  \endverbatim
  or something like
  \verbatim
  67820.806  2 A : ---- Missing ----
  \endverbatim
  when the Polaris was not able to find the position of the tool.

  \warning All times are in local time, and are hence subject to the
  settings of your TZ environment variable. This means that if the
  data is processed in a different time zone, you will run into 
  trouble.
*/
class Polaris_MT_File
{

public:
  struct Record
  {
   double sample_time;
   unsigned int rand_num;
   char total_num;
   Quaternion<float> quat;
   CartesianCoordinate3D<float> trans;
   float rms ;
   };

  typedef std::vector<Record>::const_iterator const_iterator;

   ~Polaris_MT_File () {};
   typedef std::vector<Record>::const_iterator const_iterator;
   Polaris_MT_File(const std::string& filename);   
   
   //! get the \a n-th complete record
   /*! \warning This skips the 'missing data' records*/
   Record operator[](unsigned int n) const;

   //! iterators that go through complete records
   const_iterator begin() const { return vector_of_records.begin();}
   const_iterator end() const { return vector_of_records.end();}
   unsigned long num_samples() const { return vector_of_records.size(); }

   //! iterators that go through all tags recorded by the Polaris
   const_iterator begin_all_tags() const { return vector_of_tags.begin();}
   const_iterator end_all_tags() const { return vector_of_tags.end();}
   unsigned long num_tags() const { return vector_of_tags.size(); }

   //! start of acquisition as would have been returned by std::time()
   std::time_t get_start_time_in_secs_since_1970();
private:
   std::time_t start_time_in_secs_since_1970;

   std::vector<Record> vector_of_records;
   // this contains all tags and times (even those with 'missing data')
   std::vector<Record> vector_of_tags;
};

END_NAMESPACE_STIR

#endif
