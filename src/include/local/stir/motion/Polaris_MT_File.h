//
// $Id$
//
/*!
  \file
  \ingroup motion

  \brief Declaration of class Polaris_MT_File

  \author Sanida Mustafovic
  \author Kris Thielemans
  $Date$
  $Revision$ 
*/

/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_Polaris_MT_File__
#define __stir_Polaris_MT_File__

#include "local/stir/Quaternion.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Succeeded.h"

#include <vector>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::vector;
#endif

/* The acquired motion tracking data is formatted
				Sample Time (sec since midnight)
				Random #
				Tool Number
				Q0, Qx, Qy, Qz
				Tx, Ty, Tz
				RMS Error 
				*/

START_NAMESPACE_STIR


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
   /*! This skips the 'missing data' records*/
   Record operator[](unsigned int n) const;

   //! iterators that go through complete records
   const_iterator begin() const { return vector_of_records.begin();}
   const_iterator end() const { return vector_of_records.end();}
   unsigned long num_samples() const { return vector_of_records.size(); }

   //! iterators that go through all tags recorded by the Polaris
   const_iterator begin_all_tags() const { return vector_of_tags.begin();}
   const_iterator end_all_tags() const { return vector_of_tags.end();}
   unsigned long num_tags() const { return vector_of_tags.size(); }

  
private:

  vector<Record> vector_of_records;
  // this contains all tags and times (even those with 'missing data')
  vector<Record> vector_of_tags;
};

END_NAMESPACE_STIR

#endif
