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

#include <fstream>
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::fstream;
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
   float sample_time;
   unsigned int rand_num;
   char total_num;
   Quaternion<float> quat;
   CartesianCoordinate3D<float> trans;
   float rms ;
   };

   ~Polaris_MT_File () {};
   typedef std::vector<Record>::const_iterator const_iterator;
   Polaris_MT_File(const string& filename);

   void read_mt_file (const string& filename);
   
   
   Record operator[](unsigned int) const;
   const_iterator begin() const { return vector_of_records.begin();}
   const_iterator end() const { return vector_of_records.end();}
   unsigned long num_samples() const { return vector_of_records.size(); }

   Succeeded reset();
  
private:
  ifstream mt_stream;
  std::vector<Record> vector_of_records;

   Succeeded get_next(Record&);
   Succeeded is_end_file();
};

END_NAMESPACE_STIR

#endif
