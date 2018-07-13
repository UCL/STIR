/*
    Copyright (C) 2005 - 2011-12-31, Hammersmith Imanet Ltd
    Copyright (C) 2013, Kris Thielemans
    Copyright (C) 2013, University College London
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
/*!
  \file
  \ingroup data_buildblock
  \brief Implementation of class stir::DynamicProjData
  \author Kris Thielemans
  \author Charalampos Tsoumpas
*/

#include "stir/DynamicProjData.h"
#include "stir/ExamInfo.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ExamInfo.h"
#include "stir/TimeFrameDefinitions.h"
#include <iostream>
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/round.h"
#include "stir/error.h"
#include "stir/warning.h"
#include "stir/info.h"
#include "stir/utilities.h"
#include "stir/IO/FileSignature.h"
#include "stir/IO/InterfileHeader.h"
#include "stir/IO/interfile.h"
#include "stir/MultipleDataSetHeader.h"
#include <boost/format.hpp>
#include <fstream>
#include <math.h>

using std::string;
using std::istream;

START_NAMESPACE_STIR
// declaration of helper function
static
DynamicProjData*
read_interfile_DPDFS(const string& filename,
		     const std::ios::openmode open_mode = std::ios::in);

const double
DynamicProjData::
get_start_time_in_secs_since_1970() const
{ return this->get_exam_info_sptr()->start_time_in_secs_since_1970; }

void 
DynamicProjData::
set_start_time_in_secs_since_1970(const double start_time)
{ 
  // reset local pointer to a new one (with the same info) to avoid changing
  // other objects that happen to use the same shared_ptr
  this->exam_info_sptr.reset(new ExamInfo(*this->exam_info_sptr));
  this->exam_info_sptr->start_time_in_secs_since_1970=start_time;
}

void
DynamicProjData::
set_time_frame_definitions(const TimeFrameDefinitions& time_frame_definitions) 
{ 
  // reset local pointer to a new one (with the same info) to avoid changing
  // other objects that happen to use the same shared_ptr
  this->exam_info_sptr.reset(new ExamInfo(*this->exam_info_sptr));
  this->exam_info_sptr->time_frame_definitions=time_frame_definitions;
}

const TimeFrameDefinitions& 
DynamicProjData::
get_time_frame_definitions() const
{   return this->exam_info_sptr->time_frame_definitions;    }

shared_ptr<DynamicProjData>
DynamicProjData::
read_from_file(const string& filename) // The written projection data is read in respect to its center as origin!!!
{
  const FileSignature file_signature(filename);
  const char * signature = file_signature.get_signature();

  // Interfile
  if (is_interfile_signature(signature))
  {
#ifndef NDEBUG
    info(boost::format("DynamicProjData::read_from_file trying to read %s as Interfile") % filename);
#endif
    shared_ptr<DynamicProjData> ptr(read_interfile_DPDFS(filename, std::ios::in));
    if (!is_null_ptr(ptr))
      return ptr;
    else
      error(boost::format("DynamicProjData::read_from_file failed to read %s as Interfile") % filename);
  }

#ifdef HAVE_LLN_MATRIX
  if (strncmp(signature, "MATRIX", 6) == 0)
  {
#ifndef NDEBUG
    warning("DynamicProjData::read_from_file trying to read '%s' as ECAT7", filename.c_str());
#endif
    USING_NAMESPACE_ECAT
    USING_NAMESPACE_ECAT7

    if (is_ECAT7_emission_file(filename))
    {
      Main_header mhead;
      if (read_ECAT7_main_header(mhead, filename) == Succeeded::no)
        {
          warning("DynamicProjData::read_from_file cannot read '%s' as ECAT7", filename.c_str());
          return 0;
        }
      shared_ptr<ExamInfo> exam_info_sptr(read_ECAT7_exam_info(filename));
      DynamicProjData * dynamic_proj_data_ptr = new DynamicProjData(exam_info_sptr);

      // we no longer have a _scanner_sptr member, so next lines are commented out
      //dynamic_proj_data_ptr->_scanner_sptr.reset(
      //					 find_scanner_from_ECAT_system_type(mhead.system_type));

      const unsigned int num_frames =
        static_cast<unsigned int>(mhead.num_frames);
      dynamic_proj_data_ptr->_proj_datas.resize(num_frames); 

      for (unsigned int frame_num=1; frame_num <= num_frames; ++ frame_num)
        {
          dynamic_proj_data_ptr->_proj_datas[frame_num-1].reset(
            ECAT7_to_PDFS(filename,
                          frame_num, 
                          /*gate*/1,
                          /*  data_num, bed_num, */ 0,0));
        }
      if (is_null_ptr(dynamic_proj_data_ptr->_proj_datas[0]))
              error("DynamicProjData: No frame available\n");
      return dynamic_proj_data_ptr;
    }
    else
    {
      if (is_ECAT7_file(filename))
        {
          warning("DynamicProjData::read_from_file ECAT7 file '%s' should be projection data.", filename.c_str());
          return 0;
        }
    }
  }
  else 
    {
      warning("DynamicProjData::read_from_file '%s' seems to correspond to ECAT projection data, but not ECAT7. I cannot read this.", filename.c_str());
      return 0;
    }
#endif // end of HAVE_LLN_MATRIX

  if (strncmp(signature, "Multi", 5) == 0) {

#ifndef NDEBUG
        info(boost::format("DynamicProjData::read_from_file trying to read %s as a Multi file.") % filename);
#endif

      unique_ptr<MultipleProjData> multi_proj_data(MultipleProjData::read_from_file(filename).get());
      shared_ptr<DynamicProjData> dynamic_proj_data = MAKE_SHARED<DynamicProjData>(*multi_proj_data);

      dynamic_proj_data->set_time_frame_definitions(dynamic_proj_data->get_exam_info_sptr()->time_frame_definitions);
      return dynamic_proj_data;
  }
  
  // return a zero pointer if we get here
  warning(boost::format("DynamicProjData::read_from_file cannot read '%s'. Unsupported file format?") % filename);
  return shared_ptr<DynamicProjData>();
}

Succeeded 
DynamicProjData::
write_to_ecat7(const string& filename) const 
{
#ifndef HAVE_LLN_MATRIX
  return Succeeded::no;
#else

  Main_header mhead;
  ecat::ecat7::make_ECAT7_main_header(mhead, filename, 
				      *get_exam_info_ptr(),
                                      *get_proj_data(1).get_proj_data_info_ptr() );
    
  MatrixFile* mptr= matrix_create (filename.c_str(), MAT_CREATE, &mhead);
  if (mptr == 0)
    {
      warning("DynamicProjData::write_to_ecat7 cannot write output file %s\n", filename.c_str());
      return Succeeded::no;
    }
  for (  unsigned int frame_num = 1 ; frame_num<=this->get_num_frames() ;  ++frame_num ) 
    {
      if (ecat::ecat7::ProjData_to_ECAT7(mptr,
                                         get_proj_data(frame_num),
                                         frame_num)
          == Succeeded::no)
      {
        matrix_close(mptr);
        return Succeeded::no;
      }
    }
  matrix_close(mptr);
  return Succeeded::yes;
#endif // end of HAVE_LLN_MATRIX
}

// local helper function to read the data from Interfile
// It is largely a copy of read_interfile_PDFS (for a single time frame)
// This needs to be merged of course.
static
DynamicProjData* 
read_interfile_DPDFS(istream& input,
		    const string& directory_for_data,
		     const std::ios::openmode open_mode)
{
  
  InterfilePDFSHeader hdr;  

   if (!hdr.parse(input))
    {
      return 0; // KT 10122001 do not call ask_parameters anymore
    }

  // KT 14/01/2000 added directory capability
  // prepend directory_for_data to the data_file_name from the header

  char full_data_file_name[max_filename_length];
  strcpy(full_data_file_name, hdr.data_file_name.c_str());
  prepend_directory_name(full_data_file_name, directory_for_data.c_str());  
 
  for (unsigned int i=1; i<hdr.image_scaling_factors[0].size(); i++)
    if (hdr.image_scaling_factors[0][0] != hdr.image_scaling_factors[0][i])
      { 
	warning("Interfile warning: all image scaling factors should be equal \n"
		"at the moment. Using the first scale factor only.\n");
	break;
      }
  
   assert(hdr.data_info_ptr !=0);

   shared_ptr<std::iostream> data_in(new std::fstream (full_data_file_name, open_mode | std::ios::binary));
   if (!data_in->good())
     {
       warning("interfile parsing: error opening file %s",full_data_file_name);
       return 0;
     }
   shared_ptr<ProjDataInfo> proj_data_info_sptr(hdr.data_info_ptr->create_shared_clone());
   unsigned long data_offset = hdr.data_offset_each_dataset[0];
   // offset in file between time frames
   unsigned long data_offset_increment = 0UL; // we will find it below
   DynamicProjData * dynamic_proj_data_ptr = new DynamicProjData(hdr.get_exam_info_sptr());
   if (is_null_ptr(dynamic_proj_data_ptr))
     {
       error(boost::format("DynamicProjData: error allocating memory for new object (for file \"%1%\")")
		    % full_data_file_name);
     }
   dynamic_proj_data_ptr->resize(hdr.get_exam_info_ptr()->time_frame_definitions.get_num_time_frames());
   
   // now set all proj_data_sptrs
   for (unsigned int frame_num=1; frame_num <= dynamic_proj_data_ptr->get_num_frames(); ++frame_num)
     {
       info(boost::format("Using data offset %1% for frame %2% in file %3%\n") % data_offset % frame_num % full_data_file_name);
       shared_ptr<ExamInfo> current_exam_info_sptr(new ExamInfo(*hdr.get_exam_info_sptr()));
       std::vector<std::pair<double, double> > frame_times;
       frame_times.push_back(std::make_pair(hdr.get_exam_info_ptr()->time_frame_definitions.get_start_time(frame_num),
					    hdr.get_exam_info_ptr()->time_frame_definitions.get_end_time(frame_num)));
       TimeFrameDefinitions time_frame_defs(frame_times);
       current_exam_info_sptr->set_time_frame_definitions(time_frame_defs);

       shared_ptr<ProjData> 
	 proj_data_sptr(new ProjDataFromStream(current_exam_info_sptr,
					       proj_data_info_sptr,
					       data_in,
					       data_offset,
					       hdr.segment_sequence,
					       hdr.storage_order,
					       hdr.type_of_numbers,
					       hdr.file_byte_order,
					       static_cast<float>(hdr.image_scaling_factors[0][0])));
       if (is_null_ptr(proj_data_sptr))
	 {
	   error(boost::format("DynamicProjData: error reading frame %1% from file '%2%'")
		 % frame_num % full_data_file_name);
	 }
      
       dynamic_proj_data_ptr->set_proj_data_sptr(proj_data_sptr, frame_num);

       // TODO, move offset code to InterfileHeader.cxx
       // find data offset increment
       if (frame_num==1)
	 {
	   int num_sinos = 0;
	   for (int s=proj_data_info_sptr->get_min_segment_num();
		s<=proj_data_info_sptr->get_max_segment_num();
		++s)
	     {
	       num_sinos += proj_data_info_sptr->get_num_axial_poss(s);
	     }
	   data_offset_increment = 
		   static_cast<unsigned long>(num_sinos * 
	                                  proj_data_info_sptr->get_num_tangential_poss() * proj_data_info_sptr->get_num_views() * 
	                                  hdr.type_of_numbers.size_in_bytes());
	 }

       // find offset of next frame
       if (frame_num < dynamic_proj_data_ptr->get_num_frames())
	 {
	   // note that hdr.data_offset uses zero-based indexing, so next line finds the offset for frame frame_num+1
	   if (hdr.data_offset_each_dataset[frame_num]>0)
	     {
	       if (fabs(static_cast<double>(data_offset) - hdr.data_offset_each_dataset[frame_num]) < data_offset_increment)
		 {
		   error(boost::format("data offset for frame %1% is too small. Difference in offsets needs to be at least %2%")
			 % (frame_num+1) % data_offset_increment);
		 }
	       data_offset = hdr.data_offset_each_dataset[frame_num];
	     }
	   else
	     data_offset += data_offset_increment;
	 }
     } // end loop over frames

   return dynamic_proj_data_ptr;
}
static
DynamicProjData*
read_interfile_DPDFS(const string& filename,
		     const std::ios::openmode open_mode)
{
  std::ifstream image_stream(filename.c_str(), open_mode);
  if (!image_stream)
    { 
      error(boost::format("DynamicProjData: couldn't open file '%s'") % filename);
    }
  
  return read_interfile_DPDFS(image_stream, get_directory_name(filename), open_mode);
}

END_NAMESPACE_STIR
