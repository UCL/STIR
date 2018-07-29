/*!

  \file
  \ingroup projdata

  \brief Implementations for class stir::ProjDataGEAdvance

  \author Palak Wadhwa
  \author Kris Thielemans

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2009-06-22, Hammersmith Imanet Ltd
    Copyright (C) 2011, Kris Thielemans
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



#include "stir/ProjDataFromGEHDF5.h"
#include "stir/Succeeded.h"
#include "stir/Viewgram.h"
#include "stir/Sinogram.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include "stir/IO/read_data.h"
#include "stir/IO/GEHDF5Data.h"
#include "stir/IndexRange.h"
#include "stir/IndexRange3D.h"
#include "H5Cpp.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::vector;
using std::ios;
using std::accumulate;
using std::find;
using std::iostream;
using std::streamoff;
#endif

START_NAMESPACE_STIR

ProjDataFromGEHDF5::ProjDataFromGEHDF5(const std::string& sinogram_filename)
  :
  sino_stream(s),
  offset(0),
  on_disk_data_type(NumericType::SHORT),
  on_disk_byte_order(ByteOrder::big_endian)
{  
    _sinogram_filename=sinogram_filename;

}
  
Viewgram<float>
ProjDataFromGEHDF5::
get_viewgram(const int view_num, const int segment_num,
             const bool make_num_tangential_poss_odd) const
  {

Viewgram<float> viewgram = get_empty_viewgram(view_num,segment_num,make_num_tangential_poss_odd);
      /* PW Modifies this bit to get th time slices from GE HDF5 instead of Sgl.
    Calculate number of time slices from the length of the data (file size minus header).
      _num_time_slices =
        static_cast<int>((end_stream_position - static_cast<streampos>(512)) /
                         SIZE_OF_SINGLES_RECORD);
    */
       // Allocate the main array.
    

     // while ( slice < _num_time_slices) {

    this->open(&_sinogram_filename);
    //PW Open the dataset from that file here.
     char datasetname[300];
     sprintf(datasetname,"/SegmentData/Segment2/3D_TOF_Sinogram/view%d", viewgram.get_view_num() );
   H5::DataSet dataset=this->file.openDataSet(datasetname);

   const int    NX_SUB = 1981;    // hyperslab dimensions
    const int    NY_SUB = 27;
    const int    NZ_SUB = 357;
    const int    NX = 1981;        // output buffer dimensions
    const int    NY = 27;
    const int    NZ = 357;
    const int    RANK_OUT = 3;

    //PW Now find out the type of this dataset.
          H5T_class_t type_class = dataset.getTypeClass();

    //PW Get datatype class and print it out.

          if( type_class == H5T_INTEGER )
          {
    //PW Get the integer type

         H5::IntType intype = dataset.getIntType();

        H5std_string order_string;
             H5T_order_t order = intype.getOrder( order_string );

              //PW Get size of the data element stored in file and print it.

             size_t size = intype.getSize();
          }

    //PW Get dataspace of the dataset.
         H5::DataSpace dataspace = dataset.getSpace();

    //PW Get the number of dimensions in the dataspace.
          int rank = dataspace.getSimpleExtentNdims();

    //PW Get the dimension size of each dimension in the dataspace and display them.

          hsize_t dims_out[3];
          int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);

    //PW Define hyperslab in the dataset; implicitly giving strike and block NULL.

          hsize_t offset[3];   // TODO hyperslab offset in the file
          hsize_t count[3];    // TODO size of the hyperslab in the file
          offset[0] = 0;
          offset[1] = 0;
          offset[2] = 0;
          count[0]  = NX_SUB;
          count[1]  = NY_SUB;
          count[2]  = NZ_SUB;
          dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );
    //PW Define the memory dataspace.
          hsize_t dimsm[3];              /* TODO memory space dimensions */
          dimsm[0] = viewgram.get_num_axial_poss();
          dimsm[1] = 27;
          dimsm[2] = viewgram.get_num_tangential_poss();
          H5::DataSpace memspace( RANK_OUT, dimsm );

    //PW Define memory hyperslab.

          hsize_t      offset_out[3];   // hyperslab offset in memory
          hsize_t      count_out[3];    // size of the hyperslab in memory
          offset_out[0] = segment_offset;
          offset_out[1] = 0;
          offset_out[2] = 0;
          count_out[0]  = NX_SUB;
          count_out[1]  = NY_SUB;
          count_out[2]  = NZ_SUB;
          memspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out );
    //PW Read data from hyperslab in the file into the hyperslab in memory.

          Array<3,float> tof_data(IndexRange3D(dimsm[0],dimsm[1],dimsm[2]));
          Array<1,int> buffer(dimsm[0]*dimsm[1]*dimsm[2]);
          dataset.read( buffer.get_data_ptr(), H5::PredType::NATIVE_INT, memspace, dataspace );
          buffer.release_data_ptr();

          std::copy(buffer.begin(), buffer.end(), tof_data.begin_all());
          for (int ax_pos= viewgram.get_min_axial_pos_num();ax_pos = viewgram.get_max_axial_pos_num();ax_pos++)
    {
              viewgram[ax_pos] = tof_data[ax_pos - viewgram.get_min_axial_pos_num()].sum();
}

}


Succeeded ProjDataFromGEHDF5::set_viewgram(const Viewgram<float>& v)
{
  // TODO
  // but this is difficult: how to adjust the scale factors when writing only 1 viewgram ?
  warning("ProjDataFromGEHDF5::set_viewgram not implemented yet\n");
  return Succeeded::no;
}

Sinogram<float> ProjDataFromGEHDF5::get_sinogram(const int ax_pos_num, const int segment_num,const bool make_num_tangential_poss_odd) const
{ 
  // TODO
  error("ProjDataGEAdvance::get_sinogram not implemented yet\n"); 
  return get_empty_sinogram(ax_pos_num, segment_num);}

Succeeded ProjDataFromGEHDF5::set_sinogram(const Sinogram<float>& s)
{
  // TODO
  warning("ProjDataFromGEHDF5::set_sinogram not implemented yet\n");
  return Succeeded::no;
}

END_NAMESPACE_STIR

