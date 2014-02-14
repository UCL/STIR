//
//
/*!
  \file 
  \ingroup utilities
 
  \brief This programme extracts projection data by segment into 3d
  image files

  \author Kris Thielemans
  \author PARAPET project


  This utility extracts projection data by segment into a sequence of
  3d image files. It is mainly useful to import segments into external
  image display/manipulation programmes which do not understand 3D-PET
  data, but can read Interfile images.
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
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
#include "stir/ProjData.h"
#include "stir/SegmentByView.h"
#include "stir/SegmentBySinogram.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "boost/format.hpp"


#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR


int main(int argc, char *argv[])
{
    if(argc<2) {
        cerr<<"Usage: " << argv[0] << "<file name> (*.hs)\n";
        exit(EXIT_FAILURE);
    }
  
    char const * const filename = argv[1];

    shared_ptr<ProjData>  s3d = ProjData::read_from_file(filename);

    const bool extract_by_view =
      ask_num("Extract as SegmentByView (0) or BySinogram (1)?", 0,1,0)==0;

    for (int segment_num = s3d->get_min_segment_num(); 
         segment_num <=  s3d->get_max_segment_num();
         ++segment_num)
    {
        string output_filename=filename;
        replace_extension(output_filename, "");
        output_filename+="seg";
	output_filename+=boost::str(boost::format("%d") % segment_num);

        if (extract_by_view)
        {
            SegmentByView <float> segment= s3d->get_segment_by_view(segment_num);

            write_basic_interfile(output_filename + "_by_view", segment);
	}
        else {
            SegmentBySinogram<float> segment = s3d->get_segment_by_sinogram(segment_num);  
            write_basic_interfile(output_filename + "_by_sino", segment);
	}
    }

    return EXIT_SUCCESS;
}
