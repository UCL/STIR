//
//$Id$
//
/*!
  \file 
  \ingroup utilities
 
  \brief This programme extracts projection data by segment into 3d
  image files

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$

  This utility extracts projection data by segment into a sequence of
  3d image files. It is mainly useful to import segments into external
  image display/manipulation programmes which do not understand 3D-PET
  data, but can read Interfile images.
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/ProjData.h"
#include "stir/SegmentByView.h"
#include "stir/SegmentBySinogram.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"


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
        char output_filename[max_filename_length];
        strcpy(output_filename, filename);
        replace_extension(output_filename, "");
        sprintf(output_filename+strlen(output_filename), "seg%d", segment_num);

        if (extract_by_view)
        {
            SegmentByView <float> segment= s3d->get_segment_by_view(segment_num);

            write_basic_interfile(strcat(output_filename, "_by_view"), segment);
	}
        else {
            SegmentBySinogram<float> segment = s3d->get_segment_by_sinogram(segment_num);  
            write_basic_interfile(strcat(output_filename, "_by_sino"), segment);
	}
    }

    return EXIT_SUCCESS;
}
