//
//$Date$: $Date$
//
/*!
  \file 
  \ingroup utilities
 
  \brief  This programme extracts projection data by segment into 3d image files

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/
#include "ProjData.h"
#include "Sinogram.h"
#include "SegmentByView.h"
#include "SegmentBySinogram.h"
#include "CartesianCoordinate3D.h"
#include "interfile.h"
#include "utilities.h"


#ifndef TOMO_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_TOMO


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
