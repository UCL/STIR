#include "local/stir/listmode/TimeFrameDefinitions.h"
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::cout;
#endif

USING_NAMESPACE_STIR

int
main(int argc, char* argv[])
{
  if(argc !=3)
  {
    cerr << "Usage:" << argv[0] << " Frame def filename, Frame number " << endl;
    return EXIT_FAILURE;
  }
  
  TimeFrameDefinitions time_def(argv[1]);
  const int frame_num = atoi(argv[2]);
  const double start_frame = time_def.get_start_time(frame_num);
  const double end_frame = time_def.get_end_time(frame_num);
  double frame_duration = end_frame-start_frame;

  cout << "Start frame :" << start_frame*1000 << endl;
  cout << "Frame duration :" << frame_duration*1000 << endl;



  return EXIT_SUCCESS;
}

