//
//
/*!
  \file
  \ingroup utilities

\brief Construct FanProjData from ProjData

\author Nikos Efthimiou

 */

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjData.h"
#include "stir/utilities.h"
#include "stir/NumericType.h"
#include "stir/ByteOrder.h"
#include "stir/ParsingObject.h"
#include "stir/FilePath.h"
#include "stir/ML_norm.h"
#include "stir/IO/write_data.h"

#include <iostream>
#include <fstream>
#include "stir/warning.h"
#include "stir/error.h"

using std::ios;
using std::iostream;
using std::streamoff;
using std::fstream;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;

USING_NAMESPACE_STIR

static void print_usage_and_exit()
{
  std::cerr << "\nUsage construct_fanProjData_fromProjData [output_filename] [input_projdata_filename]\n" <<std::endl;
  exit(EXIT_FAILURE);
}

int
main(int argc, char* argv[])
{

  if(argc != 3)
      print_usage_and_exit();

   shared_ptr<ProjData> in_proj_data_sptr = ProjData::read_from_file(argv[2]);

   const std::string output_file_name = argv[1];
   FanProjData out_fan_data;
  make_fan_data_remove_gaps(out_fan_data, *in_proj_data_sptr);

   shared_ptr<iostream> fan_stream(
       new std::fstream(output_file_name, std::ios::out | std::ios::binary));
   // Array<1, float> value(1);
   float scale = 1.f;

   info("Writing to disk...");
   if(write_data(*fan_stream, out_fan_data, NumericType::Type::FLOAT,scale,
                  ByteOrder::Order::little_endian) == Succeeded::no)
    error("Error writing FanProjData\n");

   info("Finished writing.");
  return EXIT_SUCCESS;
}
