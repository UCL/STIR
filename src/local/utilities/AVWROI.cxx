#include <stdio.h>
#include <stdlib.h>
#include "AVW.h"
#include "AVW_ObjectMap.h"
#include "AVW_ImageFile.h"

#define ARRAY_FULL
#include "IndexRange3D.h"
#include "VoxelsOnCartesianGrid.h"
#include "CartesianCoordinate3D.h"
#include "interfile.h"
#include "utilities.h"

#ifndef TOMO_NO_NAMESPACES
using std::copy;
using std::cerr;
#endif

USING_NAMESPACE_TOMO

template <typename elemT>
static 
void 
AVW_Volume_to_VoxelsOnCartesianGrid_help(VoxelsOnCartesianGrid<float>& image,
                                    elemT const* avw_data)
{
  // copy(avw_data, avw_data+avw_volume->VoxelsPerVolume, image->begin_all());
 
  // AVW data seems to be y-flipped
  for (int z=image.get_min_z(); z<=image.get_max_z(); ++z)
  {
    for (int y=image.get_max_y(); y>=image.get_min_y(); --y)
    {
      for (int x=image.get_min_x(); x<=image.get_max_x(); ++x)
        image[z][y][x] = static_cast<float>(*avw_data++);
      //copy(avw_data, avw_data + image.get_x_size(), image[z][y].begin());
      //avw_data += image.get_x_size();
    }
  }
}


VoxelsOnCartesianGrid<float> *
AVW_Volume_to_VoxelsOnCartesianGrid(AVW_Volume const* const avw_volume)
{
  // find sizes et al 

  const int size_x = avw_volume->Width;
  const int size_y = avw_volume->Height;
  const int size_z = avw_volume->Depth;
  IndexRange3D range(0, size_z-1,
                     -(size_y/2), -(size_y/2)+size_y-1,
                     -(size_x/2), -(size_x/2)+size_x-1);

  float voxel_size_x = 
    static_cast<float>(AVW_GetNumericInfo("VoxelWidth", avw_volume->Info));
  if (voxel_size_x==0)
  {
    warning("AVW_Volume_to_VoxelsOnCartesianGrid: VoxelWidth not found or 0, I set it to 1\n");
    voxel_size_x = 1.F;
  }
  
  float voxel_size_y = 
    static_cast<float>(AVW_GetNumericInfo("VoxelHeight", avw_volume->Info));
  if (voxel_size_y==0)
  {
    warning("AVW_Volume_to_VoxelsOnCartesianGrid: VoxelHeight not found or 0, I set it to 1\n");
    voxel_size_y = 1.F;
  }
  
  float voxel_size_z = 
    static_cast<float>(AVW_GetNumericInfo("VoxelDepth", avw_volume->Info));
  if (voxel_size_z==0)
  {
    warning("AVW_Volume_to_VoxelsOnCartesianGrid: VoxelDepth not found or 0, I set it to 1\n");
    voxel_size_z = 1.F;
  }


  const CartesianCoordinate3D<float> 
    voxel_size(voxel_size_z, voxel_size_y, voxel_size_x);

  // construct VoxelsOnCartesianGrid
  VoxelsOnCartesianGrid<float> * volume_ptr =
    new VoxelsOnCartesianGrid<float>(range, 
                                     CartesianCoordinate3D<float>(0,0,0),
                                     voxel_size);

  // fill in data 
  switch(avw_volume->DataType)
  {
  case AVW_SIGNED_CHAR:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr, reinterpret_cast<signed char const *>(avw_volume->Mem));      
      break;
    }
  case AVW_UNSIGNED_CHAR:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr, reinterpret_cast<unsigned char const *>(avw_volume->Mem));
      break;
    }
  case AVW_UNSIGNED_SHORT:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr, reinterpret_cast<unsigned short const *>(avw_volume->Mem));
      break;
    }
  case AVW_SIGNED_SHORT:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr, reinterpret_cast<signed short const *>(avw_volume->Mem));
      break;
    }
  case AVW_UNSIGNED_INT:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr, reinterpret_cast<unsigned int const *>(avw_volume->Mem));
      break;
    }
  case AVW_SIGNED_INT:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr, reinterpret_cast<signed int const *>(avw_volume->Mem));
      break;
    }
  case AVW_FLOAT:
    {
      AVW_Volume_to_VoxelsOnCartesianGrid_help(*volume_ptr, reinterpret_cast<float const *>(avw_volume->Mem));
      break;
    }
  default:
    {
      warning("AVW_Volume_to_VoxelsOnCartesianGrid: unsupported data type %d\n",
        avw_volume->DataType);
      return 0;
    }
  }
         
  return volume_ptr;
}

int
main(int argc, char **argv)
{
  if (argc != 2)
  {
    cerr << "Usage : " << argv[0] << " analyzeobjectmapfile.obj\n";
    return EXIT_FAILURE;
  }
  char *objectfile = argv[1];
  const bool write_interfile_images = ask("write_interfile_images",true);
  {
    // open non-existent file first
    // this is necessary to get AVW_LoadObjectMap to work
    AVW_ImageFile *db= AVW_OpenImageFile("xxx","r");
  }
  
  fprintf(stdout, "Reading ObjectMap %s...\n", objectfile); fflush(stdout);
  AVW_ObjectMap *object_map = AVW_LoadObjectMap(objectfile);
  if(!object_map) { AVW_Error("AVW_LoadObjectMap"); exit(1); }
  
  printf("Number of objects: %d\n", object_map->NumberOfObjects);
  {
    int object_num;
    AVW_Volume *volume = NULL;
    AVW_Image *image=NULL;
    for (object_num=0; object_num<object_map->NumberOfObjects; ++object_num)
    {
      printf("Object %d: %s\n", object_num, object_map->Object[object_num]->Name);
/* these seem to be irrelevant values
      printf("x : %d,%d\n", object_map->Object[object_num]->MinimumXValue,
                            object_map->Object[object_num]->MaximumXValue);
      printf("y : %d,%d\n", object_map->Object[object_num]->MinimumYValue,
                            object_map->Object[object_num]->MaximumYValue);
      printf("z : %d,%d\n", object_map->Object[object_num]->MinimumZValue,
                            object_map->Object[object_num]->MaximumZValue);
*/
      //volume = AVW_GetObject(object_map, object_num, volume);
      //image = AVW_GetOrthogonal(volume, AVW_TRANSVERSE,0,image);
      //AVW_ShowImage(image);
          
#if 0
      {
        volume = AVW_GetObject(object_map, object_num, volume);
        AVW_ImageFile   *outfile=AVW_NULL;
        char string[256],outfilename[256];
        sprintf(outfilename, "f%d.avw", object_num);
        if((outfile = AVW_CreateImageFile(outfilename, "AnalyzeAVW",
          volume->Width, volume->Height,
          volume->Depth, volume->DataType))==NULL)
        {
          sprintf(string,"Error Creating <%s>\n",outfilename);
          AVW_Error(string);
          AVW_CloseImageFile(outfile);
          exit(1);
        }
        /* write the volume to a new file */
        if((AVW_WriteVolume(outfile, 0, volume)) == AVW_FAIL)
        {
          sprintf(string,"Error Writing <%s>\n", outfilename);
          AVW_Error(string);
          AVW_CloseImageFile(outfile);
          exit(1);
        }
        
        /* close the image files */
        AVW_CloseImageFile(outfile);
      }
#endif
      if (write_interfile_images && ask("Write this one?",true))
      {
        volume = AVW_GetObject(object_map, object_num, volume);
        char outfilename[256];
        ask_filename_with_extension(outfilename, "Name for output file",".v");
        //sprintf(outfilename, "f%d", object_num);
        VoxelsOnCartesianGrid<float> * pp_volume =
          AVW_Volume_to_VoxelsOnCartesianGrid(volume);
        warning("Setting voxel size to 966\n");
        pp_volume->set_voxel_size(Coordinate3D<float>(2.425F,2.25F,2.25F));
        write_basic_interfile(outfilename, *pp_volume);
        delete pp_volume;
    }


    }
  }
#if 0
	AVW_Volume *volume;
        AVW_RenderedImage *rendered=NULL;
	AVW_RenderParameters *render_param;
	int angle;
	char imagefile[256], objectfile[256];
	
	fprintf(stdout, "Reading Volume...\n"); fflush(stdout);
	sprintf(imagefile, "%s/images/ExtrasData/mri3Dhead.hdr", getenv("BIR"));

	db = AVW_OpenImageFile(imagefile,"r");
	if(!db) { AVW_Error("AVW_OpenImageFile"); exit(1); }
	volume = AVW_ReadVolume(db, 0, NULL);
	if(!volume) { AVW_Error("AVW_ReadVolume"); exit(1); }
	AVW_CloseImageFile(db);
	
	fprintf(stdout, "Reading ObjectMap...\n"); fflush(stdout);
	sprintf(objectfile, "%s/images/ExtrasData/mri3Dhead.obj", getenv("BIR"));
	object_map = AVW_LoadObjectMap(objectfile);
	if(!object_map) { AVW_Error("AVW_LoadObjectMap"); exit(1); }

	render_param = AVW_InitializeRenderParameters(volume, object_map, NULL);
	if(!render_param) { AVW_Error("AVW_InitializeRenderParameters"); exit(1); }
	
	render_param->Type = AVW_TRANSPARENCY_SHADING;
	render_param->ThresholdMinimum = 35;

	render_param->Matrix = AVW_RotateMatrix(render_param->Matrix, -90., 0., 0., render_param->Matrix);
	
	fprintf(stdout, "Rendering"); 
	for(angle = 0; angle < 360; angle += 30)
		{
		fprintf(stdout, "."); fflush(stdout);
		fflush(stdout);
		rendered = AVW_RenderVolume(render_param, rendered);
		if(!rendered) { AVW_Error("AVW_RenderVolume"); exit(1); }
		
		AVW_ShowImage(rendered->Image);
		render_param->Matrix = AVW_RotateMatrix(render_param->Matrix, 0., 30., 0., render_param->Matrix);
		}
	fprintf(stdout, "\n"); fflush(stdout);
	
#endif
	
        return(0);
}


