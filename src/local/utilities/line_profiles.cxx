#if 1


#include "stir/interfile.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/utilities.h"

#include "stir/IndexRange2D.h"
//#include "stir/eval_buildblock/eval_common.h"
#include "stir/Array.h"
#include "stir/recon_array_functions.h"
#include "stir/ArrayFunction.h"


#include <iostream> 
#include <fstream>
#include <numeric>


#define SMALL_VALUE = 0.0000001;


#ifndef STIR_NO_NAMESPACES
using std::iostream;
using std::ofstream;
using std::ifstream;
using std::ios;



using std::cerr;
using std::endl;
using std::cout;
#endif


// Date:24/080/2000



/* This program finds the max value in the plane and takes the profile 
 which is then written to a file. 
 Note: only certain number of planes are taken, e.g only those that actually
 contain the object itself, since only for those the max value and the profile make sense.
 */
/*
    Copyright (C) 2000- $Year:$, IRSL
    See STIR/LICENSE.txt for details
*/


USING_NAMESPACE_STIR


void get_plane(Array<2,float>& plane, 
	       const VoxelsOnCartesianGrid<float>& input_image,
	       int plane_num, int min_row, int max_row,int min_colum, int max_colum); 

void get_plane_row(Array<1,float>& row,
                   const Array<2,float>& input_plane,
		   const int xcoord,const int plane_num,int min_row,int max_row);

void get_plane_colum(Array<1,float>& colum,
		     const Array<2,float>& input_plane,
		     const int xcoord,const int plane_num, int min_colum, int max_colum); 

void put_end_points_to_zero(Array<1,float>& row_out,const Array<1,float>& row_in);

string get_file_index(const string filename);

VoxelsOnCartesianGrid<float> ask_interfile_image(const char *const input_query);

//bool check_if_sensible(Array<1,float> array_input);

 
int main(int argc, char *argv[])
{ 
  VoxelsOnCartesianGrid<float> input_image;

  if (argc!= 4)
    cerr << "Usage: image_list min_plane num min_row max_row min_col max_col"<<endl<<endl;    

  ifstream in (argv[1]);  
  const int maxsize_length = 100;
  char filename[80];

   if (!in)
  {
    cout<< "Cannot open input file.\n";
    return 1;
  }
    
  bool profile = ask("Do you what X(Y) or Y(N) profile ?", true);
  
  int xcoord_of_max_value;
  int ycoord_of_max_value; 

 while(!in.eof())
 {      
    in.getline(filename,maxsize_length);
    if(strlen(filename)==0)
      break;
  int file_counter = atoi(get_file_index(filename).c_str());
    
  input_image= (*read_interfile_image(filename));
  
  ofstream profile_file;
  ofstream profile_file2nd;

  string out_file = filename;
  int str_len = strlen(filename);
  out_file.insert(str_len-3,"out");
  int strnewlenght = strlen(out_file.c_str());
  out_file.erase(strnewlenght-2,2);
  string exten;
  if (profile)
    exten = "X_prof_sinimage_planenum1_central";
  else
    exten = "Y_prof_sinimage_planenum1_central";

  out_file.append(exten);

 
  profile_file.open(out_file.c_str(),ios::out);
  //profile_file2nd.open(out_file2nd.c_str(),ios::out);
  profile_file << " plane number        " <<                " row" << endl;
     
  int zs,ys,xs, ze,ye,xe;
  
  zs=input_image.get_min_z();
  ys=input_image.get_min_y();
  xs=input_image.get_min_x(); 
  
  ze=input_image.get_max_z();  
  ye=input_image.get_max_y(); 
  xe=input_image.get_max_x();
 
  cerr << zs << " " << ze << endl;
  cerr << ys << " " << ye << endl;
  cerr << xs << " " << xe << endl;


 
  cerr << atoi(argv[2]) << endl;
  cerr << atoi(argv[3]) << endl;
  int counter = 0;
  int min_plane_num = atoi(argv[2]);
  int max_plane_num = atoi(argv[3]);    
  int min_colum_num = atoi(argv[4]);
  int max_colum_num = atoi(argv[5]);    
  int min_row_num   = atoi(argv[6]);
  int max_row_num   = atoi(argv[7]);  
  
  Array<1,float> row(min_row_num,max_row_num);
  Array<1,float> row_out(min_row_num,max_row_num);
  Array<1,float> colum(min_colum_num,max_colum_num);
  Array<1,float> colum_out(min_colum_num,max_colum_num);
  Array<2,float> plane_tmp(IndexRange2D(min_colum_num,max_colum_num,min_row_num,max_row_num));  
 

  //cerr << " now entering the loop" << endl;
  for (int plane_num=min_plane_num;plane_num<= max_plane_num;plane_num++)  
  {   
    get_plane(plane_tmp,input_image,plane_num,min_row_num,max_row_num,min_colum_num,max_colum_num);
    float max_value= plane_tmp.find_max();    
    cerr << " MAX value is " << max_value  << endl;
    
    for (int row_index = min_row_num;row_index <= max_row_num;row_index++)
      for (int colum_index =min_colum_num;colum_index<=max_colum_num;colum_index++)
     //for (int row_index = -30;row_index <= 30;row_index++)
      //for (int colum_index =-30;colum_index<=30;colum_index++)
      {
	if (plane_tmp[colum_index][row_index]== max_value)
	{
	  xcoord_of_max_value = row_index;
	  ycoord_of_max_value = colum_index;
	  cerr << " X- coor of max value" << xcoord_of_max_value << endl;
	  cerr << " Y- coor of max value" << ycoord_of_max_value << endl;



	 // cerr plane_tmp[ycoord_of_max_value][xcoord_of_max_value]<<endl;
	}
      }
      // as you have found the max value in the image( e.g. there might be more than one 
      // use any - in this case the last that has been found.
	if (profile)
	{
	  get_plane_row(row,plane_tmp,ycoord_of_max_value,plane_num,min_row_num,max_row_num);
	  put_end_points_to_zero(row_out,row);
	  profile_file << counter <<"      "; 
          for (int row_index = row_out.get_min_index();row_index<=row_out.get_max_index();row_index++)
	  { 
	    
	   
	    profile_file << row_out[row_index]<< " " ;  
	    //cerr << " print row after puttting ends to zero" << endl;	    
	    cerr << row[row_index] << " " ;
	   // cerr << endl;
	  }
	  cerr << endl<< endl;
	  cerr << endl;
	  profile_file<< endl << endl;	  
	}
	else
	{
	  get_plane_colum(colum,plane_tmp,xcoord_of_max_value,plane_num,min_colum_num,max_colum_num);
	  put_end_points_to_zero(colum_out,colum);
	  profile_file << counter <<"      "; 
          for (int colum_index = colum_out.get_min_index();colum_index<=colum_out.get_max_index();colum_index++)
	  {     
	   // profile_file <<  row[row_index]<< " " ;  
	    profile_file << colum_out[colum_index]<< " " ;  
	    cerr << colum[colum_index] << " " ;
	    
	  }
	  cerr << endl<< endl;
	  cerr << endl;
	  profile_file<< endl << endl;	  

	}
  }
    
  }
  cerr << xcoord_of_max_value << endl;
  cerr << ycoord_of_max_value << endl;

  
  return EXIT_SUCCESS;
}


void put_end_points_to_zero( Array<1,float>& row_out,const Array<1,float>& row_in)
{  
  int min_index = row_in.get_min_index();
  int max_index = row_in.get_max_index();

  if (row_in[min_index] >= row_in[(min_index+ max_index)/2])
      warning("WARNING: line profile: "
              "line profile too long - take less pixels into accout.\n");
   for ( int i=min_index;i<=max_index;i++)
   { row_out[i] = row_in[i]; }

  float max_value = row_in.find_max();

  for ( int i=min_index;i<=max_index;i++)
  {
    row_out[i] = row_in[i]-row_in[min_index];
  }
 /* if (row_in[min_index] <0.00001 && row_in[max_index] <0.00001)
  {
    for ( int i=min_index;i<=max_index;i++)
    {
      row_out[i] = row_in[i];}
  }
  else  
  for ( int i=min_index;i<=max_index;i++)
  {    
    if ((max_value/row_in[i])>1.5)
      row_out[i] = 0;
    else
      row_out[i] = row_in[i]-row_in[min_index];
  }*/
  
}

    




void get_plane(Array<2,float>& plane, 
	       const VoxelsOnCartesianGrid<float>& input_image,
	       int plane_num, int min_row, int max_row,int min_colum, int max_colum) 
{
    int zs,ze;
    zs=input_image.get_min_z();
    ze=input_image.get_max_z();     

      for (int y=min_colum; y<= max_colum; y++)
        for (int x=min_row; x<= max_row; x++)
           plane[y][x] = input_image[zs+plane_num][y][x];

}

void get_plane_row(Array<1,float>& row,
		   const Array<2,float>& input_plane,		   
		   const int ycoord,const int plane_num, int min_row,int max_row) 
{
   

    //int  xs=input_plane[1].get_min_index();
    //int  xe=input_plane[1].get_max_index(); 
    int  ys=input_plane.get_min_index();
    int  ye=input_plane.get_max_index(); 

    for (int x=min_row; x<= max_row; x++)

         row [x]= input_plane[ycoord][x];
    
}

void get_plane_colum(Array<1,float>& colum,
		   const Array<2,float>& input_plane,
		   const int xcoord,const int plane_num, int min_colum,int max_colum) 
{   

    int  xs=input_plane[1].get_min_index();
    int  xe=input_plane[1].get_max_index(); 
   // int  ys=input_plane.get_min_index();
   // int  ye=input_plane.get_max_index(); 

    for (int y=min_colum; y<= max_colum; y++)
         colum [y]= input_plane[y][xcoord];
    
}

/*
bool check_if_sensible(Array<1,float> array_input)
{
  float max_value = array_input.find_max();
  if ( array_input[0] ==0 && array_input[array_input_get_max_index]==0]*/




VoxelsOnCartesianGrid<float> ask_interfile_image(const char *const input_query)
{
    char filename[max_filename_length];

    system("ls *hv");
    ask_filename_with_extension(filename, input_query, ".hv");

    shared_ptr<DiscretisedDensity<3,float> > image_ptr =
      DiscretisedDensity<3,float>::read_from_file(filename);
    return 
      * dynamic_cast<VoxelsOnCartesianGrid<float>*>(image_ptr.get());

}


string get_file_index(const string filename)
{
  int last_dot_position = filename.rfind(".");
  int last_undsc_position = filename.rfind("_");

  string file_index  = filename.substr(++last_undsc_position, last_dot_position-last_undsc_position);

  return file_index;
}


#endif


