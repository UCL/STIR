//
// $Id$: $Date$
//


//MJ 9/11/98 Introduced math mode


#include "pet_common.h" 
#include <iostream> 
#include <fstream>

#include <numeric>
#include "imagedata.h"


//MJ 2/11/98 include Tensor functions
#include "TensorFunction.h"

#include "display.h"
// KT 13/10/98 include interfile 
#include "interfile.h"
// KT 23/10/98 use ask_* version all over the place
#include "utilities.h"
//MJ 17/11/98 include new functions
#include "recon_array_functions.h"

#define ZERO_TOL 0.000001

//enable this at some point
//const int rim_trunc_image=2;

void display_halo(const PETImageOfVolume& main_buffer);

void clean_rim(PETImageOfVolume& main_buffer);
void get_plane(PETImageOfVolume& main_buffer);
void get_plane_row(PETImageOfVolume& main_buffer);

PETImageOfVolume ask_interfile_image(char *input_query);
PETImageOfVolume get_interfile_image(char *filename);

void show_menu();
void show_math_menu();
void math_mode(PETImageOfVolume &main_buffer, int &quit_from_math);


main(int argc, char *argv[]){



//  PETScannerInfo::Scanner_type scanner_type = PETScannerInfo::RPT;
// PETScannerInfo scanner(scanner_type);
// KT 13/10/98 include interfile

#if 0
  int scanner_num = 0;
  PETScannerInfo scanner;
  char scanner_char[200];


  scanner_num = ask_num("Enter scanner number (0: RPT, 1: 953, 2: 966, 3: GE) ? ", 0,3,0);
  switch( scanner_num )
    {
    case 0:
      scanner = (PETScannerInfo::RPT);
      sprintf(scanner_char,"%s","RPT");
      break;
    case 1:
      scanner = (PETScannerInfo::E953);
      sprintf(scanner_char,"%s","953");   
      break;
    case 2:
      scanner = (PETScannerInfo::E966);
      sprintf(scanner_char,"%s","966"); 
      break;
    case 3:
      scanner = (PETScannerInfo::Advance);
      sprintf(scanner_char,"%s","Advance");   
      break;
    default:
      PETerror("Wrong scanner number\n"); cerr<<endl;Abort();
    }

 
  if(argc==3){
    scanner.num_bins /= 4;
    scanner.num_views /= 4;
    scanner.num_rings /= 1;
  

    scanner.bin_size = 2* scanner.FOV_radius / scanner.num_bins;
    scanner.ring_spacing = scanner.FOV_axial / scanner.num_rings;
  }



  Point3D origin(0,0,0);
  Point3D voxel_size(scanner.bin_size,
                     scanner.bin_size,
                     scanner.ring_spacing/2); 
  /*
    Point3D voxel_size(scanner.FOV_radius*2/scanner.num_bins,
    scanner.FOV_radius*2/scanner.num_bins,
    scanner.FOV_axial/(2*scanner.num_rings - 1));
  */
  // Create the image
  // two  manipulations are needed now: 
  // -it needs to have an odd number of elements
  // - it needs to be larger than expected because of some overflow in the projectors

  
  int max_bin = (-scanner.num_bins/2) + scanner.num_bins-1;
  if (scanner.num_bins % 2 == 0)
    max_bin++;
   



  PETImageOfVolume 
    main_buffer(Tensor3D<float>(
				0, 2*scanner.num_rings-2,
				(-scanner.num_bins/2), max_bin,
				(-scanner.num_bins/2), max_bin),
		origin,
		voxel_size);


  // Open file with data
  ifstream input;
  open_read_binary(input, argv[1]);
  main_buffer.read_data(input);   
#endif // now interfile

//MJ 13/09/98 restored command line input capability (as requested by KT)

  PETImageOfVolume 
    main_buffer = (argc>1) ? get_interfile_image(argv[1]):ask_interfile_image("File to load in main buffer? ");



   
  int zs,ys,xs, ze,ye,xe,choice;

  zs=main_buffer.get_min_z();
  ys=main_buffer.get_min_y();
  xs=main_buffer.get_min_x(); 
  
  ze=main_buffer.get_max_z();  
  ye=main_buffer.get_max_y(); 
  xe=main_buffer.get_max_x();

 
 

  cerr<<"resx: "<<main_buffer.get_x_size()<<endl;
  cerr<<"resy: "<<main_buffer.get_y_size()<<endl;
  cerr<<"resz: "<<main_buffer.get_z_size()<<endl;
  cerr << "Min and Max in image " << main_buffer.find_min() 
       << " " << main_buffer.find_max() << endl;


  int plane=1,quit_from_math=0;


    show_menu();

  do{ //start main mode



    // KT 23/10/98 use ask_num
    choice = ask_num("Selection: ",0,13,13);


     switch(choice){
    case 0:
      break;

    case 1:
      {  
	// KT 13/10/98 add text to images
	VectorWithOffset<float> 
	  scale_factors(main_buffer.get_min_index3(), 
			main_buffer.get_max_index3());
	scale_factors.fill(1.F);
	VectorWithOffset<char *> 
	  text(main_buffer.get_min_index3(),
	       main_buffer.get_max_index3());
	for (int i=main_buffer.get_min_index3(); 
	     i<= main_buffer.get_max_index3(); 
	     i++)
	  {
	    char *str = new char [15];
	    sprintf(str, "image %d", i);
	    text[i] = str;
	  }
	display(main_buffer, scale_factors, text, 
		main_buffer.find_max(), 0, 0);
 
	break;
      }
    case 2:
      {
	for (int z=zs; z<=ze; z++)
	  for (int y=ys; y <= ye; y++)
	    for (int x=xs; x<= xe; x++){
	      cerr<<main_buffer[z][y][x]<<" ";
	    }

	break;
      }

    case 3:
      {
	plane=1;
	
	while(plane>0 && plane <=ze-zs+1 ){
	  plane = ask_num("Input plane # (0 to exit)", 0, ze-zs+1, zs);
	  
	  if(plane==0) break;	  	 
	  
	  for (int y=ys; y <= ye; y++)
	    for (int x=xs; x<= xe; x++){
	      cerr<<main_buffer[zs+plane-1][y][x]<<" ";
	      // pause
	    }
	}
	break;

      }
 
    case 4:
      {
	plane=1;
	
	while(plane>0 && plane <=ze-zs+1 ){
	  plane = ask_num("Input plane # (0 to exit)", 0, ze-zs+1, zs);
	  
	  if(plane==0) break;	  	 

	  cerr << "Min and Max in plane " 
	       << main_buffer[zs+plane-1].find_min() 
	       << " " << main_buffer[zs+plane-1].find_max() << endl;
	  
	}
	break;
      }


    case 5:
      {
	//MJ 6/11/98 disabled display halo. Substituted min/max.
	//	display_halo(main_buffer);

  cerr << "Min and Max in image " << main_buffer.find_min() 
       << " " << main_buffer.find_max() << endl;

	break;
      }

    case 6:
      {
	clean_rim(main_buffer);

	break;
      }

    case 7:
      {
	get_plane(main_buffer);
	break;
      }

    case 8:
     {
       get_plane_row(main_buffer);
       break;
     }

    case 9:
      {	
	cerr<<endl<<"The number of counts is: "<<main_buffer.sum()<<endl;      
	break;
      }
     
    case 10: 
      {//math mode

	math_mode(main_buffer,quit_from_math);
	break;

      } //end case math mode

    case 11:
      {//reload buffer from main menu

	main_buffer = ask_interfile_image("File to load in buffer? ");
	break;
      }
    
    case 12:
      {

    char outfile[max_filename_length];
    ask_filename_with_extension(outfile, 
				"Output filename (without extension) ",
				"");
    write_basic_interfile(outfile, main_buffer);
    break;
      }  

   case 13:
    {

      show_menu();
     
  
    }


     } //end switch main mode




     

  }while(choice>0 && choice<=13 && (!quit_from_math));

  return 0;

}


  /***************** Miscellaneous Functions  *******/



void display_halo(const PETImageOfVolume& input_image){

  static int flag_halo=0;
  static PETImageOfVolume halo=input_image;

  if (flag_halo==0){
    int zs,ys,xs, ze,ye,xe;

    zs=input_image.get_min_z();
    ys=input_image.get_min_y();
    xs=input_image.get_min_x(); 
  
    ze=input_image.get_max_z();  
    ye=input_image.get_max_y(); 
    xe=input_image.get_max_x();

  
    float zm=(zs+ze)/2.;
    float ym=(ys+ye)/2.;
    float xm=(xs+xe)/2.;
  
    float d;
 
    for (int z=zs; z<=ze; z++)
      for (int y=ys; y <= ye; y++)
	for (int x=xs; x<= xe; x++){


	  d=sqrt(pow(xm-x,2)+pow(ym-y,2));

	  if(halo[z][y][x]>ZERO_TOL && d>(xe-xm) ) 
	    halo[z][y][x]=1.0;   
	  else
	    halo[z][y][x]=0.0;
	
	}
 

    flag_halo=1;
  }

  display(halo, halo.find_max());     

}


void clean_rim(PETImageOfVolume& input_image){


  const int xe=input_image.get_max_x();
  const int xs=input_image.get_min_x();
  const int xm=(xs+xe)/2;
  

  //MJ 17/11/98 changed default to 2, int to const int
  const int rim_trunc=ask_num("How many voxels to trim? ",0,(int)(xe-xm),2);

  //MJ 17/11/98 use new function
  truncate_rim(input_image,rim_trunc);

}

void get_plane(PETImageOfVolume& input_image){


  int zs,ys,xs, ze,ye,xe;

  zs=input_image.get_min_z();
  ys=input_image.get_min_y();
  xs=input_image.get_min_x(); 
  
  ze=input_image.get_max_z();  
  ye=input_image.get_max_y(); 
  xe=input_image.get_max_x();

  float zm=(zs+ze)/2.;
  float ym=(ys+ye)/2.;
  float xm=(xs+xe)/2.;


  int plane=ask_num("Which plane?",1,ze-zs+1,(int)zm-zs+1);

  ofstream profile;
  ask_filename_and_open(profile,  "Output filename ",
			".prof", ios::out); 

  for (int y=ys; y<= ye; y++)
    for (int x=xs; x<= xe; x++)
      profile<<input_image[zs+plane-1][y][x]<<" ";
      

}

void get_plane_row(PETImageOfVolume& input_image){


  int zs,ys,xs, ze,ye,xe;

  zs=input_image.get_min_z();
  ys=input_image.get_min_y();
  xs=input_image.get_min_x(); 
  
  ze=input_image.get_max_z();  
  ye=input_image.get_max_y(); 
  xe=input_image.get_max_x();

  float zm=(zs+ze)/2.;
  float ym=(ys+ye)/2.;
  float xm=(xs+xe)/2.;
  

  ofstream profile;
  ask_filename_and_open(profile,  "Output filename ",
			".prof", ios::out); 

  int axdir=ask_num("Which axis direction (z=0,y=1,x=2)?",0,2,2);

 

  if (axdir==0){


    int xcoord=ask_num("X COORDINATE: ",1,xe-xs+1,(int)xm-xs+1);
    int ycoord=ask_num("Y COORDINATE: ",1,ye-ys+1,(int)ym-ys+1);

 
    for (int z=zs; z<= ze; z++)
      profile<<input_image[z][ycoord+ys-1][xcoord+xs-1]<<" ";

  }


  else if (axdir==1){


    int zcoord=ask_num("Z COORDINATE: ",1,ze-zs+1,(int)zm-zs+1);
    int ycoord=ask_num("Y COORDINATE: ",1,ye-ys+1,(int)ym-ys+1);

    for (int x=xs; x<= xe; x++)
      profile<<input_image[zcoord+zs-1][ycoord+ys-1][x]<<" ";



  }

  else{ //axdir=2

    int zcoord=ask_num("Z COORDINATE: ",1,ze-zs+1,(int)zm-zs+1);
    int xcoord=ask_num("X COORDINATE: ",1,xe-xs+1,(int)xm-xs+1);
 
    for (int y=ys; y<= ye; y++)
      profile<<input_image[zcoord+zs-1][y][xcoord+xs-1]<<" ";

  }


}

PETImageOfVolume ask_interfile_image(char *input_query){


    char filename[max_filename_length];
    ask_filename_with_extension(filename, 
				input_query,
				"");

    ifstream image_stream(filename);
    if (!image_stream)
    { 
      PETerror("Couldn't open file %s", filename);
      cerr<<endl;
      exit(1);
    }

return read_interfile_image(image_stream);

}

//MJ 12/9/98 added for command line input capability

PETImageOfVolume get_interfile_image(char *filename){


    ifstream image_stream(filename);
    if (!image_stream)
    { 
      PETerror("Couldn't open file %s", filename);
      cerr<<endl;
      exit(1);
    }

return read_interfile_image(image_stream);

}




void show_menu(){

  cerr<<"\n\
    MAIN MODE:\n\
    0. Quit\n\
    1. Display\n\
    2. Data total\n\
    3. Data plane-wise\n\
    4. Min/Max for plane\n\
    5. Min/Max in image\n\
    6. Trim rim \n\
    7. Get plane\n\
    8. Get row\n\
    9. Counts\n\
    10. Math mode\n\
    11. Reload main buffer\n\
    12. Buffer to file\n\
    13. Redisplay menu"<<endl;
}


void show_math_menu(){
  cerr<<"\n\
    MATH MODE:\n\
    0. Quit \n\
    1. Display math buffer\n\
    2. Absolute difference\n\
    3. Add image\n\
    4. Subtract image\n\
    5. Multiply image\n\
    6. Add scalar\n\
    7. Multiply scalar\n\
    8. Divide scalar \n\
    9. Min/Max in image\n\
    10. Main buffer --> Math buffer\n\
    11. Math buffer --> Main Buffer\n\
    12. Reload main buffer\n\
    13. Redisplay menu\n\
    14. Main mode"<<endl;
   
}


void math_mode(PETImageOfVolume &main_buffer, int &quit_from_math){


  PETImageOfVolume math_buffer=main_buffer; //initialize math buffer

	int operation;
	show_math_menu();
	
	do{

	operation=ask_num("Choose Operation: ",0,14,13);

       

    switch(operation){ //math mode

    case 0:
      { 
	quit_from_math=1;
	break;
      }

    case 1:
    {//display math buffer

       display(math_buffer, math_buffer.find_max());

      break;
    }

    case 2:
      { //absolute difference

  PETImageOfVolume 
    aux_image = ask_interfile_image("What image to compare with?");

  math_buffer-=aux_image;
  in_place_abs(math_buffer);
  clean_rim(math_buffer);

  cerr <<endl<< "Min and Max absolute difference " << math_buffer.find_min() 
       << " " << math_buffer.find_max() << endl;

    cerr<<endl<<"Difference (L1 norm): "<<math_buffer.sum()<<endl;      
	


    break;
    }
      

  case 3:
    {// image addition


  PETImageOfVolume 
    aux_image = ask_interfile_image("What image to add?");

  math_buffer+=aux_image;


    break;

      }


 case 4: 
   {//image subtraction


  PETImageOfVolume 
    aux_image = ask_interfile_image("What image to subtract?");

  math_buffer-=aux_image;


    break;

      }

 case 5: 
   {// image multiplication


  PETImageOfVolume 
    aux_image = ask_interfile_image("What image to multiply?");

  math_buffer*=aux_image;

    break;

      }

 //TODO add Kris' image division routine

 case 6: //MJ 6/11/98 new
   {//scalar addition

  float scalar=ask_num("What scalar to add?", -100000,+100000,0);

  math_buffer+=scalar;

    break;

      }

 case 7: //MJ 6/11/98 new
   {//scalar mulltiplication

  float scalar=ask_num("What scalar to multiply?", -100000,+100000,1);

  math_buffer*=scalar;



    break;

      }

 case 8:
   {//scalar division

	float scalar=0.0;

	  do{

	    scalar=ask_num("What scalar to divide?", -100000,+100000,1);

	    if(scalar==0.0) 
	      cerr<<endl<<"Illegal -- division by 0"<<endl;

	  }while(scalar==0.0);



	  math_buffer/=scalar;

	  break;

      }

    case 9:
      {
  cerr << "Min and Max in image " << math_buffer.find_min() 
       << " " << math_buffer.find_max() << endl;

	break;
      }

    case 10:
      {//reinitialize math buffer

	math_buffer=main_buffer;


	break;
      }

    case 11:
      {//dump math buffer to main buffer

	main_buffer=math_buffer;


	break;
      }




    case 12:
      {//reload main buffer in math mode

	main_buffer = ask_interfile_image("File to load in buffer? ");
	break;
      }





    case 13:
      {//redisplay menu

	show_math_menu();
	break;
      }


    case 14:
      {//go back to main mode

	show_menu();

      }


     }// end switch math mode


      }while(operation>0 && operation<14);


}
