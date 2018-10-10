/*
    Copyright (C) 2000- 2002, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir_experimental/Shape/EllipsoidalCylinder.h"
#include "stir_experimental/Shape/CombinedShape3D.h"

  


START_NAMESPACE_STIR

class LineSource_phantom
{
public:

  inline LineSource_phantom();
 
  inline const shared_ptr<Shape3D>& get_A_ptr() const
  { return A_ptr; }
  
  inline shared_ptr<Shape3D> make_union_ptr(const float fraction) const;
   
  inline void translate(const CartesianCoordinate3D<float>& direction);
  inline void scale(const CartesianCoordinate3D<float>& scale3D);

private: 
  shared_ptr<Shape3D> A_ptr;
 
};


LineSource_phantom::LineSource_phantom()
{
  A_ptr = new EllipsoidalCylinder (150,0,0,
    CartesianCoordinate3D<float>(0,0,0),
    0,0,0);
}

void 
LineSource_phantom::translate(const CartesianCoordinate3D<float>& direction)
{
  A_ptr->translate(direction); 
}

void 
LineSource_phantom::scale(const CartesianCoordinate3D<float>& scale3D)
{
  //check this!!!  
  A_ptr->scale_around_origin(scale3D);
}

shared_ptr<Shape3D>
LineSource_phantom::make_union_ptr(const float fraction) const
{
   shared_ptr<Shape3D> full_A = get_A_ptr()->clone();
   
   return full_A ;
     

}

class CylindersWithLineSource_phantom
{
public:

  inline CylindersWithLineSource_phantom();
 
  //inline const shared_ptr<Shape3D>& get_A_ptr() const
  //{ return A_ptr; }
  
  inline const shared_ptr<Shape3D>& get_B_ptr() const
  { return B_ptr; }

 inline const shared_ptr<Shape3D>& get_C_ptr() const
  { return C_ptr; }

 //inline const shared_ptr<Shape3D>& get_B1_ptr() const
  //{ return B1_ptr; }

 //inline const shared_ptr<Shape3D>& get_C1_ptr() const
  //{ return C1_ptr; }


  inline shared_ptr<Shape3D> make_union_ptr(const float fraction) const;
   
  inline void translate(const CartesianCoordinate3D<float>& direction);
  inline void scale(const CartesianCoordinate3D<float>& scale3D);

private: 
  //shared_ptr<Shape3D> A_ptr;
  shared_ptr<Shape3D> B_ptr;
  shared_ptr<Shape3D> C_ptr;
  //shared_ptr<Shape3D> B1_ptr;
 //shared_ptr<Shape3D> C1_ptr;
 
};


CylindersWithLineSource_phantom::CylindersWithLineSource_phantom()
{
  // A_ptr = new EllipsoidalCylinder (150,19,19,
   //CartesianCoordinate3D<float>(0,0,0),
    //0,0,0);

  // for 966
  //B_ptr = new EllipsoidalCylinder (500,60,60,
   // CartesianCoordinate3D<float>(0,0,0),   
  //  0,0,0);


  
  // two planes only
 /* B_ptr = new EllipsoidalCylinder (1,6,6,
    CartesianCoordinate3D<float>(0,-14,0),   
    0,0,0);
  
  C_ptr = new EllipsoidalCylinder (1,6,6,
    CartesianCoordinate3D<float>(0,14,0),
    0,0,0);*/
  // off centre

  // this is to check the bloody arthifacts in new filters

  /*B_ptr = new EllipsoidalCylinder (1,6,6,
    CartesianCoordinate3D<float>(0,-13,4),   
    0,0,0);
  
  C_ptr = new EllipsoidalCylinder (1,6,6,
    CartesianCoordinate3D<float>(0,13,4),
    0,0,0);*/


  // these are the ones noramly used
  B_ptr = new EllipsoidalCylinder (1,8,8,
    CartesianCoordinate3D<float>(0,-14,5),   
    0,0,0);
  
  C_ptr = new EllipsoidalCylinder (1,8,8,
    CartesianCoordinate3D<float>(0,14,5),
    0,0,0);

  // the one that I use for resolution 
 /* B_ptr = new EllipsoidalCylinder (100,8,8,
    CartesianCoordinate3D<float>(0,-14,0),   
    0,0,0);
  
  C_ptr = new EllipsoidalCylinder (100,8,8,
    CartesianCoordinate3D<float>(0,14,0),
    0,0,0);*/

   // the one that I use for resolution 
  /* B1_ptr = new EllipsoidalCylinder (1,6,6,
    CartesianCoordinate3D<float>(0,0,-14),   
    0,0,0);

  C1_ptr = new EllipsoidalCylinder (1,6,6,
    CartesianCoordinate3D<float>(0,0,14),
    0,0,0);*/
  // off centre

  /*B1_ptr = new EllipsoidalCylinder (1,6,6,
    CartesianCoordinate3D<float>(0,-2,-14),   
   0,0,0);

  C1_ptr = new EllipsoidalCylinder (1,6,6,
    CartesianCoordinate3D<float>(0,-2,14),
    0,0,0);*/

}
void 
CylindersWithLineSource_phantom::translate(const CartesianCoordinate3D<float>& direction)
{
//  A_ptr->translate(direction); 
 B_ptr->translate(direction); 
 C_ptr->translate(direction); 
// B1_ptr->translate(direction); 
// C1_ptr->translate(direction); 
}

void 
CylindersWithLineSource_phantom::scale(const CartesianCoordinate3D<float>& scale3D)
{
  //check this!!!  
  //A_ptr->scale_around_origin(scale3D);
  B_ptr->scale_around_origin(scale3D);
  C_ptr->scale_around_origin(scale3D);
  //B1_ptr->scale_around_origin(scale3D);
  //C1_ptr->scale_around_origin(scale3D);

  

}

shared_ptr<Shape3D>
CylindersWithLineSource_phantom::make_union_ptr(const float fraction) const
{
   //shared_ptr<Shape3D> full_A = get_A_ptr()->clone();
   shared_ptr<Shape3D> full_B = get_B_ptr()->clone();
   shared_ptr<Shape3D> full_C = get_C_ptr()->clone();
    
   //shared_ptr<Shape3D> full_B1 = get_B1_ptr()->clone();
   //shared_ptr<Shape3D> full_C1 = get_C1_ptr()->clone();
   
   shared_ptr<Shape3D>  AB_union =
    new CombinedShape3D <logical_and<bool> >( full_C,full_B);

   //shared_ptr<Shape3D>  AB1_union =
     //new CombinedShape3D <logical_and<bool> >( full_C1,full_B1);

   //shared_ptr<Shape3D>  AB1_AB_union =
     //new CombinedShape3D <logical_and<bool> >( AB_union, AB1_union);
   
 //  shared_ptr<Shape3D>  ABC_union =
    // new CombinedShape3D <logical_and<bool> >( AB_union,full_C);
   
   //return AB1_AB_union;
    return AB_union;
   //return full_B;
    

}







// OLD
#if 0

class CylindersWithLineSource_phantom
{
public:


  inline CylindersWithLineSource_phantom();
 
  inline const shared_ptr<Shape3D>& get_A_ptr() const
  { return A_ptr; }
  
  //inline const shared_ptr<Shape3D>& get_B_ptr() const
 // { return B_ptr; }

  /* inline const shared_ptr<Shape3D>& get_C_ptr() const
  { return C_ptr; }

   inline const shared_ptr<Shape3D>& get_D_ptr() const
  { return D_ptr; }*/

 inline shared_ptr<Shape3D> make_union_ptr(const float fraction) const;
   
  inline void translate(const CartesianCoordinate3D<float>& direction);
  inline void scale(const CartesianCoordinate3D<float>& scale3D);

private: 
  shared_ptr<Shape3D> A_ptr;
 // shared_ptr<Shape3D> B_ptr;
 // shared_ptr<Shape3D> C_ptr;
 //shared_ptr<Shape3D> D_ptr;
  


};


CylindersWithLineSource_phantom::CylindersWithLineSource_phantom()
{

  A_ptr = new EllipsoidalCylinder (100,1,1,
    CartesianCoordinate3D<float>(0,0,0),
    0,0,0);


 //  A_ptr = new EllipsoidalCylinder (170,10,10,
   // CartesianCoordinate3D<float>(0,15,-10),
  //  0,0,0);

 // B_ptr = new EllipsoidalCylinder (100,10,10,
   // CartesianCoordinate3D<float>(0,-15,-10),
   // 0,0,0);

  /* A_ptr = new EllipsoidalCylinder (100,90,90,
    CartesianCoordinate3D<float>(0,15,-25),
    0,0,0);

  B_ptr = new EllipsoidalCylinder (150,1,1,
    CartesianCoordinate3D<float>(0,0,0),
    0,0,0);*/


 /* C_ptr = new EllipsoidalCylinder (170,1,1,
    CartesianCoordinate3D<float>(0,0,-25),
    0,0,0);*/

 /* D_ptr = new EllipsoidalCylinder (170,1,1,
    CartesianCoordinate3D<float>(0,0,20),
    0,0,0);*/


}


void 
CylindersWithLineSource_phantom::translate(const CartesianCoordinate3D<float>& direction)
{
  
  A_ptr->translate(direction);
 // B_ptr->translate(direction);
 // C_ptr->translate(direction);
//  D_ptr->translate(direction);

}

void 
CylindersWithLineSource_phantom::scale(const CartesianCoordinate3D<float>& scale3D)
{
  //check this!!!  
  A_ptr->scale_around_origin(scale3D);
 // B_ptr->scale_around_origin(scale3D);
  //C_ptr->scale_around_origin(scale3D);
 // D_ptr->scale_around_origin(scale3D);

}

shared_ptr<Shape3D>
CylindersWithLineSource_phantom::make_union_ptr(const float fraction) const
{
   shared_ptr<Shape3D> full_A = get_A_ptr()->clone();
  // shared_ptr<Shape3D> full_B = get_B_ptr()->clone();
   //shared_ptr<Shape3D> full_C = get_C_ptr()->clone();
  // shared_ptr<Shape3D> full_D = get_D_ptr()->clone();
   
   //shared_ptr<Shape3D>  AB_union =
   
   return full_A ;
     
  // new CombinedShape3D <logical_and<bool> >( full_A,full_B);

  
  
   /* shared_ptr<Shape3D>  ABC_union = 
     new CombinedShape3D <logical_and<bool> >( AB_union,full_C);*/

   // return 
  //  new CombinedShape3D< logical_and<bool> >( AB_union,full_D);

}

#endif
END_NAMESPACE_STIR