#include "gen.h"
#include <string.h>
#define SCreen_compiling
#include "screen.h"

void put_text (x,y,nm)
int  x,y;
int  nm;
{
  int  xx,yy;
  char str[11];

  SC_MASK(SC_M_ANNOTATE);
  SC_COLOR(SC_C_ANNOTATE);
  SC_TJUST(2,1);
  xx = x;
  yy = y - 15;
  SC_MOVE(xx,yy);
  sprintf(str,"IMAGE %d",nm);
  SC_TEXT(str);
  SC_TJUST(1,1);
  SC_MASK(SC_M_ALL);
}

void put_textstr (x,y,str)
int  x,y;
char str[];
{
  int  xx,yy;

  SC_MASK(SC_M_ANNOTATE);
  SC_COLOR(SC_C_ANNOTATE);
  SC_TJUST(2,1);
  xx = x;
  yy = y - 15;
  SC_MOVE(xx,yy);
  SC_TEXT(str);
  SC_TJUST(1,1);
  SC_MASK(SC_M_ALL);
}

#ifdef ANSI
void annotation(short x,short y,short rctx,short rcty,short scale,short color)
#else
void annotation(x,y,rctx,rcty,scale,color)
short int x,y,scale,rctx,rcty,color;
#endif
{
 char str[5],delstr[7];

 memset(delstr,0xdb,sizeof(delstr)-1);
 delstr[sizeof(delstr)-1] = '\0';

 if ((rctx >=100) && (rcty>=30))
 {
 SC_MOVE(x+5*scale,y+18);
 SC_COLOR(SC_C_BACKGROUND);
 SC_TEXT(delstr);
 SC_COLOR(color);
 sprintf(str,"x:%d",x);
 SC_TEXT(str);
 SC_MOVE(x+5*scale,y+6);
 SC_COLOR(SC_C_BACKGROUND);
 SC_TEXT(delstr);
 SC_COLOR(color);
 sprintf(str,"y:%d",y);
 SC_TEXT(str);
 }
 if ((rctx >=55) && (rcty>=20))
 {
 SC_MOVE(x+5*scale,y+rcty-12);
 SC_COLOR(SC_C_BACKGROUND);
 SC_TEXT(delstr);
 SC_COLOR(color);
 sprintf(str,"s:x%d",scale);
 SC_TEXT(str);
 }
 SC_MOVE(x,y);
}


#ifdef ANSI
char input(short *plx,short *prx,short *ply,short *pry,
           short *x,short *y,short rx,short ry,short step,short *scale)
#else
char input(plx,prx,ply,pry,x,y,rx,ry,step,scale)
short int *plx,*prx,*ply,*pry,*x,*y,rx,ry,step,*scale;
#endif
{
 short int rectx,recty,sx,sy,max,startx,starty;
 short int old_scale;
 char c,ch,ch1;

startx=sx= *x;
starty=sy= *y;
rectx=rx * *scale;
recty=ry * *scale;
while ((startx+rectx-1 > *prx) || (starty+recty-1 > *pry))
{
 *scale -=1;
 rectx=rx * *scale;recty=ry * *scale;
}
SC_MASK(SC_M_ANNOTATE);
SC_COLOR(SC_C_ANNOTATE);
SC_MOVE(sx,sy);
SC_RECT(startx+rectx-1,starty+recty-1);
annotation(sx,sy,rectx,recty,*scale,SC_C_ANNOTATE);
SC_FLUSH();
while ((ch=c=(char)getch()) != 0x0d)
{
 if ((ch=='s')|| (ch=='S'))
 {
      max = 1;
      while( (startx+rx*(max+1)-1 < *prx) && (starty+ry*(max+1)-1 < *pry))
        max++;
      printf("\n\t *************************************************");
      old_scale= *scale;
      *scale=asknr("Give a new scale factor",1,max,1);
      SC_MOVE(startx,starty);
      SC_COLOR(SC_C_BACKGROUND);
      SC_RECT(startx+rectx-1,starty+recty-1);
      annotation(startx,starty,rectx,recty,old_scale,SC_C_BACKGROUND);
      SC_MASK(SC_M_ALL);
      SC_FLUSH();
      *x=sx;
      *y=sy;
      return c;
 }
 else
 {
  if ((ch =='d') || (ch=='D'))
  {
       annotation(startx,starty,rectx,recty,*scale,SC_C_BACKGROUND);
       SC_FLUSH();
       printf("\n\t ********************************************************");
       printf("\n\t * Use cursor keys to change the size of the rectangle! *");
       printf("\n\t * Type Return to end this session !                    *");
       printf("\n\t ********************************************************");
       rx=rectx/ *scale;ry=recty/ *scale;
       while ((ch1=(char)getch()) !=0x0d)
       {
       SC_COLOR(SC_C_BACKGROUND);
       SC_RECT(sx+(rx* *scale)-1,sy+(ry* *scale)-1);
       if (KB_DIRECTION(ch1))
       {
         switch (ch1)
         {
          case KB_UPARROW : ry+=step;
           if (ry>*pry) ry= *pry; break;
          case KB_DNARROW : ry-=step;
           if (ry<*ply) ry= *ply; break;
          case KB_RTARROW : rx+=step;
           if (rx>*prx) rx= *prx; break;
          case KB_LTARROW : rx-=step;
           if (rx<*plx) rx= *plx; break;
         }
        SC_COLOR(SC_C_ANNOTATE);
        SC_RECT(sx+(rx* *scale)-1,sy+(ry* *scale)-1);
        SC_FLUSH();
       }
       }
       SC_MASK(SC_M_ALL);
       SC_CLEAR_BLOCK(SC_C_BACKGROUND,sx,sx+(rx* *scale)-1,sy,sy+(ry* *scale)-1);
       SC_MASK(SC_M_ANNOTATE);
       SC_COLOR(SC_C_ANNOTATE);
       SC_MOVE(startx,starty);
       SC_RECT(startx+rectx-1,starty+recty-1);
       annotation(startx,starty,rectx,recty,*scale,SC_C_ANNOTATE);
       SC_FLUSH();
       rx=recty/ *scale;ry=recty/ *scale;
 }
 else
   {
    if (KB_DIRECTION(ch))
    {  SC_MOVE(sx,sy);
       SC_COLOR(SC_C_BACKGROUND);
       SC_RECT(sx+rectx-1,sy+recty-1);
       switch (ch)
       {
       case KB_UPARROW : sy+=step;
         if ((sy+recty)>*pry) sy= *pry-recty; break;
       case KB_DNARROW : sy-=step;
         if ((sy)<*ply) sy= *ply; break;
       case KB_RTARROW : sx+=step;
         if ((sx+rectx)>*prx) sx= *prx-rectx; break;
       case KB_LTARROW : sx-=step;
         if ((sx)<*plx) sx= *plx; break;
       }
       SC_MOVE(sx,sy);
       SC_COLOR(SC_C_ANNOTATE);
       SC_RECT(sx+rectx-1,sy+recty-1);
       startx=sx;
       starty=sy;
       annotation(sx,sy,rectx,recty,*scale,SC_C_ANNOTATE);
       SC_FLUSH();
    }
   }
 }
}

annotation(sx,sy,rectx,recty,*scale,SC_C_BACKGROUND);
SC_COLOR(SC_C_BACKGROUND);
SC_RECT(sx+rectx-1,sy+recty-1);
SC_MASK(SC_M_ALL);
SC_FLUSH();
*x=sx;
*y=sy;
return c;
}

void input_message()
{
  printf("\n\n\t***************************************************");
  printf("\n\t*                 KEYPAD:                         *");
  printf("\n\t*                 -------                         *");
  printf("\n\t* Use the cursor keys to move the cursor          *");
  printf("\n\t* S:change scale factor  D:delete array on screen *");
  printf("\n\t* CR: end the cursor movement                     *");
  printf("\n\t***************************************************");
}

int center_sc_images(Pscale,min_x,max_x,min_y,max_y, SIZE_X,SIZE_Y,sc_image,nr_sc)
int     *Pscale, max_x,max_y,min_x,min_y, SIZE_X,SIZE_Y,nr_sc;
screen_image_t sc_image[];
{ int   inc_x,inc_y,start_x,start_y;
  int   nr_x,nr_y,LENGTH_X,LENGTH_Y;
  int   nr,i,j,x,y,scale;

  /* how many times can we enlarge the images ?                         */
  LENGTH_X = max_x-min_x;               /* size in pixels of the region */
  LENGTH_Y = max_y-min_y;

  /* KT&CL 3/12/97 changed the test if 1 image fits in the window */
  if (LENGTH_X < SIZE_X || LENGTH_Y < SIZE_Y+15)
  { message("\nEven one image of this size doesn't fit in the window");
    return(0);
  }

  /* Now we find the maximum scale of the images */
  /* KT&CL 3/12/97 start from 1 now */
  scale = 1;
  do
  {
    /* we compute the number of images of this scale that fit in the
       allowed region. Place for text under the images is reserved      */
    scale++;
    nr_x = LENGTH_X / (scale*(SIZE_X-1) + 1);
    nr_y = LENGTH_Y / (scale*(SIZE_Y-1) + 1+15);
  } while (nr_x*nr_y >= nr_sc);
  scale--;                              /* one too far                  */

  nr_x = LENGTH_X / (scale*(SIZE_X-1) + 1);
  nr_y = LENGTH_Y / (scale*(SIZE_Y-1) + 1+15);
  /* we want to center the images in the region
     eg: we have now computed that a maximum of 4x3 images fits
         but we have only 7 images ...                                  */
  /* KT 11/12/97 added first if to check if all images fit or not, and the else
     case to change nr_sc */
  if (nr_x*nr_y > nr_sc)
    {
      if (nr_y==1) nr_x=nr_sc;
      else
	{
	  nr_y = nr_sc / nr_x;
	  if (nr_x*nr_y < nr_sc) nr_y++;
	}
    }
  else
    {
      nr_sc = nr_x*nr_y;
    }

  if (*Pscale!=0 && scale>*Pscale)
      scale = *Pscale;
  *Pscale = scale;

  LENGTH_X = scale*(SIZE_X-1)+1;
  LENGTH_Y = scale*(SIZE_Y-1)+1;
  start_x = (min_x + max_x - nr_x*LENGTH_X)/2;
  start_y = (min_y + max_y - nr_y*(LENGTH_Y+15))/2;
  if (start_y<15)
    start_y = 15;
  inc_x   = (max_x-start_x) / nr_x;
  inc_y   = (max_y-start_y) / nr_y;
  start_y += inc_y*(nr_y-1);            /* begin on top of screen       */

  for(nr=0, i=0, y=start_y; i<nr_y; i++, y -=inc_y)
    for(j=0, x=start_x; j<nr_x && nr<nr_sc; j++, x +=inc_x, nr++)
    { sc_image[nr].sx = x;
      sc_image[nr].sy = y;
    }

  return(nr_sc);
}

void draw_sc_images(size_x,size_y,sc_image,no)
int size_x,size_y,no;
screen_image_t sc_image[];
{ int i;

  for (i=0; i<no; i++)
  { SC_PutImg(sc_image[i].image,sc_image[i].sx,sc_image[i].sy,size_x,size_y);
    put_textstr(sc_image[i].sx+size_x/2,sc_image[i].sy,sc_image[i].text);
  }
  SC_FLUSH();
}
