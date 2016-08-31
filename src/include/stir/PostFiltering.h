//
//
/*
    Copyright (C) 2016, UCL

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
#ifndef __stir_PostFiltering_H__
#define  __stir_PostFiltering_H__

/*!
  \file

  \brief This file declares the helper class PostFiltering
  \ingroup utilities
  \author Nikos Efthimiou
  \author Kris Thielemans
*/

START_NAMESPACE_STIR

class PostFiltering
{
public:
  PostFiltering();
  shared_ptr<DataProcessor<DiscretisedDensity<3,float> > > filter_ptr;
public:
  KeyParser parser;

};

PostFiltering::PostFiltering()
{
  filter_ptr.reset();
  parser.add_start_key("PostFilteringParameters");
  parser.add_parsing_key("PostFilter type", &filter_ptr);
  parser.add_stop_key("END PostFiltering Parameters");

}

END_NAMESPACE_STIR

#endif
