import numpy
import stir

def toNumpy(image):
    minind=stir.Int3BasicCoordinate()
    maxind=stir.Int3BasicCoordinate()
    image.get_regular_range(minind, maxind);
    sizes=maxind-minind+1;

    npimage=numpy.zeros( (sizes[1], sizes[2], sizes[3]), dtype=numpy.float32);
    #print 'start'
    ind=stir.Int3BasicCoordinate()
    for i1 in range(sizes[1]):
        ind[1]=i1+minind[1];
        #print ind[1]
        for i2 in range(sizes[2]):
            ind[2]=i2+minind[2];
            #print '...', ind[2]
            for i3 in range(sizes[3]):
                ind[3]=i3+minind[3];
                npimage[i1,i2,i3] = image[ind];
    #print 'stop'

    return npimage;

