import numpy
import stir

def toNumpy(image):
    minind=stir.Int3BasicCoordinate()
    maxind=stir.Int3BasicCoordinate()
    image.get_regular_range(minind, maxind);
    sizes=maxind-minind+1;

    npimage=numpy.fromiter(image.flat(), dtype=numpy.float32);
    npimage=npimage.reshape(sizes[1], sizes[2], sizes[3]);

    return npimage;

