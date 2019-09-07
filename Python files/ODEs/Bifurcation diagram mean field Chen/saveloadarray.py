'''
Help script for saving and loading arrays
'''


import numpy as np


def SaveArray(data, filename):
    # Write the array to disk
    with open(filename, 'w+') as outfile:

        outfile.write('# Array shape ( w0, tSteps, dimensions): {0}\n'.format(data.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:

            # The formatting string indicates that out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-10.4f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
        print("Data saved to ", filename)


def LoadArray(filename, shape):
    # shape is a tuple of the dimensions of the output array
    # Read the array from disk
    if not isinstance(filename, str):
        filename = str(filename)
    new_data = np.loadtxt(filename)

    # Note that this returned a 2D array!

    new_data = new_data.reshape(shape)
    return new_data
