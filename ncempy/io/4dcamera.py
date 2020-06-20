"""
A class to read stempy electron counted data sets in different forms like memmapping the file, loading
as a compressed dense dataset, loading using dask delayed.
"""

from pathlib import Path
import tempfile

import h5py
import numpy as np


class file4DC:
    def __init__(self, filename, mode='r', verbose=False):

        self.v = verbose
        self.mode = mode

        # necessary declarations, if something fails
        self.filename = None
        self.fid = None
        self.scan_dimensions = None
        self.frame_dimensions = None
        self.frames = None
        self.scan_positions = None
        self.num_frames = None

        # check filename type
        if isinstance(filename, str):
            pass
        elif isinstance(filename, Path):
            filename = str(filename)
        else:
            raise TypeError('Filename is supposed to be a string or pathlib.Path')
        self.filename = filename

        # try opening the file
        try:
            self.fid = h5py.File(filename, mode)
        except IOError:
            print('File does not exist.')

        self._parse_file()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fid.close()

    def _parse_file(self):
        """ Read the meta data in the file needed to interpret the electron strikes.

        """

        if 'electron_events' in self.fid:
            self.frames = self.fid['electron_events/frames']
            self.scan_positions = self.fid['electron_events/scan_positions']

            self.scan_dimensions = [self.scan_positions.attrs[x] for x in ['Nx', 'Ny']]
            try:
                self.frame_dimensions = [self.frames.attrs[x] for x in ['Nx', 'Ny']]
            except KeyError:
                self.frame_dimensions = (576,576)
            self.num_frames = self.frames.shape[0]

    def getDataset(self):
        """ Get the sparse electron counted data.

        Returns
        -------
            : np.ndarray
                A raveled 1D numpy ndarray of variable length arrays. Each element in the array contains the raveled
                position of an electron strike in the frame. Use scan_dimensions to determine the scan position and
                frame_dimensions to determine the position of each strike.

        Notes
        -----
            If you want to get 'dense' version of the data see other functions like getDenseFrame()
        """
        # Get the electron counted data
        dd = self.fid['electron_events/frames'][:]
        return dd

    def getMemmap(self):
        """
        Get the h5py data set location. No data is loaded.

        Returns
        -------
            : h5py.data_set
                A link to the h5py dataset on disk
        """
        return self.fid['electron_events/frames']

    def getDenseFrame(self, start, end):
        """ Get a dense frame summed from the start frame number to the end frame number.

        To do: Allow user to sum frames in a square ROI using start and end as tuples.

        Parameters
        ----------
        start : int
            If int treat as raveled array and sum. If tuple then treat the array as a 4D array and
            start is the lower left corner of a box to sum in.

        end : int
            If int treat as raveled array and sum. If tuple then treat the array as a 4D array and
            start is the upper right corner of a box to sum in.

        Returns
        -------
            : np.ndarray
                An ndarray of the summed counts. np.dtype is uint32. Shape is frame_dimensions
        """
        dp0 = np.zeros(int(self.frame_dimensions[0]) * int(self.frame_dimensions[1]), dtype='<u4')
        for ii, ev in enumerate(self.frames[start:end]):
            dp0[ev] += 1
        return dp0.reshape(self.frame_dimensions)

    def getDenseDataset(self, compression='lzf', mode='4D'):
        """
        Create a dense version of the data set in memory using the h5py 'core' driver and data set compression. The
        data can reach compressions of about 30x such that a 1024x1024 scan is only 20 GB instead of 650 GB. The
        exact compression depends on the number of electron events per frame though. The file can be created as
        either a 3D or 4D data set.

        Loading is somewhat slow, but accessing the data is fast. See also getDaskDataset() to get dense data
        without loading the entire file into memory.

        To do: Allow user to use a backing store so that the file can be saved when computed.

        Parameters
        ----------
        compression : str, default is 'lzf'
            The compression type passed directly to h5py.create_dataset.
        mode : str, default = '4D'
            Either 4D or 3D to indicate if the file should be ordered as a 4D data set (frame_ky, frame_kx, ry, rx) or
            a 3D dataset (frame, ry, rx).

        Returns
        -------
            file_hdl : h5py.File
                An h5py.file handle to the data in memory. There is no file backing so if the file handle is deleted
                then the data is deleted. Access the data as file_hdl['data']
        """
        if mode == '4D':
            sh = self.scan_dimensions
        elif mode == '3D':
            sh = (self.num_frames,)
        else:
            print('Incorrect mode {}. must be 3D or 4D'.format(mode))
            return

        file_name = tempfile.TemporaryFile() # not sure if this is needed.

        # Create an HDF5 file in memory using the core driver. No backing store is used.
        file_hdl = h5py.File(file_name.name, driver='core', backing_store=False)
        # Create a compressed data set (30x compression is possible)
        file_hdl.create_dataset('data', shape=(*sh, *self.scan_dimensions),
                                dtype='<u1', compression=compression,
                                chunks=(1, self.scan_dimensions[0], self.scan_dimensions[1]))

        for ii in range(self.frames):
            file_hdl['data'][ii, :, :] = self.getDenseFrame(ii, ii+1)

        return file_hdl

    def getDaskDataset(self):
        """Use dask to lazily read the sparse data and return a dense data set. This allows the user to access
        the data on disk as a dense 3D array.

        To do: Allow the user to get a 4D array.

        Returns
        -------
            arr : dask.array.Array
                Returns a dask array. Access the data as if it were a normal 3d ndarray arr[0,0,:,:]
        """
        import dask
        import dask.array as da

        sparse_read = dask.delayed(self.getDenseFrame, pure=True)  # Lazy version of sparse_to_dense

        # List containing each read event. Nothing is read until accessed
        lazy_images = [sparse_read(ee) for ee in self.frames]

        # Construct a delayed small dask array for every frame
        arrays = [da.from_delayed(lazy_image,  dtype=np.uint8, shape=self.frame_dimensions)
                  for lazy_image in lazy_images]

        arr = da.stack(arrays, axis=0)  # Stack all small dask arrays into one
        return arr


if __name__ == '__main__':
    with file4DC('c:/users/linol/data/data_scan218_electrons.4dc') as f0:
        dp = f0.getDenseFrame(0, 10)
        print(dp[0:10, 0:10])