# Copyright 2022 University of Edinburgh
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

H5_SUFFIXES=["h5", "hdf5"]
ADIOS2_SUFFIXES=["bp4", "bp5", "bp"]
class Loader():

    def __init__(self, filename):
        def is_h5(suff):
            return suff in H5_SUFFIXES
        def is_adios2(suff):
            return (suff in ADIOS2_SUFFIXES) or is_h5(suff)

        suff = filename.split(".")[-1]
        if is_adios2(suff):
            self.reader = ADIOS2Reader(filename)
        elif is_h5(suff):
            self.reader = HDF5Reader(filename)
        else:
            self.reader = BinaryReader(filename)

class Reader():

    def __init__(self, filename):
        self.filename = filename
        
class ADIOS2Reader(Reader):

    def __init__(self, filename):
        Reader.__init__(filename)

class HDF5Reader(Reader):

    def __init__(self, filename):
        Reader.__init__(filename)

class BinaryReader(Reader):

    def __init__(self, filename):
        Reader.__init__(filename)
