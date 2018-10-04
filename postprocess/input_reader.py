# Copyright 2018 G. Deskos

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from .mesh import Mesh
import json

class InputReader():
    """
    InputReader is a helper class which parses json input files and provides an
    interface to instantiate model objects in Py4Incompact3D. This class handles input
    validation regarding input type, but does not enforce value checking. It is
    designed to function as a singleton object, but that is not enforced or required.

    inputs:
        None

    outputs:
        self: InputReader - an instantiated InputReader object
    """
    def __init__(self):

        self._validObjects=["mesh"]

        self._mesh_properties = {
            "Nx" : int,
            "Ny" : int,
            "Nz" : int,
            "Lx" : float,
            "Ly" : float,
            "Lz" : float,
            "BCx": int,
            "BCy": int,
            "BCz": int
        }



    def _parseJSON(self, filename):
        """
        Opens the input json file and parses the contents into a python dict

        inputs:
            filename: str - path to the json input file

        outputs:
            data: dict - contents of the json input file
        """
        with open(filename) as jsonfile:
            data = json.load(jsonfile)
        return data

    def _validateJSON(self, json_dict, type_map):
        """
        Verifies that the expected fields exist in the json input file and
        validates the type of the input data by casting the fields to
        appropriate values based on the predefined type maps in

        _turbineProperties

        _wakeProperties

        _farmProperties

        inputs:
            json_dict: dict - Input dictionary with all elements of type str

            type_map: dict - Predefined type map for type checking inputs
                             structured as {"property": type}
        outputs:
            validated: dict - Validated and correctly typed input property
                              dictionary
        """

        validated = {}

        # validate the object type
        if "type" not in json_dict:
            raise KeyError("'type' key is required")

        if json_dict["type"] not in self._validObjects:
            raise ValueError("'type' must be one of {}".format(", ".join(self._validObjects)))

        validated["type"] = json_dict["type"]

        # validate the description
        if "description" not in json_dict:
            raise KeyError("'description' key is required")

        validated["description"] = json_dict["description"]

        # validate the properties dictionary
        if "properties" not in json_dict:
            raise KeyError("'properties' key is required")
        # check every attribute in the predefined type dictionary for existence
        # and proper type in the given inputs
        propDict = {}
        properties = json_dict["properties"]
        for element in type_map:
            if element not in properties:
                raise KeyError("'{}' is required for object type '{}'".format(element, validated["type"]))

            value, error = self._cast_to_type(type_map[element], properties[element])
            if error is not None:
                raise error("'{}' must be of type '{}'".format(element, type_map[element]))

            propDict[element] = value

        validated["properties"] = propDict

        return validated

    def _cast_to_type(self, typecast, value):
        """
        Casts the string input to the type in typecast

        inputs:
            typcast: type - the type class to use on value

            value: str - the input string to cast to 'typecast'

        outputs:
            position 0: type or None - the casted value

            position 1: None or Error - the caught error
        """
        try:
            return typecast(value), None
        except ValueError:
            return None, ValueError

    def _build_mesh(self, json_dict):
        """
        Instantiates a Turbine object from a given input file

        inputs:
            json_dict: dict - Input dictionary describing a turbine model

        outputs:
            turbine: Turbine - instantiated Turbine object
        """
        propertyDict = self._validateJSON(json_dict, self._mesh_properties)
        return Mesh(propertyDict)


    def read(self, input_file):
        """
        Parses main input file

        inputs:
            input_file: str - path to the json input file

        outputs:
            farm: instantiated FLORIS model of wind farm
        """
        json_dict = self._parseJSON(input_file)

        mesh = self._build_mesh(json_dict["mesh"])
        return mesh
