# -*- coding: utf-8 -*-
# Copyright 2024 Matthew Fitzpatrick.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
r"""Contains tests for the module :mod:`czekitout.convert`.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For creating ordered dictionaries.
import collections

# For getting the basename of a function within the function itself.
import inspect

# For randomly selecting items in dictionaries.
import random

# To create path objects.
import pathlib



# For general array handling.
import numpy as np

# For operations related to unit tests.
import pytest



# For converting objects.
import czekitout.convert



##################################
## Define classes and functions ##
##################################



@pytest.fixture
def dict_1_of_objs_for_which_to_test_conversions_to_dicts():
    std_dict = {"b": "hello", "a": 2, "seq": [1, 0, 9]}
    ordered_dict = collections.OrderedDict(std_dict)

    fixture_output = {"std_dict": std_dict,
                      "ordered_dict": collections.OrderedDict(std_dict),
                      "seq": std_dict["seq"]}

    return fixture_output



@pytest.fixture
def dict_1_of_objs_for_which_to_test_conversions_to_strs():
    std_str_1 = "hello"
    byte_str_1 = b"hello"

    fixture_output = {"std_str_1": std_str_1,
                      "byte_str_1": bytes(std_str_1, "utf-8"),
                      "numpy_str_1": np.array(std_str_1),
                      "numpy_str_2": np.array(std_str_1, dtype="S"),
                      "numpy_byte_str_1": np.array(byte_str_1),
                      "numpy_byte_str_2": np.array(byte_str_1, dtype="S"),
                      "path_1": pathlib.Path(std_str_1),
                      "seq": [1, 0, 9],
                      "int_cls": int}

    return fixture_output



@pytest.fixture
def dict_1_of_objs_for_which_to_test_conversions_to_slice_related_objs():
    float_1 = 5.0
    float_2 = 5.5
    int_1 = int(float_1)
    int_2 = -int_1
    int_seq_1 = [1, 0, 9]
    int_seq_2 = tuple(int_seq_1)
    empty_seq_1 = []
    empty_seq_2 = tuple()
    slice_obj_1 = slice(None)
    mixed_type_seq_1 = (1.0, 0, 9)
    mixed_type_seq_2 = (1.0, 0.5, 9)
    mixed_type_seq_3 = (int_seq_1, int_seq_2, int_1, slice_obj_1)
    mixed_type_seq_4 = [int_seq_1, int_1, slice_obj_1]
    mixed_type_seq_5 = (int_seq_1, int_1, slice_obj_1)

    fixture_output = {"float_1": float_1,
                      "float_2": float_2,
                      "int_1": int_1,
                      "int_2": int_2,
                      "int_seq_1": int_seq_1,
                      "int_seq_2": int_seq_2,
                      "empty_seq_1": empty_seq_1,
                      "empty_seq_2": empty_seq_2,
                      "slice_obj_1": slice_obj_1,
                      "mixed_type_seq_1": mixed_type_seq_1,
                      "mixed_type_seq_2": mixed_type_seq_2,
                      "mixed_type_seq_3": mixed_type_seq_3,
                      "mixed_type_seq_4": mixed_type_seq_4,
                      "mixed_type_seq_5": mixed_type_seq_5}

    return fixture_output



@pytest.fixture
def dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_strs():
    str_seq_1 = ["hello", "there"]
    str_seq_2 = tuple(str_seq_1)
    str_seq_3 = ("/foo/bar", "/path/to nowhere")
    mixed_type_seq_1 = ([1, 0, 9], "/foo/bar")

    fixture_output = {"str_seq_1": str_seq_1,
                      "str_seq_2": str_seq_2,
                      "str_seq_3": str_seq_3,
                      "mixed_type_seq_1": mixed_type_seq_1}

    return fixture_output



@pytest.fixture
def dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars():
    fixture_output = \
        {"quadruplet_of_negative_floats_1": [-1.5, -2.5, -3.5, -4.5],
         "quadruplet_of_nonnegative_floats_1": [3.5, 2.5, 1.5, 0.0],
         "quadruplet_of_positive_floats_1": [3.0, 2.0, 1.0, 4.0],
         "quadruplet_of_mixed_types_1": [3.5, 2, 1.5, 4],
         "quadruplet_of_negative_ints_1": [-1, -2, -3, -4],
         "quadruplet_of_nonnegative_ints_1": [3, 2, 1, 0],
         "quadruplet_of_positive_ints_1": [3, 2, 1, 4],
         "pair_of_negative_floats_1": [-1.5, -11.5],
         "pair_of_nonnegative_floats_1": [5.5, 0.0],
         "pair_of_positive_floats_1": [1.0, 5.0],
         "pair_of_mixed_types_1": [1.5, 5],
         "pair_of_negative_ints_1": [-1, -11],
         "pair_of_nonnegative_ints_1": [5, 0],
         "pair_of_positive_ints_1": [1, 5]}

    fixture_output_key_subset_1 = tuple(fixture_output.keys())

    for key_1 in fixture_output_key_subset_1:
        list_obj = fixture_output[key_1]
        tuple_obj = tuple(list_obj)
        key_2 = key_1.replace("1", "2")
        fixture_output[key_2] = tuple_obj

    fixture_output["none_obj_1"] = None
    fixture_output["word_1"] = "foo"

    return fixture_output



@pytest.fixture
def dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalar_pairs():
    fixture_output = \
        {"positive_int_1": 5,
         "pair_of_nonnegative_ints_1": [5, 0],
         "pairs_of_negative_floats_1": [[-5.5, -1.5], (-3.0, -4.5)],
         "pairs_of_nonnegative_floats_1": [[5.5, 1.5], (0.0, 4.5)],
         "pairs_of_positive_floats_1": [[5.0, 1.0], (3.0, 4.0)],
         "pairs_of_mixed_types_1": [[5.5, 1.5], (3, 4)],
         "pairs_of_negative_ints_1": [[-5, -1], (-3, -4)],
         "pairs_of_nonnegative_ints_1": [[5, 1], (0, 4)],
         "pairs_of_positive_ints_1": [[5, 1], (3, 4)]}

    return fixture_output



@pytest.fixture
def dict_1_of_objs_for_which_to_test_conversions_to_scalars():
    fixture_output = {"float_1": 0.0,
                      "float_2": 1.0,
                      "float_3": -5.0,
                      "float_4": 5.0,
                      "float_5": -5.25,
                      "float_6": 5.25,
                      "complex_7": 5.25+1.25j}

    fixture_output_key_subset_1 = tuple(fixture_output.keys())

    for key_1 in fixture_output_key_subset_1:
        obj_1 = fixture_output[key_1]

        std_type_1 = type(obj_1)
        std_type_name_1 = std_type_1.__name__
        std_type_map = {"complex": complex, std_type_name_1: std_type_1}
        if (abs(obj_1 - round(complex(obj_1).real)) == 0.0):
            std_type_map["int"] = int
            if (0 <= round(complex(obj_1).real) <= 1):
                std_type_map["bool"] = bool

        std_type_name_subset_1 = (std_type_name_1,)
        std_type_name_subset_2 = std_type_map.keys()

        for std_type_name_2 in std_type_name_subset_2:
            std_type_2 = std_type_map[std_type_name_2]

            obj_2 = std_type_2(obj_1)
            key_2 = key_1.replace(std_type_name_1, std_type_name_2)
            fixture_output[key_2] = obj_2
            
            obj_3 = np.array(obj_2)
            key_3 = "numpy_" + key_2
            fixture_output[key_3] = obj_3

            obj_4 = str(obj_2)
            key_4 = "str_" + key_2
            fixture_output[key_4] = obj_4

            obj_5 = np.array(obj_4)
            key_5 = "numpy_" + key_4
            fixture_output[key_5] = obj_5

    fixture_output["none_obj_1"] = None
    fixture_output["int_seq_1"] = [0, 1]
    fixture_output["word_1"] = "foo"

    return fixture_output



@pytest.fixture
def dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays():
    seq_of_seq_of_real_numbers = (1.5, (2.5, 3.5))    
    complex_numpy_array_3d = (np.random.rand(10, 3, 4)
                              + 1j*np.random.rand(10, 3, 4))
    complex_numpy_array_3d[2, 0, 0] = 0
    complex_numpy_array_3d[7, 0, 0] = 0
    
    fixture_output = {"seq_of_seq_of_real_numbers_1": \
                      seq_of_seq_of_real_numbers,
                      "tuple_of_real_numbers_1": \
                      tuple(complex_numpy_array_3d.real[:, 0, 0]),
                      "numpy_array_2d_of_strings_1": \
                      np.array((("hello", " ", "world"), ("foo", " ", "bar"))),
                      "nonnegative_numpy_array_1d_1": \
                      complex_numpy_array_3d.real[:, 0, 0],
                      "real_numpy_array_1d_with_negative_values_1": \
                      complex_numpy_array_3d.real[:, 0, 0] - 0.5,
                      "complex_numpy_array_1d_1": \
                      complex_numpy_array_3d[:, 0, 0],
                      "bool_numpy_array_1d_1": \
                      (complex_numpy_array_3d.real[:, 0, 0] > 0.5),
                      "nonnegative_two_column_numpy_matrix_1": \
                      complex_numpy_array_3d.real[:, :2, 0],
                      "two_column_numpy_matrix_with_negative_values_1": \
                      complex_numpy_array_3d.real[:, :2, 0] - 0.5,
                      "nonnegative_three_column_numpy_matrix_1": \
                      complex_numpy_array_3d.real[:, :, 0],
                      "complex_two_column_numpy_matrix_1": \
                      complex_numpy_array_3d[:, :2, 0],
                      "bool_two_column_numpy_matrix_1": \
                      (complex_numpy_array_3d.real[:, :2, 0] > 0.5),
                      "real_numpy_array_3d_1": \
                      complex_numpy_array_3d.real,
                      "bool_numpy_array_3d_1": \
                      (complex_numpy_array_3d.real > 0.5),
                      "complex_numpy_array_3d_1": \
                      complex_numpy_array_3d}

    fixture_output_key_subset_1 = tuple(fixture_output.keys())

    for key_1 in fixture_output_key_subset_1:
        obj_1 = fixture_output[key_1]
        if isinstance(obj_1, np.ndarray):
            obj_2 = obj_1.tolist()
            key_2 = key_1[:-1] + "2"
            fixture_output[key_2] = obj_2

    return fixture_output



def run_generic_test(name_of_test, dict_of_objs_for_which_to_test_conversions):
    basename_of_func_alias = \
        "expected_result_map_of_" + name_of_test
    func_alias = \
        globals()[basename_of_func_alias]
    kwargs = \
        {"dict_of_objs_for_which_to_test_conversions": \
         dict_of_objs_for_which_to_test_conversions}
    expected_result_map = \
        func_alias(**kwargs)

    basename_of_func_alias = \
        "expected_exception_map_of_" + name_of_test
    func_alias = \
        globals()[basename_of_func_alias]
    expected_exception_map = \
        func_alias(**kwargs)

    module_alias = czekitout.convert
    basename_of_func_to_test = "_".join(name_of_test.split("_")[3:])
    func_to_test = module_alias.__dict__[basename_of_func_to_test]

    for obj_name in dict_of_objs_for_which_to_test_conversions:
        obj = dict_of_objs_for_which_to_test_conversions[obj_name]
        expected_result = expected_result_map[obj_name]
        expected_exception = expected_exception_map[obj_name]

        if expected_exception is None:
            assert np.all(func_to_test(obj, obj_name) == expected_result)
        else:
            with pytest.raises(expected_exception) as err_info:
                func_to_test(obj, obj_name)

    return None



def test_1_of_to_dict(
        dict_1_of_objs_for_which_to_test_conversions_to_dicts):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_dict(
        dict_of_objs_for_which_to_test_conversions):
    std_dict = dict_of_objs_for_which_to_test_conversions["std_dict"]

    expected_result_map = {"std_dict": std_dict,
                           "ordered_dict": std_dict,
                           "seq": None}

    return expected_result_map



def expected_exception_map_of_test_1_of_to_dict(
        dict_of_objs_for_which_to_test_conversions):
    key_subset = ("seq",)

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        expected_exception_map[key] = (TypeError if key in key_subset else None)

    return expected_exception_map



def test_1_of_to_str_from_str_like(
        dict_1_of_objs_for_which_to_test_conversions_to_strs):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_str_from_str_like(
        dict_of_objs_for_which_to_test_conversions):
    std_str_1 = dict_of_objs_for_which_to_test_conversions["std_str_1"]

    expected_result_map = {"std_str_1": std_str_1,
                           "byte_str_1": std_str_1,
                           "numpy_str_1": std_str_1,
                           "numpy_str_2": std_str_1,
                           "numpy_byte_str_1": std_str_1,
                           "numpy_byte_str_2": std_str_1,
                           "path_1": std_str_1,
                           "seq": None,
                           "int_cls": None}

    return expected_result_map



def expected_exception_map_of_test_1_of_to_str_from_str_like(
        dict_of_objs_for_which_to_test_conversions):
    key_subset = ("seq", "int_cls")

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        expected_exception_map[key] = (TypeError if key in key_subset else None)

    return expected_exception_map



def test_1_of_to_single_dim_slice(
        dict_1_of_objs_for_which_to_test_conversions_to_slice_related_objs):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_single_dim_slice(
        dict_of_objs_for_which_to_test_conversions):
    int_1 = \
        dict_of_objs_for_which_to_test_conversions["int_1"]
    int_2 = \
        dict_of_objs_for_which_to_test_conversions["int_2"]
    int_seq_1 = \
        dict_of_objs_for_which_to_test_conversions["int_seq_1"]
    empty_seq_1 = \
        dict_of_objs_for_which_to_test_conversions["empty_seq_1"]
    slice_obj_1 = \
        dict_of_objs_for_which_to_test_conversions["slice_obj_1"]
    mixed_type_seq_1 = \
        dict_of_objs_for_which_to_test_conversions["mixed_type_seq_1"]

    int_seq_3 = [int(elem) for elem in mixed_type_seq_1]

    expected_result_map = {"float_1": int_1,
                           "float_2": None,
                           "int_1": int_1,
                           "int_2": int_2,
                           "int_seq_1": int_seq_1,
                           "int_seq_2": int_seq_1,
                           "empty_seq_1": empty_seq_1,
                           "empty_seq_2": empty_seq_1,
                           "slice_obj_1": slice_obj_1,
                           "mixed_type_seq_1": int_seq_3,
                           "mixed_type_seq_2": None,
                           "mixed_type_seq_3": None,
                           "mixed_type_seq_4": None,
                           "mixed_type_seq_5": None}

    return expected_result_map



def expected_exception_map_of_test_1_of_to_single_dim_slice(
        dict_of_objs_for_which_to_test_conversions):
    key_subset = ("float_2",
                  "mixed_type_seq_2",
                  "mixed_type_seq_3",
                  "mixed_type_seq_4",
                  "mixed_type_seq_5")

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        expected_exception_map[key] = (TypeError if key in key_subset else None)

    return expected_exception_map



def test_1_of_to_multi_dim_slice(
        dict_1_of_objs_for_which_to_test_conversions_to_slice_related_objs):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_multi_dim_slice(
        dict_of_objs_for_which_to_test_conversions):
    int_seq_2 = \
        dict_of_objs_for_which_to_test_conversions["int_seq_2"]
    empty_seq_2 = \
        dict_of_objs_for_which_to_test_conversions["empty_seq_2"]
    mixed_type_seq_1 = \
        dict_of_objs_for_which_to_test_conversions["mixed_type_seq_1"]
    mixed_type_seq_5 = \
        dict_of_objs_for_which_to_test_conversions["mixed_type_seq_5"]

    int_seq_4 = tuple(int(elem) for elem in mixed_type_seq_1)

    expected_result_map = {"float_1": None,
                           "float_2": None,
                           "int_1": None,
                           "int_2": None,
                           "int_seq_1": int_seq_2,
                           "int_seq_2": int_seq_2,
                           "empty_seq_1": empty_seq_2,
                           "empty_seq_2": empty_seq_2,
                           "slice_obj_1": None,
                           "mixed_type_seq_1": int_seq_4,
                           "mixed_type_seq_2": None,
                           "mixed_type_seq_3": None,
                           "mixed_type_seq_4": mixed_type_seq_5,
                           "mixed_type_seq_5": mixed_type_seq_5}

    return expected_result_map



def expected_exception_map_of_test_1_of_to_multi_dim_slice(
        dict_of_objs_for_which_to_test_conversions):
    key_subset = ("float_1",
                  "float_2",
                  "int_1",
                  "int_2",
                  "slice_obj_1",
                  "mixed_type_seq_2",
                  "mixed_type_seq_3")

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        expected_exception_map[key] = (TypeError if key in key_subset else None)

    return expected_exception_map



def test_1_of_to_list_of_strs(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_strs):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_list_of_strs(
        dict_of_objs_for_which_to_test_conversions):
    str_seq_1 = dict_of_objs_for_which_to_test_conversions["str_seq_1"]
    str_seq_3 = dict_of_objs_for_which_to_test_conversions["str_seq_3"]

    str_seq_4 = list(str_seq_3)

    expected_result_map = {"str_seq_1": str_seq_1,
                           "str_seq_2": str_seq_1,
                           "str_seq_3": str_seq_4,
                           "mixed_type_seq_1": None}

    return expected_result_map



def expected_exception_map_of_test_1_of_to_list_of_strs(
        dict_of_objs_for_which_to_test_conversions):
    key_subset = ("mixed_type_seq_1",)

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        expected_exception_map[key] = (TypeError if key in key_subset else None)

    return expected_exception_map



def test_1_of_to_tuple_of_strs(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_strs):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_tuple_of_strs(
        dict_of_objs_for_which_to_test_conversions):
    str_seq_2 = dict_of_objs_for_which_to_test_conversions["str_seq_2"]
    str_seq_3 = dict_of_objs_for_which_to_test_conversions["str_seq_3"]

    expected_result_map = {"str_seq_1": str_seq_2,
                           "str_seq_2": str_seq_2,
                           "str_seq_3": str_seq_3,
                           "mixed_type_seq_1": None}

    return expected_result_map



def expected_exception_map_of_test_1_of_to_tuple_of_strs(
        dict_of_objs_for_which_to_test_conversions):
    key_subset = ("mixed_type_seq_1",)

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        expected_exception_map[key] = (TypeError if key in key_subset else None)

    return expected_exception_map



def test_1_of_to_list_of_ints(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_list_of_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if "ints" in key:
            seq_of_ints = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = list(seq_of_ints)
        elif "positive_floats" in key:
            seq_of_floats = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = [int(elem)
                                        for elem
                                        in seq_of_floats]
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_list_of_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("ints" in key) or ("positive_floats" in key):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_tuple_of_ints(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_tuple_of_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if "ints" in key:
            seq_of_ints = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(seq_of_ints)
        elif "positive_floats" in key:
            seq_of_floats = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(int(elem)
                                             for elem
                                             in seq_of_floats)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_tuple_of_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("ints" in key) or ("positive_floats" in key):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_list_of_positive_ints(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_list_of_positive_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if "positive_ints" in key:
            seq_of_ints = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = list(seq_of_ints)
        elif "positive_floats" in key:
            seq_of_floats = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = [int(elem)
                                        for elem
                                        in seq_of_floats]
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_list_of_positive_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("positive_ints" in key) or ("positive_floats" in key):
            expected_exception_map[key] = None
        else:
            if "ints" in key:
                expected_exception_map[key] = ValueError
            else:
                expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_tuple_of_positive_ints(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_tuple_of_positive_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if "positive_ints" in key:
            seq_of_ints = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(seq_of_ints)
        elif "positive_floats" in key:
            seq_of_floats = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(int(elem)
                                             for elem
                                             in seq_of_floats)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_tuple_of_positive_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("positive_ints" in key) or ("positive_floats" in key):
            expected_exception_map[key] = None
        else:
            if "ints" in key:
                expected_exception_map[key] = ValueError
            else:
                expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_list_of_nonnegative_ints(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_list_of_nonnegative_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("positive_ints" in key) or ("nonnegative_ints" in key):
            seq_of_ints = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = list(seq_of_ints)
        elif "positive_floats" in key:
            seq_of_floats = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = [int(elem)
                                        for elem
                                        in seq_of_floats]
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_list_of_nonnegative_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (("positive_ints" in key)
            or ("nonnegative_ints" in key)
            or ("positive_floats" in key)):
            expected_exception_map[key] = None
        else:
            if "_negative_ints" in key:
                expected_exception_map[key] = ValueError
            else:
                expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_tuple_of_nonnegative_ints(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_tuple_of_nonnegative_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("positive_ints" in key) or ("nonnegative_ints" in key):
            seq_of_ints = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(seq_of_ints)
        elif "positive_floats" in key:
            seq_of_floats = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(int(elem)
                                             for elem
                                             in seq_of_floats)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_tuple_of_nonnegative_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (("positive_ints" in key)
            or ("nonnegative_ints" in key)
            or ("positive_floats" in key)):
            expected_exception_map[key] = None
        else:
            if "_negative_ints" in key:
                expected_exception_map[key] = ValueError
            else:
                expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_list_of_floats(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_list_of_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
        if key not in ("none_obj_1", "word_1"):
            expected_result_map[key] = [float(elem)
                                        for elem
                                        in seq_of_scalars]
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_list_of_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if key not in ("none_obj_1", "word_1"):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_tuple_of_floats(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_tuple_of_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
        if key not in ("none_obj_1", "word_1"):
            expected_result_map[key] = tuple(float(elem)
                                             for elem
                                             in seq_of_scalars)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_tuple_of_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if key not in ("none_obj_1", "word_1"):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_list_of_positive_floats(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_list_of_positive_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("positive" in key) or ("mixed_types" in key):
            seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = [float(elem)
                                        for elem
                                        in seq_of_scalars]
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_list_of_positive_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("positive" in key) or ("mixed_types" in key):
            expected_exception_map[key] = None
        elif key in ("none_obj_1", "word_1"):
            expected_exception_map[key] = TypeError
        else:
            expected_exception_map[key] = ValueError

    return expected_exception_map



def test_1_of_to_tuple_of_positive_floats(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_tuple_of_positive_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("positive" in key) or ("mixed_types" in key):
            seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(float(elem)
                                             for elem
                                             in seq_of_scalars)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_tuple_of_positive_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("positive" in key) or ("mixed_types" in key):
            expected_exception_map[key] = None
        elif key in ("none_obj_1", "word_1"):
            expected_exception_map[key] = TypeError
        else:
            expected_exception_map[key] = ValueError

    return expected_exception_map



def test_1_of_to_list_of_nonnegative_floats(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_list_of_nonnegative_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("_negative_" in key) or (key in ("none_obj_1", "word_1")):
            expected_result_map[key] = None
        else:
            seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = [float(elem)
                                        for elem
                                        in seq_of_scalars]

    return expected_result_map



def expected_exception_map_of_test_1_of_to_list_of_nonnegative_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("_negative_" in key):
            expected_exception_map[key] = ValueError
        elif key in ("none_obj_1", "word_1"):
            expected_exception_map[key] = TypeError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_tuple_of_nonnegative_floats(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_tuple_of_nonnegative_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("_negative_" in key) or (key in ("none_obj_1", "word_1")):
            expected_result_map[key] = None
        else:
            seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(float(elem)
                                             for elem
                                             in seq_of_scalars)

    return expected_result_map



def expected_exception_map_of_test_1_of_to_tuple_of_nonnegative_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("_negative_" in key):
            expected_exception_map[key] = ValueError
        elif key in ("none_obj_1", "word_1"):
            expected_exception_map[key] = TypeError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_pair_of_floats(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_pair_of_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("pair" in key):
            seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(float(elem)
                                             for elem
                                             in seq_of_scalars)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_pair_of_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("pair" in key):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_pair_of_positive_floats(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_pair_of_positive_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("pair" in key) and ("negative" not in key):
            seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(float(elem)
                                             for elem
                                             in seq_of_scalars)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_pair_of_positive_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("pair" in key) and ("negative" not in key):
            expected_exception_map[key] = None
        else:
            if ("pair" in key) and ("negative" in key):
                expected_exception_map[key] = ValueError
            else:
                expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_pair_of_nonnegative_floats(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_pair_of_nonnegative_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("pair" in key) and ("_negative_" not in key):
            seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(float(elem)
                                             for elem
                                             in seq_of_scalars)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_pair_of_nonnegative_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("pair" in key) and ("_negative_" not in key):
            expected_exception_map[key] = None
        else:
            if ("pair" in key) and ("_negative_" in key):
                expected_exception_map[key] = ValueError
            else:
                expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_pair_of_ints(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_pair_of_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("pair" in key) and (("ints" in key) or ("positive_floats" in key)):
            seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(int(elem)
                                             for elem
                                             in seq_of_scalars)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_pair_of_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("pair" in key) and (("ints" in key) or ("positive_floats" in key)):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_pair_of_positive_ints(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_pair_of_positive_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("pair" in key) and ("positive" in key):
            seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(int(elem)
                                             for elem
                                             in seq_of_scalars)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_pair_of_positive_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("pair" in key) and ("positive" in key):
            expected_exception_map[key] = None
        else:
            if ("pair" in key) and ("negative_ints" in key):
                expected_exception_map[key] = ValueError
            else:
                expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_pair_of_nonnegative_ints(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_pair_of_nonnegative_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("pair_of_positive" in key) or ("pair_of_nonnegative_ints" in key):
            seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(int(elem)
                                             for elem
                                             in seq_of_scalars)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_pair_of_nonnegative_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("pair_of_positive" in key) or ("pair_of_nonnegative_ints" in key):
            expected_exception_map[key] = None
        else:
            if ("pair" in key) and ("_negative_ints" in key):
                expected_exception_map[key] = ValueError
            else:
                expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_quadruplet_of_nonnegative_ints(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_quadruplet_of_nonnegative_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (("quadruplet_of_positive_ints" in key)
            or ("quadruplet_of_nonnegative_ints" in key)
            or ("quadruplet_of_positive_floats" in key)):
            seq_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(int(elem)
                                             for elem
                                             in seq_of_scalars)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_quadruplet_of_nonnegative_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (("quadruplet_of_positive_ints" in key)
            or ("quadruplet_of_nonnegative_ints" in key)
            or ("quadruplet_of_positive_floats" in key)):
            expected_exception_map[key] = None
        else:
            if ("quadruplet" in key) and ("_negative_ints" in key):
                expected_exception_map[key] = ValueError
            else:
                expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_pairs_of_floats(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalar_pairs):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_pairs_of_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (("_int_" in key) or ("pair_" in key)):
            expected_result_map[key] = None
        else:
            pairs_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(tuple(float(num) for num in pair)
                                             for pair in pairs_of_scalars)

    return expected_result_map



def expected_exception_map_of_test_1_of_to_pairs_of_floats(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (("_int_" in key) or ("pair_" in key)):
            expected_exception_map[key] = TypeError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_pairs_of_ints(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalar_pairs):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_pairs_of_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (("pairs" in key) and (("ints" in key) or ("positive" in key))):
            pairs_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(tuple(float(num) for num in pair)
                                             for pair in pairs_of_scalars)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_pairs_of_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (("pairs" in key) and (("ints" in key) or ("positive" in key))):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_pairs_of_nonnegative_ints(
        dict_1_of_objs_for_which_to_test_conversions_to_seqs_of_scalar_pairs):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_pairs_of_nonnegative_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (("pairs_of_positive_floats" in key)
            or ("pairs_of_nonnegative_ints" in key)
            or ("pairs_of_positive_ints" in key)):
            pairs_of_scalars = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = tuple(tuple(float(num) for num in pair)
                                             for pair in pairs_of_scalars)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_pairs_of_nonnegative_ints(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (("pairs_of_positive_floats" in key)
            or ("pairs_of_nonnegative_ints" in key)
            or ("pairs_of_positive_ints" in key)):
            expected_exception_map[key] = None
        else:
            if ("pairs" in key) and ("_negative_ints" in key):
                expected_exception_map[key] = ValueError
            else:
                expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_float(
        dict_1_of_objs_for_which_to_test_conversions_to_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_float(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (int(key.split("_")[-1]) > 6) or (key in key_subset_1):
            expected_result_map[key] = None
        else:
            obj = dict_of_objs_for_which_to_test_conversions[key]
            obj_as_str = str(np.array(obj).tolist())
            if obj_as_str in ("True", "False"):
                expected_result_map[key] = float(obj_as_str == "True")
            else:
                expected_result_map[key] = complex(obj_as_str).real

    return expected_result_map



def expected_exception_map_of_test_1_of_to_float(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (int(key.split("_")[-1]) > 6) or (key in key_subset_1):
            expected_exception_map[key] = TypeError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_int(
        dict_1_of_objs_for_which_to_test_conversions_to_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_int(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (int(key.split("_")[-1]) > 4) or (key in key_subset_1):
            expected_result_map[key] = None
        else:
            obj = dict_of_objs_for_which_to_test_conversions[key]
            obj_as_str = str(np.array(obj).tolist())
            if obj_as_str in ("True", "False"):
                expected_result_map[key] = int(obj_as_str == "True")
            else:
                expected_result_map[key] = round(complex(obj_as_str).real)

    return expected_result_map



def expected_exception_map_of_test_1_of_to_int(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (int(key.split("_")[-1]) > 4) or (key in key_subset_1):
            expected_exception_map[key] = TypeError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_bool(
        dict_1_of_objs_for_which_to_test_conversions_to_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_bool(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (int(key.split("_")[-1]) > 2) or (key in key_subset_1):
            expected_result_map[key] = None
        else:
            obj = dict_of_objs_for_which_to_test_conversions[key]
            obj_as_str = str(np.array(obj).tolist())
            if obj_as_str in ("True", "False"):
                expected_result_map[key] = (obj_as_str == "True")
            else:
                expected_result_map[key] = bool(round(complex(obj_as_str).real))

    return expected_result_map



def expected_exception_map_of_test_1_of_to_bool(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (int(key.split("_")[-1]) > 2) or (key in key_subset_1):
            expected_exception_map[key] = TypeError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_positive_float(
        dict_1_of_objs_for_which_to_test_conversions_to_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_positive_float(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (int(key.split("_")[-1]) in (1, 3, 5, 7)) or (key in key_subset_1):
            expected_result_map[key] = None
        else:
            obj = dict_of_objs_for_which_to_test_conversions[key]
            obj_as_str = str(np.array(obj).tolist())
            if obj_as_str in ("True", "False"):
                expected_result_map[key] = float(obj_as_str == "True")
            else:
                expected_result_map[key] = complex(obj_as_str).real

    return expected_result_map



def expected_exception_map_of_test_1_of_to_positive_float(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (int(key.split("_")[-1]) in (1, 3, 5, 7)) or (key in key_subset_1):
            if (int(key.split("_")[-1]) in (7,)) or (key in key_subset_1):
                expected_exception_map[key] = TypeError
            else:
                expected_exception_map[key] = ValueError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_nonnegative_float(
        dict_1_of_objs_for_which_to_test_conversions_to_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_nonnegative_float(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (int(key.split("_")[-1]) in (3, 5, 7)) or (key in key_subset_1):
            expected_result_map[key] = None
        else:
            obj = dict_of_objs_for_which_to_test_conversions[key]
            obj_as_str = str(np.array(obj).tolist())
            if obj_as_str in ("True", "False"):
                expected_result_map[key] = float(obj_as_str == "True")
            else:
                expected_result_map[key] = complex(obj_as_str).real

    return expected_result_map



def expected_exception_map_of_test_1_of_to_nonnegative_float(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (int(key.split("_")[-1]) in (3, 5, 7)) or (key in key_subset_1):
            if (int(key.split("_")[-1]) in (7,)) or (key in key_subset_1):
                expected_exception_map[key] = TypeError
            else:
                expected_exception_map[key] = ValueError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_nonnegative_int(
        dict_1_of_objs_for_which_to_test_conversions_to_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_nonnegative_int(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (int(key.split("_")[-1]) in (3, 5, 6, 7)) or (key in key_subset_1):
            expected_result_map[key] = None
        else:
            obj = dict_of_objs_for_which_to_test_conversions[key]
            obj_as_str = str(np.array(obj).tolist())
            if obj_as_str in ("True", "False"):
                expected_result_map[key] = int(obj_as_str == "True")
            else:
                expected_result_map[key] = round(complex(obj_as_str).real)

    return expected_result_map



def expected_exception_map_of_test_1_of_to_nonnegative_int(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (int(key.split("_")[-1]) in (3, 5, 6, 7)) or (key in key_subset_1):
            if (int(key.split("_")[-1]) in (5, 6, 7)) or (key in key_subset_1):
                expected_exception_map[key] = TypeError
            else:
                expected_exception_map[key] = ValueError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_positive_int(
        dict_1_of_objs_for_which_to_test_conversions_to_scalars):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_positive_int(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ((int(key.split("_")[-1]) in (1, 3, 5, 6, 7))
            or (key in key_subset_1)):
            expected_result_map[key] = None
        else:
            obj = dict_of_objs_for_which_to_test_conversions[key]
            obj_as_str = str(np.array(obj).tolist())
            if obj_as_str in ("True", "False"):
                expected_result_map[key] = int(obj_as_str == "True")
            else:
                expected_result_map[key] = round(complex(obj_as_str).real)

    return expected_result_map



def expected_exception_map_of_test_1_of_to_positive_int(
        dict_of_objs_for_which_to_test_conversions):
    key_subset_1 = ("none_obj_1", "int_seq_1", "word_1")

    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ((int(key.split("_")[-1]) in (1, 3, 5, 6, 7))
            or (key in key_subset_1)):
            if (int(key.split("_")[-1]) in (5, 6, 7)) or (key in key_subset_1):
                expected_exception_map[key] = TypeError
            else:
                expected_exception_map[key] = ValueError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_numpy_array(
        dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_numpy_array(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if "seq_of_seq" in key:
            expected_result_map[key] = None
        else:
            obj_to_convert = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = np.array(obj_to_convert)

    return expected_result_map



def expected_exception_map_of_test_1_of_to_numpy_array(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if "seq_of_seq" in key:
            expected_exception_map[key] = TypeError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_real_numpy_array(
        dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_real_numpy_array(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("seq_of_seq" in key) or ("complex" in key) or ("strings" in key):
            expected_result_map[key] = None
        else:
            obj_to_convert = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = np.array(obj_to_convert, dtype="float")

    return expected_result_map



def expected_exception_map_of_test_1_of_to_real_numpy_array(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("seq_of_seq" in key) or ("complex" in key) or ("strings" in key):
            expected_exception_map[key] = TypeError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_real_numpy_array_1d(
        dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_real_numpy_array_1d(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ((("1d" in key) or ("tuple" in key))
            and ("complex" not in key)
            and ("strings" not in key)):
            obj_to_convert = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = np.array(obj_to_convert, dtype="float")
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_real_numpy_array_1d(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ((("1d" in key) or ("tuple" in key))
            and ("complex" not in key)
            and ("strings" not in key)):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_real_numpy_matrix(
        dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_real_numpy_matrix(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("matrix" in key) and ("complex" not in key):
            obj_to_convert = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = np.array(obj_to_convert, dtype="float")
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_real_numpy_matrix(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("matrix" in key) and ("complex" not in key):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_real_two_column_numpy_matrix(
        dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_real_two_column_numpy_matrix(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("two_column" in key) and ("complex" not in key):
            obj_to_convert = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = np.array(obj_to_convert, dtype="float")
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_real_two_column_numpy_matrix(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("two_column" in key) and ("complex" not in key):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_real_numpy_array_3d(
        dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_real_numpy_array_3d(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("3d" in key) and ("complex" not in key):
            obj_to_convert = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = np.array(obj_to_convert, dtype="float")
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_real_numpy_array_3d(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("3d" in key) and ("complex" not in key):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_nonnegative_numpy_array(
        dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_nonnegative_numpy_array(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (("seq_of_seq" not in key)
            and ("complex" not in key)
            and ("strings" not in key)
            and ("_negative" not in key)):
            obj_to_convert = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = np.array(obj_to_convert, dtype="float")
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_nonnegative_numpy_array(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if (("seq_of_seq" not in key)
            and ("complex" not in key)
            and ("strings" not in key)
            and ("_negative" not in key)):
            expected_exception_map[key] = None
        else:
            if "_negative" in key:
                expected_exception_map[key] = ValueError
            else:
                expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_nonnegative_numpy_matrix(
        dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_nonnegative_numpy_matrix(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("matrix" in key) and (("nonnegative" in key) or ("bool" in key)):
            obj_to_convert = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = np.array(obj_to_convert, dtype="float")
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_nonnegative_numpy_matrix(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("matrix" in key) and (("nonnegative" in key) or ("bool" in key)):
            expected_exception_map[key] = None
        else:
            if ("_negative" in key) and ("matrix" in key):
                expected_exception_map[key] = ValueError
            else:
                expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_bool_numpy_matrix(
        dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_bool_numpy_matrix(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("matrix" in key) and ("bool" in key):
            obj_to_convert = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = np.array(obj_to_convert, dtype="float")
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_bool_numpy_matrix(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("matrix" in key) and ("bool" in key):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_bool_numpy_array_3d(
        dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_bool_numpy_array_3d(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("3d" in key) and ("bool" in key):
            obj_to_convert = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = np.array(obj_to_convert, dtype="float")
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_bool_numpy_array_3d(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("3d" in key) and ("bool" in key):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



def test_1_of_to_complex_numpy_array(
        dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_complex_numpy_array(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("seq_of_seq" in key) or ("strings" in key):
            expected_result_map[key] = None
        else:
            obj_to_convert = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = np.array(obj_to_convert, dtype=complex)

    return expected_result_map



def expected_exception_map_of_test_1_of_to_complex_numpy_array(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("seq_of_seq" in key) or ("strings" in key):
            expected_exception_map[key] = TypeError
        else:
            expected_exception_map[key] = None

    return expected_exception_map



def test_1_of_to_complex_numpy_matrix(
        dict_1_of_objs_for_which_to_test_conversions_to_numpy_arrays):
    kwargs = {"dict_of_objs_for_which_to_test_conversions": \
              random.choice(list(locals().values())),
              "name_of_test": \
              inspect.stack()[0][3],}
    run_generic_test(**kwargs)

    return None



def expected_result_map_of_test_1_of_to_complex_numpy_matrix(
        dict_of_objs_for_which_to_test_conversions):
    expected_result_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("matrix" in key):
            obj_to_convert = dict_of_objs_for_which_to_test_conversions[key]
            expected_result_map[key] = np.array(obj_to_convert, dtype=complex)
        else:
            expected_result_map[key] = None

    return expected_result_map



def expected_exception_map_of_test_1_of_to_complex_numpy_matrix(
        dict_of_objs_for_which_to_test_conversions):
    expected_exception_map = dict()
    for key in dict_of_objs_for_which_to_test_conversions:
        if ("matrix" in key):
            expected_exception_map[key] = None
        else:
            expected_exception_map[key] = TypeError

    return expected_exception_map



###########################
## Define error messages ##
###########################
