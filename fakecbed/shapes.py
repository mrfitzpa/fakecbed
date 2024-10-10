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
r"""For creating undistorted geometric shapes.

This module contains classes that represent different undistorted geometric
shapes that can be combined to construct intensity patterns that imitate
convergent beam diffraction beam (CBED) patterns. As a shorthand, we refer to
these intensity patterns that imitate CBED patterns as "fake CBED patterns".

Users can create images of fake CBED patterns using the
:mod:`fakecbed.discretized` module. These images are formed by specifying a
series of parameters, with the most important parameters being: the set of
undistorted geometric shapes and Gaussian filters that determine the undistorted
noiseless fake CBED pattern; and a distortion model which transforms the
undistorted fake CBED pattern into a distorted fake CBED pattern.

Let :math:`u_{x}` and :math:`u_{y}` be the fractional horizontal and vertical
coordinates, respectively, of a point in an undistorted image, where
:math:`\left(u_{x},u_{y}\right)=\left(0,0\right)` is the bottom left corner of
the image. Secondly, let :math:`q_{x}` and :math:`q_{y}` be the fractional
horizontal and vertical coordinates, respectively, of a point in a distorted
image, where :math:`\left(q_{x},q_{y}\right)=\left(0,0\right)` is the bottom
left corner of the image. When users specify a distortion model, represented by
:obj:`distoptica.DistortionModel` object, they also specify a coordinate
transformation which maps a given coordinate pair
:math:`\left(u_{x},u_{y}\right)` to a corresponding coordinate pair
:math:`\left(q_{x},q_{y}\right)`, and implicitly a right-inverse to said
coordinate transformation that maps a coordinate pair
:math:`\left(q_{x},q_{y}\right)` to a corresponding coordinate pair
:math:`\left(u_{x},u_{y}\right)`, when such a relationship exists for
:math:`\left(q_{x},q_{y}\right)`.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For accessing attributes of functions.
import inspect

# For randomly selecting items in dictionaries.
import random

# For performing deep copies.
import copy

# For defining abstract base classes.
import abc



# For general array handling.
import numpy as np
import torch

# For validating and converting objects.
import czekitout.check
import czekitout.convert

# For defining classes that support enforced validation, updatability,
# pre-serialization, and de-serialization.
import fancytypes

# For validating, pre-serializing, and de-pre-serializing certain objects.
import distoptica



##################################
## Define classes and functions ##
##################################

# List of public objects in module.
__all__ = ["UniformDisk",
           "Peak",
           "Band",
           "PlaneWave",
           "NonUniformDisk"]



def _r(x, y, center):
    x_c, y_c = center
    x_minus_x_c = x-x_c
    y_minus_y_c = y-y_c

    r = torch.sqrt(x_minus_x_c*x_minus_x_c + y_minus_y_c*y_minus_y_c)

    return r



def _r_sq(x, y, center):
    x_c, y_c = center
    x_minus_x_c = x-x_c
    y_minus_y_c = y-y_c

    r_sq = x_minus_x_c*x_minus_x_c + y_minus_y_c*y_minus_y_c

    return r_sq



def _check_and_convert_cartesian_coords(params):
    x, y = params["cartesian_coords"]

    params["real_torch_matrix"] = x
    params["name_of_alias_of_real_torch_matrix"] = "x"
    x = distoptica._check_and_convert_real_torch_matrix(params)

    params["real_torch_matrix"] = y
    params["name_of_alias_of_real_torch_matrix"] = "y"
    y = distoptica._check_and_convert_real_torch_matrix(params)

    del params["real_torch_matrix"]
    del params["name_of_alias_of_real_torch_matrix"]

    if x.shape != y.shape:
        unformatted_err_msg = _check_and_convert_cartesian_coords_err_msg_1
        err_msg = unformatted_err_msg.format("x", "y")
        raise ValueError(err_msg)

    cartesian_coords = (x, y)

    return cartesian_coords



def _check_and_convert_device(params):
    params["name_of_obj_alias_of_torch_device_obj"] = "device"
    device = _check_and_convert_torch_device_obj(params)

    del params["name_of_obj_alias_of_torch_device_obj"]

    return device



def _check_and_convert_torch_device_obj(params):
    obj_name = params["name_of_obj_alias_of_torch_device_obj"]
    obj = params[obj_name]

    if obj is None:
        torch_device_obj = torch.device("cuda"
                                        if torch.cuda.is_available()
                                        else "cpu")
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": (torch.device, type(None))}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        torch_device_obj = obj

    return torch_device_obj



_default_x = ((0.5,),)
_default_y = _default_x
_default_device = None



_cls_alias = fancytypes.PreSerializableAndUpdatable
class _BaseShape(_cls_alias):
    def __init__(self, ctor_params):
        kwargs = ctor_params
        kwargs["skip_cls_tests"] = True
        fancytypes.PreSerializableAndUpdatable.__init__(self, **kwargs)

        return None



    @classmethod
    def get_validation_and_conversion_funcs(cls):
        validation_and_conversion_funcs = \
            cls._validation_and_conversion_funcs_.copy()

        return validation_and_conversion_funcs


    
    @classmethod
    def get_pre_serialization_funcs(cls):
        pre_serialization_funcs = \
            cls._pre_serialization_funcs_.copy()

        return pre_serialization_funcs


    
    @classmethod
    def get_de_pre_serialization_funcs(cls):
        de_pre_serialization_funcs = \
            cls._de_pre_serialization_funcs_.copy()

        return de_pre_serialization_funcs



    def eval(self, x=_default_x, y=_default_y, device=_default_device):
        params = {"cartesian_coords": (x, y), "device": device}
        device = _check_and_convert_device(params)
        x, y = _check_and_convert_cartesian_coords(params)
        
        result = self._eval(x, y)

        return result



    def _eval(self, x, y):
        pass



def _check_and_convert_center(params):
    center = distoptica._check_and_convert_center(params)

    return center



def _pre_serialize_center(center):
    serializable_rep = distoptica._pre_serialize_center(center)
    
    return serializable_rep



def _de_pre_serialize_center(serializable_rep):
    center = distoptica._de_pre_serialize_center(serializable_rep)

    return center



def _check_and_convert_radius(params):
    obj_name = "radius"
    obj = params[obj_name]
    radius = czekitout.convert.to_positive_float(obj, obj_name)

    return radius



def _pre_serialize_radius(radius):
    serializable_rep = radius
    
    return serializable_rep



def _de_pre_serialize_radius(serializable_rep):
    radius = serializable_rep

    return radius



def _check_and_convert_intra_disk_val(params):
    obj_name = "intra_disk_val"
    obj = params[obj_name]
    intra_disk_val = czekitout.convert.to_float(obj, obj_name)

    return intra_disk_val



def _pre_serialize_intra_disk_val(intra_disk_val):
    serializable_rep = intra_disk_val
    
    return serializable_rep



def _de_pre_serialize_intra_disk_val(serializable_rep):
    intra_disk_val = serializable_rep

    return intra_disk_val



_default_center = distoptica._default_center
_default_radius = 0.05
_default_intra_disk_val = 1



class UniformDisk(_BaseShape):
    r"""Insert description here.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(``2``,)), optional
        Insert description here.
    radius : `float`, optional
        Insert description here.
    intra_disk_val : `float`, optional
        Insert description here.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"center": _check_and_convert_center,
         "radius": _check_and_convert_radius,
         "intra_disk_val": _check_and_convert_intra_disk_val}

    _pre_serialization_funcs = \
        {"center": _pre_serialize_center,
         "radius": _pre_serialize_radius,
         "intra_disk_val": _pre_serialize_intra_disk_val}

    _de_pre_serialization_funcs = \
        {"center": _de_pre_serialize_center,
         "radius": _de_pre_serialize_radius,
         "intra_disk_val": _de_pre_serialize_intra_disk_val}

    def __init__(self,
                 center=_default_center,
                 radius=_default_radius,
                 intra_disk_val=_default_intra_disk_val):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        _BaseShape.__init__(self, ctor_params)

        self._post_base_update()

        return None



    def _post_base_update(self):
        self._center = self._core_attrs["center"]
        self._radius = self._core_attrs["radius"]
        self._intra_disk_val = self._core_attrs["intra_disk_val"]

        self._step_func = (self._step_func_1
                           if (self._intra_disk_val == 1)
                           else self._step_func_2)

        return None



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self._post_base_update()

        return None



    def _eval(self, x, y):
        center = self._center
        
        r_sq = _r_sq(x, y, center)

        result = self._step_func(r_sq)

        return result



    def _step_func_1(self, r_sq):
        R = self._radius

        R_sq = R*R

        result = (R_sq > r_sq)

        return result



    def _step_func_2(self, r_sq):
        R = self._radius
        A = self._intra_disk_val

        R_sq = R*R

        one = torch.tensor(1.0, device=r_sq.device)

        result = A * torch.heaviside(R_sq-r_sq, one)

        return result



def _check_and_convert_width(params):
    obj_name = "width"
    obj = params[obj_name]
    width = czekitout.convert.to_positive_float(obj, obj_name)

    return width



def _pre_serialize_width(width):
    serializable_rep = width
    
    return serializable_rep



def _de_pre_serialize_width(serializable_rep):
    width = serializable_rep

    return width



def _check_and_convert_val_at_center(params):
    obj_name = "val_at_center"
    obj = params[obj_name]
    val_at_center = czekitout.convert.to_float(obj, obj_name)

    return val_at_center



def _pre_serialize_val_at_center(val_at_center):
    serializable_rep = val_at_center
    
    return serializable_rep



def _de_pre_serialize_val_at_center(serializable_rep):
    val_at_center = serializable_rep

    return val_at_center



_default_width = 0.05
_default_val_at_center = 1



class _GaussianPeak(_BaseShape):
    r"""Insert description here.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(``2``,)), optional
        Insert description here.
    width : `float`, optional
        Insert description here.
    val_at_center : `float`, optional
        Insert description here.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"center": _check_and_convert_center,
         "width": _check_and_convert_width,
         "val_at_center": _check_and_convert_val_at_center}

    _pre_serialization_funcs = \
        {"center": _pre_serialize_center,
         "width": _pre_serialize_width,
         "val_at_center": _pre_serialize_val_at_center}

    _de_pre_serialization_funcs = \
        {"center": _de_pre_serialize_center,
         "width": _de_pre_serialize_width,
         "val_at_center": _de_pre_serialize_val_at_center}

    def __init__(self,
                 center=_default_center,
                 width=_default_width,
                 val_at_center=_default_val_at_center):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        _BaseShape.__init__(self, ctor_params)

        self._post_base_update()

        return None



    def _post_base_update(self):
        self._center = self._core_attrs["center"]
        self._width = self._core_attrs["width"]
        self._val_at_center = self._core_attrs["val_at_center"]

        return None



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self._post_base_update()

        return None



    def _eval(self, x, y):
        result = self._eval_1(x, y)

        return result



    def _eval_1(self, x, y):
        center = self._center
        
        r = _r(x, y, center)
        
        result = self._eval_2(r)

        return result



    def _eval_2(self, r):
        sigma = self._width
        A = self._val_at_center
        r_over_sigma = r/sigma
        
        result = A * torch.exp(-0.5 * r_over_sigma * r_over_sigma)

        return result



class _LorentzianPeak(_BaseShape):
    r"""Insert description here.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(``2``,)), optional
        Insert description here.
    width : `float`, optional
        Insert description here.
    val_at_center : `float`, optional
        Insert description here.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"center": _check_and_convert_center,
         "width": _check_and_convert_width,
         "val_at_center": _check_and_convert_val_at_center}

    _pre_serialization_funcs = \
        {"center": _pre_serialize_center,
         "width": _pre_serialize_width,
         "val_at_center": _pre_serialize_val_at_center}

    _de_pre_serialization_funcs = \
        {"center": _de_pre_serialize_center,
         "width": _de_pre_serialize_width,
         "val_at_center": _de_pre_serialize_val_at_center}

    def __init__(self,
                 center=_default_center,
                 width=_default_width,
                 val_at_center=_default_val_at_center):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        _BaseShape.__init__(self, ctor_params)

        self._post_base_update()

        return None



    def _post_base_update(self):
        self._center = self._core_attrs["center"]
        self._width = self._core_attrs["width"]
        self._val_at_center = self._core_attrs["val_at_center"]

        return None



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self._post_base_update()

        return None



    def _eval(self, x, y):
        result = self._eval_1(x, y)

        return result



    def _eval_1(self, x, y):
        center = self._center

        r = _r(x, y, center)
        
        result = self._eval_2(r)

        return result



    def _eval_2(self, r):
        sigma = self._width
        A = self._val_at_center
        r_over_sigma = r/sigma
        
        result = A / (r_over_sigma*r_over_sigma + 1)

        return result



class _ExponentialPeak(_BaseShape):
    r"""Insert description here.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(``2``,)), optional
        Insert description here.
    width : `float`, optional
        Insert description here.
    val_at_center : `float`, optional
        Insert description here.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"center": _check_and_convert_center,
         "width": _check_and_convert_width,
         "val_at_center": _check_and_convert_val_at_center}

    _pre_serialization_funcs = \
        {"center": _pre_serialize_center,
         "width": _pre_serialize_width,
         "val_at_center": _pre_serialize_val_at_center}

    _de_pre_serialization_funcs = \
        {"center": _de_pre_serialize_center,
         "width": _de_pre_serialize_width,
         "val_at_center": _de_pre_serialize_val_at_center}

    def __init__(self,
                 center=_default_center,
                 width=_default_width,
                 val_at_center=_default_val_at_center):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        _BaseShape.__init__(self, ctor_params)

        self._post_base_update()

        return None



    def _post_base_update(self):
        self._center = self._core_attrs["center"]
        self._width = self._core_attrs["width"]
        self._val_at_center = self._core_attrs["val_at_center"]

        return None



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self._post_base_update()

        return None



    def _eval(self, x, y):
        result = self._eval_1(x, y)

        return result



    def _eval_1(self, x, y):
        center = self._center
        
        r = _r(x, y, center)
        
        result = self._eval_2(r)

        return result



    def _eval_2(self, r):
        sigma = self._width
        A = self._val_at_center
        r_over_sigma = r/sigma
        
        result = A * torch.exp(-r_over_sigma)

        return result



def _check_and_convert_functional_form(params):
    obj_name = "functional_form"
    obj = params[obj_name]
    functional_form = czekitout.convert.to_str_from_str_like(obj, obj_name)

    accepted_values = ("gaussian", "lorentzian", "exponential")
    if functional_form not in accepted_values:
        raise ValueError(_check_and_convert_functional_form_err_msg_1)

    return functional_form



def _pre_serialize_functional_form(functional_form):
    serializable_rep = functional_form
    
    return serializable_rep



def _de_pre_serialize_functional_form(serializable_rep):
    functional_form = serializable_rep

    return functional_form



_default_functional_form = "gaussian"



class Peak(_BaseShape):
    r"""Insert description here.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(``2``,)), optional
        Insert description here.
    width : `float`, optional
        Insert description here.
    val_at_center : `float`, optional
        Insert description here.
    functional_form: ``"gaussian"`` | ``"lorentzian"`` | ``"exponential"``, optional
        Insert description here.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"center": _check_and_convert_center,
         "width": _check_and_convert_width,
         "val_at_center": _check_and_convert_val_at_center,
         "functional_form": _check_and_convert_functional_form}

    _pre_serialization_funcs = \
        {"center": _pre_serialize_center,
         "width": _pre_serialize_width,
         "val_at_center": _pre_serialize_val_at_center,
         "functional_form": _pre_serialize_functional_form}

    _de_pre_serialization_funcs = \
        {"center": _de_pre_serialize_center,
         "width": _de_pre_serialize_width,
         "val_at_center": _de_pre_serialize_val_at_center,
         "functional_form": _de_pre_serialize_functional_form}

    def __init__(self,
                 center=_default_center,
                 width=_default_width,
                 val_at_center=_default_val_at_center,
                 functional_form=_default_functional_form):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        _BaseShape.__init__(self, ctor_params)

        self._post_base_update()

        return None



    def _post_base_update(self):
        kwargs = {"center": self._core_attrs["center"],
                  "width": self._core_attrs["width"],
                  "val_at_center": self._core_attrs["val_at_center"]}
        
        functional_form = self._core_attrs["functional_form"]
        if functional_form == "gaussian":
            self._peak = _GaussianPeak(**kwargs)
        elif functional_form == "lorentzian":
            self._peak = _LorentzianPeak(**kwargs)
        else:
            self._peak = _ExponentialPeak(**kwargs)

        return None



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self._post_base_update()

        return None



    def _eval(self, x, y):
        result = self._eval_1(x, y)

        return result



    def _eval_1(self, x, y):
        result = self._peak._eval_1(x, y)

        return result



    def _eval_2(self, r):
        result = self._peak._eval_2(r)

        return result



def _check_and_convert_end_pt_1(params):
    obj_name = "end_pt_1"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    end_pt_1 = czekitout.convert.to_pair_of_floats(**kwargs)

    return end_pt_1



def _pre_serialize_end_pt_1(end_pt_1):
    serializable_rep = end_pt_1
    
    return serializable_rep



def _de_pre_serialize_end_pt_1(serializable_rep):
    end_pt_1 = serializable_rep

    return end_pt_1



def _check_and_convert_end_pt_2(params):
    obj_name = "end_pt_2"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    end_pt_2 = czekitout.convert.to_pair_of_floats(**kwargs)

    return end_pt_2



def _pre_serialize_end_pt_2(end_pt_2):
    serializable_rep = end_pt_2
    
    return serializable_rep



def _de_pre_serialize_end_pt_2(serializable_rep):
    end_pt_2 = serializable_rep

    return end_pt_2



def _check_and_convert_intra_band_val_without_decay(params):
    obj_name = "intra_band_val_without_decay"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    intra_band_val_without_decay = czekitout.convert.to_float(**kwargs)

    return intra_band_val_without_decay



def _pre_serialize_intra_band_val_without_decay(intra_band_val_without_decay):
    serializable_rep = intra_band_val_without_decay
    
    return serializable_rep



def _de_pre_serialize_intra_band_val_without_decay(serializable_rep):
    intra_band_val_without_decay = serializable_rep

    return intra_band_val_without_decay



def _check_and_convert_unnormalized_decay_peak(params):
    obj_name = "unnormalized_decay_peak"
    obj = copy.deepcopy(params[obj_name])
    
    if obj is None:
        kwargs = {"center": (0, 0),
                  "width": float("inf"),
                  "val_at_center": 1,
                  "functional_form": "gaussian"}
        unnormalized_decay_peak = Peak(**kwargs)
    else:
        accepted_types = (Peak, type(None))
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        unnormalized_decay_peak = obj

    return unnormalized_decay_peak



def _pre_serialize_unnormalized_decay_peak(unnormalized_decay_peak):
    serializable_rep = unnormalized_decay_peak.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_unnormalized_decay_peak(serializable_rep):
    unnormalized_decay_peak = Peak.de_pre_serialize(serializable_rep)

    return unnormalized_decay_peak



_default_end_pt_1 = (0, 0.5)
_default_end_pt_2 = (1, 0.5)
_default_width = 0.05
_default_intra_band_val_without_decay = 1
_default_unnormalized_decay_peak = None



class Band(_BaseShape):
    r"""Insert description here.

    Parameters
    ----------
    end_pt_1 : `array_like` (`float`, shape=(``2``,)), optional
        Insert description here.
    end_pt_2 : `array_like` (`float`, shape=(``2``,)), optional
        Insert description here.
    width : `float`, optional
        Insert description here.
    intra_band_val_without_decay : `float`, optional
        Insert description here.
    unnormalized_decay_peak : :class:`fakecbed.shapes.Peak` | `None`, optional
        Insert description here.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"end_pt_1": _check_and_convert_end_pt_1,
         "end_pt_2": _check_and_convert_end_pt_2,
         "width": _check_and_convert_width,
         "intra_band_val_without_decay": \
         _check_and_convert_intra_band_val_without_decay,
         "unnormalized_decay_peak": _check_and_convert_unnormalized_decay_peak}

    _pre_serialization_funcs = \
        {"end_pt_1": _pre_serialize_end_pt_1,
         "end_pt_2": _pre_serialize_end_pt_2,
         "width": _pre_serialize_width,
         "intra_band_val_without_decay": \
         _pre_serialize_intra_band_val_without_decay,
         "unnormalized_decay_peak": _pre_serialize_unnormalized_decay_peak}

    _de_pre_serialization_funcs = \
        {"end_pt_1": _de_pre_serialize_end_pt_1,
         "end_pt_2": _de_pre_serialize_end_pt_2,
         "width": _de_pre_serialize_width,
         "intra_band_val_without_decay": \
         _de_pre_serialize_intra_band_val_without_decay,
         "unnormalized_decay_peak": _de_pre_serialize_unnormalized_decay_peak}

    def __init__(self,
                 end_pt_1=\
                 _default_end_pt_1,
                 end_pt_2=\
                 _default_end_pt_2,
                 width=\
                 _default_width,
                 intra_band_val_without_decay=\
                 _default_intra_band_val_without_decay,
                 unnormalized_decay_peak=\
                 _default_unnormalized_decay_peak):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        _BaseShape.__init__(self, ctor_params)

        self._post_base_update()

        return None



    def _post_base_update(self):
        x_1, y_1 = self._core_attrs["end_pt_1"]
        x_2, y_2 = self._core_attrs["end_pt_2"]

        length = np.sqrt((x_2-x_1)**2 + (y_2-y_1)**2)
        theta = np.arctan2(y_2-y_1, x_2-x_1)
        phi = theta + (np.pi/2)

        x_3 = (x_1 + (length/2)*np.cos(theta)).item()
        y_3 = (y_1 + (length/2)*np.sin(theta)).item()
        x_4 = (x_3 + (length/2)*np.cos(phi)).item()
        y_4 = (y_3 + (length/2)*np.sin(phi)).item()

        self._a_1 = y_2-y_1 if (x_1 != x_2) else 1
        self._b_1 = x_1-x_2
        self._c_1 = (x_2*y_1-x_1*y_2) if (x_1 != x_2) else -x_1

        self._a_2 = y_4-y_3 if (x_3 != x_4) else 1
        self._b_2 = x_3-x_4
        self._c_2 = (x_4*y_3-x_3*y_4) if (x_3 != x_4) else -x_3

        self._denom_of_d_1 = \
            np.sqrt(self._a_1*self._a_1 + self._b_1*self._b_1).item()
        self._denom_of_d_2 = \
            np.sqrt(self._a_2*self._a_2 + self._b_2*self._b_2).item()
        
        self._width = \
            self._core_attrs["width"]
        self._length = \
            length
        self._intra_band_val_without_decay = \
            self._core_attrs["intra_band_val_without_decay"]
        self._unnormalized_decay_peak = \
            self._core_attrs["unnormalized_decay_peak"]

        return None



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self._post_base_update()

        return None



    def _d_1(self, x, y):
        a_1 = self._a_1
        b_1 = self._b_1
        c_1 = self._c_1
        denom_of_d_1 = self._denom_of_d_1
        
        d_1 = torch.abs(a_1*x + b_1*y + c_1) / denom_of_d_1

        return d_1



    def _d_2(self, x, y):
        a_2 = self._a_2
        b_2 = self._b_2
        c_2 = self._c_2
        denom_of_d_2 = self._denom_of_d_2
        
        d_2 = torch.abs(a_2*x + b_2*y + c_2) / denom_of_d_2

        return d_2



    def _eval(self, x, y):
        result = self._eval_1(x, y)

        return result



    def _eval_1(self, x, y):
        w_over_2 = self._width/2
        l_over_2 = self._length/2
        A = self._intra_band_val_without_decay
        unnormalized_decay_peak = self._unnormalized_decay_peak
        
        d_1 = self._d_1(x, y)
        d_2 = self._d_2(x, y)

        one = torch.tensor(1.0, device=d_1.device)
        
        result = (A
                  * torch.heaviside(w_over_2 - d_1, one)
                  * torch.heaviside(l_over_2 - d_2, one)
                  * unnormalized_decay_peak._eval(x, y))

        return result



    def _eval_2(self, x, y, r):
        w = self._width
        l = self._length
        A = self._intra_band_val_without_decay
        unnormalized_decay_peak = self._unnormalized_decay_peak
        
        d_1 = self._d_1(x, y)
        d_2 = self._d_2(x, y)

        one = torch.tensor(1.0, device=d_1.device)
        
        result = (A
                  * torch.heaviside(0.5*w - d_1, one)
                  * torch.heaviside(0.5*l - d_2, one)
                  * unnormalized_decay_peak._eval_2(r))

        return result



def _check_and_convert_amplitude(params):
    obj_name = "amplitude"
    obj = params[obj_name]
    amplitude = czekitout.convert.to_float(obj, obj_name)

    return amplitude



def _pre_serialize_amplitude(amplitude):
    serializable_rep = amplitude
    
    return serializable_rep



def _de_pre_serialize_amplitude(serializable_rep):
    amplitude = serializable_rep

    return amplitude



def _check_and_convert_wavelength(params):
    obj_name = "wavelength"
    obj = params[obj_name]
    wavelength = czekitout.convert.to_positive_float(obj, obj_name)

    return wavelength



def _pre_serialize_wavelength(wavelength):
    serializable_rep = wavelength
    
    return serializable_rep



def _de_pre_serialize_wavelength(serializable_rep):
    wavelength = serializable_rep

    return wavelength



def _check_and_convert_propagation_direction(params):
    obj_name = "propagation_direction"
    obj = params[obj_name]
    propagation_direction = czekitout.convert.to_float(obj, obj_name)

    return propagation_direction



def _pre_serialize_propagation_direction(propagation_direction):
    serializable_rep = propagation_direction
    
    return serializable_rep



def _de_pre_serialize_propagation_direction(serializable_rep):
    propagation_direction = serializable_rep

    return propagation_direction



def _check_and_convert_phase(params):
    obj_name = "phase"
    obj = params[obj_name]
    phase = czekitout.convert.to_float(obj, obj_name)

    return phase



def _pre_serialize_phase(phase):
    serializable_rep = phase
    
    return serializable_rep



def _de_pre_serialize_phase(serializable_rep):
    phase = serializable_rep

    return phase



_default_amplitude = 1
_default_wavelength = 0.01
_default_propagation_direction = 0
_default_phase = 0



class PlaneWave(_BaseShape):
    r"""Insert description here.

    Parameters
    ----------
    amplitude : `float`, optional
        Insert description here.
    wavelength : `float`, optional
        Insert description here.
    propagation_direction : `float`, optional
        Insert description here.
    phase : `float`, optional
        Insert description here.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"amplitude": _check_and_convert_amplitude,
         "wavelength": _check_and_convert_wavelength,
         "propagation_direction": _check_and_convert_propagation_direction,
         "phase": _check_and_convert_phase}

    _pre_serialization_funcs = \
        {"amplitude": _pre_serialize_amplitude,
         "wavelength": _pre_serialize_wavelength,
         "propagation_direction": _pre_serialize_propagation_direction,
         "phase": _pre_serialize_phase}

    _de_pre_serialization_funcs = \
        {"amplitude": _de_pre_serialize_amplitude,
         "wavelength": _de_pre_serialize_wavelength,
         "propagation_direction": _de_pre_serialize_propagation_direction,
         "phase": _de_pre_serialize_phase}

    def __init__(self,
                 amplitude=_default_amplitude,
                 wavelength=_default_wavelength,
                 propagation_direction=_default_propagation_direction,
                 phase=_default_phase):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        _BaseShape.__init__(self, ctor_params)

        self._post_base_update()

        return None



    def _post_base_update(self):
        self._amplitude = self._core_attrs["amplitude"]
        self._wavelength = self._core_attrs["wavelength"]
        self._propagation_direction = self._core_attrs["propagation_direction"]
        self._phase = self._core_attrs["phase"]

        L = self._wavelength

        u_x = np.cos(self._propagation_direction).item()
        u_y = np.sin(self._propagation_direction).item()

        self._k_x = (2*np.pi/L)*u_x
        self._k_y = (2*np.pi/L)*u_y

        return None



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self._post_base_update()

        return None



    def _eval(self, x, y):
        A = self._amplitude
        phi = self._phase
        k_x = self._k_x
        k_y = self._k_y

        result = A * torch.cos(x*k_x+y*k_y + phi)

        return result



def _check_and_convert_support(params):
    obj_name = "support"
    obj = copy.deepcopy(params[obj_name])
    
    if obj is None:
        support = UniformDisk()
    else:   
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": (UniformDisk, type(None))}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        support = obj

    core_attr_subset = {"intra_disk_val": 1}
    support.update(core_attr_subset)

    return support



def _pre_serialize_support(support):
    serializable_rep = support.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_support(serializable_rep):
    support = UniformDisk.de_pre_serialize(serializable_rep)

    return support



def _check_and_convert_intra_disk_shapes(params):
    try:
        obj_name = "intra_disk_shapes"
        intra_disk_shapes = params[obj_name]
        accepted_types = (UniformDisk, Peak, Band, PlaneWave)

        for intra_disk_shape in intra_disk_shapes:
            kwargs = {"obj": intra_disk_shape,
                      "obj_name": "intra_disk_shape",
                      "accepted_types": accepted_types}
            czekitout.check.if_instance_of_any_accepted_types(**kwargs)
    except:
        raise TypeError(_check_and_convert_intra_disk_shapes_err_msg_1)

    return intra_disk_shapes



def _pre_serialize_intra_disk_shapes(intra_disk_shapes):
    serializable_rep = []
    
    for intra_disk_shape in intra_disk_shapes:
        serializable_rep.append(intra_disk_shape.pre_serialize())
        
    serializable_rep = tuple(serializable_rep)
    
    return serializable_rep



def _de_pre_serialize_intra_disk_shapes(serializable_rep):
    intra_disk_shapes = []
    
    for serialized_intra_disk_shape in serializable_rep:
        if "intra_disk_val" in serialized_intra_disk_shape:
            cls = UniformDisk
        elif "val_at_center" in serialized_intra_disk_shape:
            cls = Peak
        elif "amplitude" in serialized_intra_disk_shape:
            cls = PlaneWave
        else:
            cls = Band
            
        intra_disk_shape = cls.de_pre_serialize(serialized_intra_disk_shape)
        intra_disk_shapes.append(intra_disk_shape)
        
    intra_disk_shapes = tuple(intra_disk_shapes)

    return intra_disk_shapes



_default_support = None
_default_intra_disk_shapes = tuple()



class NonUniformDisk(_BaseShape):
    r"""Insert description here.

    Parameters
    ----------
    support : :class:`fakecbed.shapes.UniformDisk` | `None`, optional
        Insert description here.
    intra_disk_shapes : `array_like` (:class:`fakecbed.shapes.UniformDisk` | :class:`fakecbed.shapes.Peak` | :class:`fakecbed.shapes.Band` | :class:`fakecbed.shapes.PlaneWave`, ndim=1), optional
        Insert description here.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    _validation_and_conversion_funcs = \
        {"support": _check_and_convert_support,
         "intra_disk_shapes": _check_and_convert_intra_disk_shapes}

    _pre_serialization_funcs = \
        {"support": _pre_serialize_support,
         "intra_disk_shapes": _pre_serialize_intra_disk_shapes}

    _de_pre_serialization_funcs = \
        {"support": _de_pre_serialize_support,
         "intra_disk_shapes": _de_pre_serialize_intra_disk_shapes}

    def __init__(self,
                 support=_default_support,
                 intra_disk_shapes=_default_intra_disk_shapes):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        _BaseShape.__init__(self, ctor_params)

        self._post_base_update()

        return None



    def _post_base_update(self):
        self._support = self._core_attrs["support"]
        self._intra_disk_shapes = self._core_attrs["intra_disk_shapes"]

        return None



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self._post_base_update()

        return None



    def _eval(self, x, y):
        result = (self._eval_without_support(x, y)
                  * self._eval_without_intra_disk_shapes(x, y))

        return result



    def _eval_without_intra_disk_shapes(self, x, y):
        support = self._support
        result = support._eval(x, y)

        return result



    def _eval_without_support(self, x, y):
        intra_disk_shapes = self._intra_disk_shapes
        
        result = torch.zeros_like(x)
        for intra_disk_shape in intra_disk_shapes:
            result += intra_disk_shape._eval(x, y)
        result = torch.abs(result)

        return result



###########################
## Define error messages ##
###########################

_check_and_convert_cartesian_coords_err_msg_1 = \
    distoptica._check_and_convert_u_x_and_u_y_err_msg_1

_check_and_convert_functional_form_err_msg_1 = \
    ("The object ``functiona_form`` must be set to ``'gaussian'``, "
     "``'lorentzian'``, or ``'exponential'``.")

_check_and_convert_intra_disk_shapes_err_msg_1 = \
    ("The object ``intra_disk_shapes`` must be a sequence of objects of any "
     "of the following types: (`fakecbed.shapes.UniformDisk`, "
     "`fakecbed.shapes.Peak`, `fakecbed.shapes.Band`, "
     "`fakecbed.shapes.PlaneWave`).")
