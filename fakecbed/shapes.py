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

This module contains classes that represent the intensity patterns of different
undistorted geometric shapes that can be combined to construct intensity
patterns that imitate convergent beam diffraction beam (CBED) patterns. As a
shorthand, we refer to these intensity patterns that imitate CBED patterns as
"fake CBED patterns".

Users can create images of fake CBED patterns using the
:mod:`fakecbed.discretized` module. These images are formed by specifying a
series of parameters, with the most important parameters being: the set of
intensity patterns of undistorted geometric shapes and Gaussian filters that
determine the undistorted noiseless fake CBED pattern; and a distortion model
which transforms the undistorted fake CBED pattern into a distorted fake CBED
pattern.

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
__all__ = ["UniformEllipse",
           "Peak",
           "Band",
           "PlaneWave",
           "NonUniformEllipse"]



def _calc_u_r_sq(u_x, u_y, center):
    u_x_c, u_y_c = center
    delta_u_x = u_x-u_x_c
    delta_u_y = u_y-u_y_c

    u_r_sq = delta_u_x*delta_u_x + delta_u_y*delta_u_y

    return u_r_sq



def _calc_u_r(u_x, u_y, center):
    u_r_sq = _calc_u_r_sq(u_x, u_y, center)
    u_r = torch.sqrt(u_r_sq)

    return u_r



def _check_and_convert_cartesian_coords(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]

    u_x, u_y = obj

    params["real_torch_matrix"] = u_x
    params["name_of_alias_of_real_torch_matrix"] = "u_x"
    u_x = _check_and_convert_real_torch_matrix(params)

    params["real_torch_matrix"] = u_y
    params["name_of_alias_of_real_torch_matrix"] = "u_y"
    u_y = _check_and_convert_real_torch_matrix(params)

    del params["real_torch_matrix"]
    del params["name_of_alias_of_real_torch_matrix"]

    if u_x.shape != u_y.shape:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format("u_x", "u_y")
        raise ValueError(err_msg)

    cartesian_coords = (u_x, u_y)

    return cartesian_coords



def _check_and_convert_real_torch_matrix(params):
    current_func_name = inspect.stack()[0][3]
    char_idx = 19
    obj_name = current_func_name[char_idx:]
    obj = params[obj_name]
    
    name_of_alias_of_real_torch_matrix = \
        params["name_of_alias_of_real_torch_matrix"]

    try:
        if not isinstance(obj, torch.Tensor):
            kwargs = {"obj": obj,
                      "obj_name": name_of_alias_of_real_torch_matrix}
            obj = czekitout.convert.to_real_numpy_matrix(**kwargs)

            obj = torch.tensor(obj,
                               dtype=torch.float32,
                               device=params["device"])
    
        if len(obj.shape) != 2:
            raise
            
        real_torch_matrix = obj.to(device=params["device"], dtype=torch.float32)

    except:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format(name_of_alias_of_real_torch_matrix)
        raise TypeError(err_msg)

    return real_torch_matrix



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



def _check_and_convert_skip_validation_and_conversion(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    skip_validation_and_conversion = czekitout.convert.to_bool(**kwargs)

    return skip_validation_and_conversion



_default_u_x = ((0.5,),)
_default_u_y = _default_x
_default_device = None
_default_skip_validation_and_conversion = False



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



    def eval(self,
             u_x=\
             _default_u_x,
             u_y=\
             _default_u_y,
             device=\
             _default_device,
             skip_validation_and_conversion=\
             _default_skip_validation_and_conversion):
        func_alias = _check_and_convert_skip_validation_and_conversion
        skip_validation_and_conversion = func_alias(params)

        if (skip_validation_and_conversion == False):
            params = {"cartesian_coords": (u_x, u_y), "device": device}
            device = _check_and_convert_device(params)
            u_x, u_y = _check_and_convert_cartesian_coords(params)
        
        result = self._eval(u_x, u_y)

        return result



    def _eval(self, u_x, u_y):
        pass



def _check_and_convert_center(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]

    cls_alias = \
        distoptica.CoordTransformParams
    validation_and_conversion_funcs = \
        cls_alias.get_validation_and_conversion_funcs()
    validation_and_conversion_func = \
        validation_and_conversion_funcs[obj_name]
    center = \
        validation_and_conversion_func(params)

    return center



def _pre_serialize_center(center):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[15:]

    cls_alias = \
        distoptica.CoordTransformParams
    pre_serialization_funcs = \
        cls_alias.get_pre_serialization_funcs()
    pre_serialization_func = \
        pre_serialization_funcs[obj_name]
    serializable_rep = \
        pre_serialization_func(obj_to_pre_serialize)
    
    return serializable_rep



def _de_pre_serialize_center(serializable_rep):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[18:]

    cls_alias = \
        distoptica.CoordTransformParams
    de_pre_serialization_funcs = \
        cls_alias.get_de_pre_serialization_funcs()
    de_pre_serialization_func = \
        de_pre_serialization_funcs[obj_name]
    center = \
        de_pre_serialization_func(serializable_rep)

    return center



def _check_and_convert_semi_major_axis(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    semi_major_axis = czekitout.convert.to_positive_float(**kwargs)

    return semi_major_axis



def _pre_serialize_semi_major_axis(semi_major_axis):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_semi_major_axis(serializable_rep):
    semi_major_axis = serializable_rep

    return semi_major_axis



def _check_and_convert_eccentricity(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    eccentricity = czekitout.convert.to_nonnegative_float(**kwargs)

    if eccentricity > 1:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise TypeError(err_msg)

    return eccentricity



def _pre_serialize_eccentricity(eccentricity):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_eccentricity(serializable_rep):
    eccentricity = serializable_rep

    return eccentricity



def _check_and_convert_rotation_angle(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    rotation_angle = czekitout.convert.to_float(**kwargs) % (2*np.pi)

    return rotation_angle



def _pre_serialize_rotation_angle(rotation_angle):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_rotation_angle(serializable_rep):
    rotation_angle = serializable_rep

    return rotation_angle



def _check_and_convert_intra_ellipse_val(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    intra_ellipse_val = czekitout.convert.to_float(**kwargs)

    return intra_ellipse_val



def _pre_serialize_intra_ellipse_val(intra_ellipse_val):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_intra_ellipse_val(serializable_rep):
    intra_ellipse_val = serializable_rep

    return intra_ellipse_val



_default_center = (0.5, 0.5)
_default_semi_major_axis = 0.05
_default_eccentricity = 0
_default_rotation_angle = 0
_default_intra_ellipse_val = 1



class UniformEllipse(_BaseShape):
    r"""The intensity pattern of a uniform ellipse.

    Let :math:`\left(u_{x;c;O},u_{y;c;O}\right)`, :math:`a_{O}`, :math:`e_{O}`,
    and :math:`\theta_{O}` be the center, the semi-major axis, the eccentricity,
    and the rotation angle of the ellipse respectively. Furthermore, let
    :math:`A_{O}` be the value of the intensity pattern inside the ellipse. The
    intensity pattern is given by:

    .. math ::
        \mathcal{I}_{O}\left(u_{x},u_{y}\right)=
        A_{O}\Theta\left(\Theta_{\arg;O}\left(u_{x},u_{y}\right)\right),
        :label: intensity_pattern_of_uniform_ellipse__1

    where :math:`\Theta\left(\cdots\right)` is the Heaviside step function, and

    .. math ::
        &\Theta_{\arg;O}\left(u_{x},u_{y}\right)\\&\quad=
        \left\{ 1-e_{O}^{2}\right\} a_{O}^{2}\\
        &\quad\quad-\left\{ 1-e_{O}^{2}\right\} 
        \left\{ \left[u_{x}-u_{x;c;O}\right]\cos\left(\theta_{O}\right)
        -\left[u_{x}-u_{x;c;O}\right]\sin\left(\theta_{O}\right)\right\} ^{2}\\
        &\quad\quad-\left\{ \left[u_{x}-
        u_{x;c;O}\right]\sin\left(\theta_{O}\right)+\left[u_{x}-
        u_{x;c;O}\right]\cos\left(\theta_{O}\right)\right\} ^{2}.
        :label: uniform_ellipse_support_arg__1

    Parameters
    ----------
    center : `array_like` (`float`, shape=(``2``,)), optional
        The center of the ellipse, :math:`\left(u_{x;c;O},u_{x;c;O}\right)`.
    semi_major_axis : `float`, optional
        The semi-major axis of the ellipse, :math:`a_{O}`. Must be positive.
    eccentricity : `float`, optional
        The eccentricity of the ellipse, :math:`e_{O}`. Must be a nonnegative
        number less than or equal to unity.
    rotation_angle : `float`, optional
        The rotation angle of the ellipse, :math:`\theta_{O}`.
    intra_ellipse_val : `float`, optional
        The value of the intensity pattern inside the ellipse.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    ctor_param_names = ("center",
                        "semi_major_axis",
                        "eccentricity",
                        "rotation_angle",
                        "intra_ellipse_val")
    kwargs = {"namespace_as_dict": globals(),
              "ctor_param_names": ctor_param_names}

    _validation_and_conversion_funcs_ = \
        fancytypes.return_validation_and_conversion_funcs(**kwargs)
    _pre_serialization_funcs_ = \
        fancytypes.return_pre_serialization_funcs(**kwargs)
    _de_pre_serialization_funcs_ = \
        fancytypes.return_de_pre_serialization_funcs(**kwargs)

    del ctor_param_names, kwargs

    

    def __init__(self,
                 center=\
                 _default_center,
                 semi_major_axis=\
                 _default_semi_major_axis,
                 eccentricity=\
                 _default_eccentricity,
                 rotation_angle=\
                 _default_rotation_angle,
                 intra_ellipse_val=\
                 _default_intra_ellipse_val,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        _BaseShape.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        return None



    def update(self, new_core_attr_subset_candidate):
        super().update(new_core_attr_subset_candidate)
        self.execute_post_core_attrs_update_actions()

        return None



    def _eval(self, u_x, u_y):
        A_O = self._intra_ellipse_val
        support_arg = self._calc_support_arg(u_x, u_y)
        one = torch.tensor(1.0, device=support_arg.device)
        result = A_O * torch.heaviside(support_arg, one)

        return result



    def _calc_support_arg(self, u_x, u_y):
        u_x_c_O, u_y_c_O = self._center
        a_O = self._semi_major_axis
        e_O = self._eccentricity
        theta_O = self._rotation_angle

        delta_u_x_O = u_x-u_x_c_O
        delta_u_y_O = u_y-u_y_c_O

        e_O_sq = e_O*e_O
        a_O_sq = a_O*a_O
        b_O_sq = (1-e_O_sq)*a_O_sq

        cos_theta_O = torch.cos(theta_O)
        sin_theta_O = torch.sin(theta_O)

        delta_u_x_O_prime = (delta_u_x_O*cos_theta_O
                             - delta_u_y_O*sin_theta_O)
        delta_u_x_O_prime_sq = delta_u_x_O_prime*delta_u_x_O_prime

        delta_u_y_O_prime = (delta_u_x_O*sin_theta_O
                             + delta_u_y_O*cos_theta_O)
        delta_u_y_O_prime_sq = delta_u_y_O_prime*delta_u_y_O_prime

        support_arg = (b_O_sq
                       - (b_O_sq/a_O_sq)*delta_u_x_O_prime_sq
                       - delta_u_y_O_prime_sq)

        return support_arg



def _check_and_convert_val_at_center(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    val_at_center = czekitout.convert.to_float(**kwargs)

    return val_at_center



def _pre_serialize_val_at_center(val_at_center):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_val_at_center(serializable_rep):
    val_at_center = serializable_rep

    return val_at_center



def _check_and_convert_width_factors(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    width_factors = czekitout.convert.to_pair_of_positive_floats(**kwargs)

    return width_factors



def _pre_serialize_width_factors(width_factors):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_width_factors(serializable_rep):
    width_factors = serializable_rep

    return width_factors



def _check_and_convert_asymmetry_factors(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    asymmetry_factors = czekitout.convert.to_pair_of_positive_floats(**kwargs)

    return asymmetry_factors



def _pre_serialize_asymmetry_factors(asymmetry_factors):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_asymmetry_factors(serializable_rep):
    asymmetry_factors = serializable_rep

    return asymmetry_factors



def _check_and_convert_reflection_factors(params):
    current_func_name = inspect.stack()[0][3]
    obj_name = current_func_name[19:]
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    reflection_factors = czekitout.convert.to_pair_of_bools(**kwargs)

    return reflection_factors



def _pre_serialize_reflection_factors(reflection_factors):
    obj_to_pre_serialize = random.choice(list(locals().values()))
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_reflection_factors(serializable_rep):
    reflection_factors = serializable_rep

    return reflection_factors



class AsymmetricGaussianPeak(_BaseShape):
    r"""The intensity pattern of an asymmetric Gaussian peak.

    Let :math:`\left(u_{x;c;\text{AG}},u_{y;c;\text{AG}}\right)`,
    :math:`\left(\sigma_{1;\text{AG}},\sigma_{2;\text{AG}}\right)`,
    :math:`\left(\eta_{1;\text{AG}},\eta_{2;\text{AG}}\right)`,
    :math:`\left(\nu_{1;\text{AG}},\nu_{2;\text{AG}}\right)`, and
    :math:`\theta_{\text{AG}}` be the center, the width factors, the asymmetry
    factors, the reflection factors, and the rotation angle of the asymmetric
    Gaussian peak respectively. Furthermore, let :math:`A_{\text{AG}}` be the
    value of the intensity pattern at the center of the peak. The intensity
    pattern is given by:

    .. math ::
        \mathcal{I}_{\text{AG}}\left(u_{x},u_{y}\right)=
        A_{\text{AG}}\prod_{\alpha=1}^{2}\left\{ \exp\left[
        -\frac{1}{2}\left\{ \frac{z_{\alpha;\text{AG}}\left(u_{x},
        u_{y}\right)}{\sigma_{\alpha;\text{AG}}}\right\} ^{2}\right]\right\} ,
        :label: intensity_pattern_of_asymmetric_gaussian_peak__1

    with :math:`\Theta\left(\cdots\right)` being the Heaviside step function,
    and

    .. math ::
        z_{\alpha;\text{AG}}\left(u_{x},u_{y}\right)=
        \frac{\tilde{z}_{\alpha;\text{AG}}\left(u_{x},
        u_{y}\right)}{\eta_{\alpha;\text{AG}}^{\left\{
        1-\nu_{\alpha;\text{AG}}\right\} -\left\{
        1-2\nu_{\alpha;\text{AG}}\right\}
        \Theta\left(\tilde{z}_{\alpha;\text{AG}}\left(u_{x},
        u_{y}\right)\right)}},
        :label: z_alpha_AG__1

    Parameters
    ----------
    center : `array_like` (`float`, shape=(``2``,)), optional
        The center of the ellipse, :math:`\left(u_{x;c;O},u_{x;c;O}\right)`.
    semi_major_axis : `float`, optional
        The semi-major axis of the ellipse, :math:`a_{O}`. Must be positive.
    eccentricity : `float`, optional
        The eccentricity of the ellipse, :math:`e_{O}`. Must be a nonnegative
        number less than or equal to unity.
    rotation_angle : `float`, optional
        The rotation angle of the ellipse, :math:`\theta_{O}`.
    intra_ellipse_val : `float`, optional
        The value of the intensity pattern inside the ellipse.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.

    """
    ctor_param_names = ("center",
                        "semi_major_axis",
                        "eccentricity",
                        "rotation_angle",
                        "intra_ellipse_val")
    kwargs = {"namespace_as_dict": globals(),
              "ctor_param_names": ctor_param_names}

    _validation_and_conversion_funcs_ = \
        fancytypes.return_validation_and_conversion_funcs(**kwargs)
    _pre_serialization_funcs_ = \
        fancytypes.return_pre_serialization_funcs(**kwargs)
    _de_pre_serialization_funcs_ = \
        fancytypes.return_de_pre_serialization_funcs(**kwargs)

    del ctor_param_names, kwargs

    

    def __init__(self,
                 center=\
                 _default_center,
                 semi_major_axis=\
                 _default_semi_major_axis,
                 eccentricity=\
                 _default_eccentricity,
                 rotation_angle=\
                 _default_rotation_angle,
                 intra_ellipse_val=\
                 _default_intra_ellipse_val,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        _BaseShape.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        return None



    def update(self, new_core_attr_subset_candidate):
        super().update(new_core_attr_subset_candidate)
        self.execute_post_core_attrs_update_actions()

        return None



    def _eval(self, u_x, u_y):
        A_O = self._intra_ellipse_val
        support_arg = self._calc_support_arg(u_x, u_y)
        one = torch.tensor(1.0, device=support_arg.device)
        result = A_O * torch.heaviside(support_arg, one)

        return result



    def _calc_support_arg(self, u_x, u_y):
        u_x_c_O, u_y_c_O = self._center
        a_O = self._semi_major_axis
        e_O = self._eccentricity
        theta_O = self._rotation_angle

        delta_u_x_O = u_x-u_x_c_O
        delta_u_y_O = u_y-u_y_c_O

        e_O_sq = e_O*e_O
        a_O_sq = a_O*a_O
        b_O_sq = (1-e_O_sq)*a_O_sq

        cos_theta_O = torch.cos(theta_O)
        sin_theta_O = torch.sin(theta_O)

        delta_u_x_O_prime = (delta_u_x_O*cos_theta_O
                             - delta_u_y_O*sin_theta_O)
        delta_u_x_O_prime_sq = delta_u_x_O_prime*delta_u_x_O_prime

        delta_u_y_O_prime = (delta_u_x_O*sin_theta_O
                             + delta_u_y_O*cos_theta_O)
        delta_u_y_O_prime_sq = delta_u_y_O_prime*delta_u_y_O_prime

        support_arg = (b_O_sq
                       - (b_O_sq/a_O_sq)*delta_u_x_O_prime_sq
                       - delta_u_y_O_prime_sq)

        return support_arg



# def _check_and_convert_width(params):
#     current_func_name = inspect.stack()[0][3]
#     obj_name = current_func_name[19:]
#     kwargs = {"obj": params[obj_name], "obj_name": obj_name}
#     width = czekitout.convert.to_positive_float(**kwargs)

#     return width



# def _pre_serialize_width(width):
#     obj_to_pre_serialize = random.choice(list(locals().values()))
#     serializable_rep = obj_to_pre_serialize
    
#     return serializable_rep



# def _de_pre_serialize_width(serializable_rep):
#     width = serializable_rep

#     return width



# def _check_and_convert_val_at_center(params):
#     current_func_name = inspect.stack()[0][3]
#     obj_name = current_func_name[19:]
#     kwargs = {"obj": params[obj_name], "obj_name": obj_name}
#     val_at_center = czekitout.convert.to_float(**kwargs)

#     return val_at_center



# def _pre_serialize_val_at_center(val_at_center):
#     obj_to_pre_serialize = random.choice(list(locals().values()))
#     serializable_rep = obj_to_pre_serialize
    
#     return serializable_rep



# def _de_pre_serialize_val_at_center(serializable_rep):
#     val_at_center = serializable_rep

#     return val_at_center



# _default_width = 0.05
# _default_val_at_center = 1



# class _GaussianPeak(_BaseShape):
#     r"""Insert description here.

#     Parameters
#     ----------
#     center : `array_like` (`float`, shape=(``2``,)), optional
#         Insert description here.
#     width : `float`, optional
#         Insert description here.
#     val_at_center : `float`, optional
#         Insert description here.

#     Attributes
#     ----------
#     core_attrs : `dict`, read-only
#         A `dict` representation of the core attributes: each `dict` key is a
#         `str` representing the name of a core attribute, and the corresponding
#         `dict` value is the object to which said core attribute is set. The core
#         attributes are the same as the construction parameters, except that 
#         their values might have been updated since construction.

#     """
#     _validation_and_conversion_funcs = \
#         {"center": _check_and_convert_center,
#          "width": _check_and_convert_width,
#          "val_at_center": _check_and_convert_val_at_center}

#     _pre_serialization_funcs = \
#         {"center": _pre_serialize_center,
#          "width": _pre_serialize_width,
#          "val_at_center": _pre_serialize_val_at_center}

#     _de_pre_serialization_funcs = \
#         {"center": _de_pre_serialize_center,
#          "width": _de_pre_serialize_width,
#          "val_at_center": _de_pre_serialize_val_at_center}

#     def __init__(self,
#                  center=_default_center,
#                  width=_default_width,
#                  val_at_center=_default_val_at_center):
#         ctor_params = {key: val
#                        for key, val in locals().items()
#                        if (key not in ("self", "__class__"))}
#         _BaseShape.__init__(self, ctor_params)

#         self._post_base_update()

#         return None



#     def _post_base_update(self):
#         self._center = self._core_attrs["center"]
#         self._width = self._core_attrs["width"]
#         self._val_at_center = self._core_attrs["val_at_center"]

#         return None



#     def update(self, core_attr_subset):
#         super().update(core_attr_subset)
#         self._post_base_update()

#         return None



#     def _eval(self, x, y):
#         result = self._eval_1(x, y)

#         return result



#     def _eval_1(self, x, y):
#         center = self._center
        
#         r = _r(x, y, center)
        
#         result = self._eval_2(r)

#         return result



#     def _eval_2(self, r):
#         sigma = self._width
#         A = self._val_at_center
#         r_over_sigma = r/sigma
        
#         result = A * torch.exp(-0.5 * r_over_sigma * r_over_sigma)

#         return result



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
    ("The objects ``{}`` and ``{}`` must be real-valued matrices of the same "
     "shape.")

_check_and_convert_real_torch_matrix_err_msg_1 = \
    ("The object ``{}`` must be a real-valued matrix.")

_check_and_convert_eccentricity_err_msg_1 = \
    ("The object ``eccentricity`` must be a nonnegative number less than or "
     "equal to unity.")

_check_and_convert_functional_form_err_msg_1 = \
    ("The object ``functiona_form`` must be set to ``'gaussian'``, "
     "``'lorentzian'``, or ``'exponential'``.")

_check_and_convert_intra_disk_shapes_err_msg_1 = \
    ("The object ``intra_disk_shapes`` must be a sequence of objects of any "
     "of the following types: (`fakecbed.shapes.UniformDisk`, "
     "`fakecbed.shapes.Peak`, `fakecbed.shapes.Band`, "
     "`fakecbed.shapes.PlaneWave`).")
