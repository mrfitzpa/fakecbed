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
:mod:`fakecbed.discretized` module. An image of a fake CBED pattern is formed by
specifying a series of parameters, with the most important parameters being: the
set of intensity patterns of undistorted shapes that determine the undistorted
noiseless non-blurred uncorrupted fake CBED pattern; and a distortion model
which transforms the undistorted noiseless non-blurred uncorrupted fake CBED
pattern into a distorted noiseless non-blurred uncorrupted fake CBED
pattern. The remaining parameters determine whether additional images effects
are applied, like e.g. shot noise or blur effects. Note that in the case of the
aforementioned shapes, we expand the notion of intensity patterns to mean a 2D
real-valued function, i.e. it can be negative. To be clear, we do not apply this
generalized notion of intensity patterns to the fake CBED patterns: in such
cases intensity patterns mean 2D real-valued functions that are strictly
nonnegative.

Let :math:`u_{x}` and :math:`u_{y}` be the fractional horizontal and vertical
coordinates, respectively, of a point in an undistorted image, where
:math:`\left(u_{x},u_{y}\right)=\left(0,0\right)` is the bottom left corner of
the image. Secondly, let :math:`q_{x}` and :math:`q_{y}` be the fractional
horizontal and vertical coordinates, respectively, of a point in a distorted
image, where :math:`\left(q_{x},q_{y}\right)=\left(0,0\right)` is the bottom
left corner of the image. When users specify a distortion model, represented by
an :obj:`distoptica.DistortionModel` object, they also specify a coordinate
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

# For performing deep copies.
import copy



# For general array handling.
import numpy as np
import torch

# For calculating factorials.
import math

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
__all__ = ["BaseShape",
           "Circle",
           "Ellipse",
           "Peak",
           "Band",
           "PlaneWave",
           "Arc",
           "GenericBlob",
           "Orbital",
           "Lune",
           "NonuniformBoundedShape"]



def _check_and_convert_cartesian_coords(params):
    obj_name = "cartesian_coords"
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

    current_func_name = "_check_and_convert_cartesian_coords"

    if u_x.shape != u_y.shape:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        err_msg = unformatted_err_msg.format("u_x", "u_y")
        raise ValueError(err_msg)

    cartesian_coords = (u_x, u_y)

    return cartesian_coords



def _check_and_convert_real_torch_matrix(params):
    obj_name = "real_torch_matrix"
    obj = params[obj_name]
    
    name_of_alias_of_real_torch_matrix = \
        params["name_of_alias_of_real_torch_matrix"]

    current_func_name = "_check_and_convert_real_torch_matrix"

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
    obj_name = "skip_validation_and_conversion"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    skip_validation_and_conversion = czekitout.convert.to_bool(**kwargs)

    return skip_validation_and_conversion



_default_u_x = ((0.5,),)
_default_u_y = _default_u_x
_default_device = None
_default_skip_validation_and_conversion = False



class BaseShape(fancytypes.PreSerializableAndUpdatable):
    r"""The intensity pattern of an undistorted geometric shape.

    See the summary documentation of the module :mod:`fakecbed.shapes` for
    additional context.

    One cannot construct an instance of the class
    :class:`fakecbed.shapes.BaseShape`, only subclasses of itself defined in
    :mod:`fakecbed` library.

    Parameters
    ----------
    ctor_params : `dict`
        The construction parameters of the subclass.

    """
    def __init__(self, ctor_params):
        if type(self) is BaseShape:
            self._eval(u_x=None, u_y=None)
        else:
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
        r"""Evaluate the intensity pattern of the undistorted shape.

        Let :math:`u_{x}` and :math:`u_{y}` be the fractional horizontal and
        vertical coordinates, respectively, of a point in an undistorted image.
        We adopt the convention where the fractional coordinate pair
        :math:`\left(u_{x},u_{y}\right)=\left(0,0\right)` is the bottom left
        corner of an image.

        Parameters
        ----------
        u_x : `torch.Tensor` (`float`, ndim=2), optional
            The fractional horizontal coordinates of the positions at which to
            evaluate the intensity pattern of the undistorted shape. 
        u_y : `torch.Tensor` (`float`, shape=``u_x.shape``), optional
            The fractional vertical coordinates of the positions at which to
            evaluate the intensity pattern of the undistorted shape.
        device : `torch.device` | `None`, optional
            This parameter specifies the device to be used to perform
            computationally intensive calls to PyTorch functions. If ``device``
            is of the type :class:`torch.device`, then ``device`` represents the
            device to be used. If ``device`` is set to ``None`` and a GPU device
            is available, then a GPU device is to be used. Otherwise, the CPU is
            used.
        skip_validation_and_conversion : `bool`, optional
            If ``skip_validation_and_conversion`` is set to ``False``, then
            validations and conversions are performed on the above parameters.

            Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
            no validations and conversions are performed on the above
            parameters. This option is desired primarily when the user wants to
            avoid potentially expensive validation and/or conversion operations.

        Returns
        -------
        result : `torch.Tensor` (`float`, shape=``u_x.shape``)
            The values of the intensity pattern at the positions specified by
            ``u_x`` and ``u_y``. For every pair of nonnegative integers ``(i,
            j)`` that does not raise an ``IndexError`` exception upon calling
            ``result[i, j]``, ``result[i, j]`` is the value of the intensity
            pattern at the position ``(u_x[i, j], u_y[i, j])`` of the 
            undistorted shape.

        """
        params = locals()

        func_alias = _check_and_convert_skip_validation_and_conversion
        skip_validation_and_conversion = func_alias(params)

        if (skip_validation_and_conversion == False):
            params = {"cartesian_coords": (u_x, u_y), "device": device}
            device = _check_and_convert_device(params)
            u_x, u_y = _check_and_convert_cartesian_coords(params)
        
        result = self._eval(u_x, u_y)

        return result



    def _eval(self, u_x, u_y):
        raise NotImplementedError(_base_shape_err_msg_1)



def _check_and_convert_center(params):
    obj_name = "center"

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
    obj_to_pre_serialize = center
    obj_name = "center"

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
    obj_name = "center"

    cls_alias = \
        distoptica.CoordTransformParams
    de_pre_serialization_funcs = \
        cls_alias.get_de_pre_serialization_funcs()
    de_pre_serialization_func = \
        de_pre_serialization_funcs[obj_name]
    center = \
        de_pre_serialization_func(serializable_rep)

    return center



def _check_and_convert_radius(params):
    obj_name = "radius"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    radius = czekitout.convert.to_positive_float(**kwargs)

    return radius



def _pre_serialize_radius(radius):
    obj_to_pre_serialize = radius
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_radius(serializable_rep):
    radius = serializable_rep

    return radius



def _check_and_convert_intra_shape_val(params):
    obj_name = "intra_shape_val"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    intra_shape_val = czekitout.convert.to_float(**kwargs)

    return intra_shape_val



def _pre_serialize_intra_shape_val(intra_shape_val):
    obj_to_pre_serialize = intra_shape_val
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_intra_shape_val(serializable_rep):
    intra_shape_val = serializable_rep

    return intra_shape_val



_default_center = (0.5, 0.5)
_default_radius = 0.05
_default_intra_shape_val = 1



class Circle(BaseShape):
    r"""The intensity pattern of a circle.

    Let :math:`\left(u_{x;c;\text{C}},u_{y;c;\text{C}}\right)`, and
    :math:`R_{\text{C}}` be the center, and the radius of the circle
    respectively. Furthermore, let :math:`A_{\text{C}}` be the value of the
    intensity pattern inside the circle. The undistorted intensity pattern of
    the circle is given by:

    .. math ::
        \mathcal{I}_{\text{C}}\left(u_{x},u_{y}\right)=
        A_{\text{C}}\Theta\left(R_{\text{C}}
        -\sqrt{\left(u_{x}-u_{x;c;\text{C}}\right)^{2}
        +\left(u_{y}-u_{y;c;\text{C}}\right)^{2}}\right),
        :label: intensity_pattern_of_circle__1

    where :math:`u_{x}` and :math:`u_{y}` are fractional horizontal and vertical
    coordinates of the undistorted intensity pattern of the circle respectively,
    and :math:`\Theta\left(\cdots\right)` is the Heaviside step function.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(``2``,)), optional
        The center of the circle, :math:`\left(u_{x;c;\text{C}},
        u_{y;c;\text{C}}\right)`.
    radius : `float`, optional
        The radius of the circle, :math:`R_{\text{C}}`. Must be positive.
    intra_shape_val : `float`, optional
        The value of the intensity pattern inside the circle.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("center",
                        "radius",
                        "intra_shape_val")
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
                 radius=\
                 _default_radius,
                 intra_shape_val=\
                 _default_intra_shape_val,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        BaseShape.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        return None



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    def _eval(self, u_x, u_y):
        u_x_c, u_y_c = self._center
        R = self._radius
        A = self._intra_shape_val
        
        delta_u_x = u_x-u_x_c
        delta_u_y = u_y-u_y_c

        u_r = torch.sqrt(delta_u_x*delta_u_x + delta_u_y*delta_u_y)
        
        result = A * (u_r <= R)

        return result



def _check_and_convert_semi_major_axis(params):
    obj_name = "semi_major_axis"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    semi_major_axis = czekitout.convert.to_positive_float(**kwargs)

    return semi_major_axis



def _pre_serialize_semi_major_axis(semi_major_axis):
    obj_to_pre_serialize = semi_major_axis
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_semi_major_axis(serializable_rep):
    semi_major_axis = serializable_rep

    return semi_major_axis



def _check_and_convert_eccentricity(params):
    obj_name = "eccentricity"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    eccentricity = czekitout.convert.to_nonnegative_float(**kwargs)

    current_func_name = "_check_and_convert_eccentricity"
    
    if eccentricity > 1:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise ValueError(err_msg)

    return eccentricity



def _pre_serialize_eccentricity(eccentricity):
    obj_to_pre_serialize = eccentricity
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_eccentricity(serializable_rep):
    eccentricity = serializable_rep

    return eccentricity



def _check_and_convert_rotation_angle(params):
    obj_name = "rotation_angle"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    rotation_angle = czekitout.convert.to_float(**kwargs) % (2*np.pi)

    return rotation_angle



def _pre_serialize_rotation_angle(rotation_angle):
    obj_to_pre_serialize = rotation_angle
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_rotation_angle(serializable_rep):
    rotation_angle = serializable_rep

    return rotation_angle



_default_semi_major_axis = _default_radius
_default_eccentricity = 0
_default_rotation_angle = 0



class Ellipse(BaseShape):
    r"""The intensity pattern of a ellipse.

    Let :math:`\left(u_{x;c;\text{E}},u_{y;c;\text{E}}\right)`,
    :math:`a_{\text{E}}`, :math:`e_{\text{E}}`, and :math:`\theta_{\text{E}}` be
    the center, the semi-major axis, the eccentricity, and the rotation angle of
    the ellipse respectively. Furthermore, let :math:`A_{\text{E}}` be the value
    of the intensity pattern inside the ellipse. The undistorted intensity
    pattern of the ellipse is given by:

    .. math ::
        \mathcal{I}_{\text{E}}\left(u_{x},u_{y}\right)=
        A_{\text{E}}\Theta\left(\Theta_{\arg;\text{E}}\left(u_{x},
        u_{y}\right)\right),
        :label: intensity_pattern_of_ellipse__1

    where :math:`u_{x}` and :math:`u_{y}` are fractional horizontal and vertical
    coordinates of the undistorted intensity pattern of the ellipse
    respectively, :math:`\Theta\left(\cdots\right)` is the Heaviside step
    function, and

    .. math ::
        &\Theta_{\arg;\text{E}}\left(u_{x},u_{y}\right)\\&\quad=
        \left\{ 1-e_{\text{E}}^{2}\right\} a_{\text{E}}^{2}\\
        &\quad\quad-\left\{ 1-e_{\text{E}}^{2}\right\} 
        \left\{ \left[u_{x}-u_{x;c;\text{E}}\right]
        \cos\left(\theta_{\text{E}}\right)
        -\left[u_{y}-u_{y;c;\text{E}}\right]
        \sin\left(\theta_{\text{E}}\right)\right\} ^{2}\\
        &\quad\quad-\left\{ \left[u_{x}-
        u_{x;c;\text{E}}\right]\sin\left(\theta_{\text{E}}\right)+\left[u_{y}-
        u_{y;c;\text{E}}\right]\cos\left(\theta_{\text{E}}\right)\right\}^{2}.
        :label: ellipse_support_arg__1

    Parameters
    ----------
    center : `array_like` (`float`, shape=(``2``,)), optional
        The center of the ellipse, :math:`\left(u_{x;c;\text{E}},
        u_{y;c;\text{E}}\right)`.
    semi_major_axis : `float`, optional
        The semi-major axis of the ellipse, :math:`a_{\text{E}}`. Must be
        positive.
    eccentricity : `float`, optional
        The eccentricity of the ellipse, :math:`e_{\text{E}}`. Must be a
        nonnegative number less than or equal to unity.
    rotation_angle : `float`, optional
        The rotation angle of the ellipse, :math:`\theta_{\text{E}}`.
    intra_shape_val : `float`, optional
        The value of the intensity pattern inside the ellipse.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("center",
                        "semi_major_axis",
                        "eccentricity",
                        "rotation_angle",
                        "intra_shape_val")
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
                 intra_shape_val=\
                 _default_intra_shape_val,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        BaseShape.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        return None



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    def _eval(self, u_x, u_y):
        A = self._intra_shape_val
        support_arg = self._calc_support_arg(u_x, u_y)
        one = torch.tensor(1.0, device=support_arg.device)
        result = A * torch.heaviside(support_arg, one)

        return result



    def _calc_support_arg(self, u_x, u_y):
        u_x_c, u_y_c = self._center
        a = self._semi_major_axis
        e = self._eccentricity
        theta = torch.tensor(self._rotation_angle, dtype=u_x.dtype)

        delta_u_x = u_x-u_x_c
        delta_u_y = u_y-u_y_c

        e_sq = e*e
        a_sq = a*a
        b_sq = (1-e_sq)*a_sq

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        delta_u_x_prime = delta_u_x*cos_theta - delta_u_y*sin_theta
        delta_u_x_prime_sq = delta_u_x_prime*delta_u_x_prime

        delta_u_y_prime = delta_u_x*sin_theta + delta_u_y*cos_theta
        delta_u_y_prime_sq = delta_u_y_prime*delta_u_y_prime

        support_arg = (b_sq
                       - (b_sq/a_sq)*delta_u_x_prime_sq
                       - delta_u_y_prime_sq)

        return support_arg



def _check_and_convert_widths(params):
    obj_name = "widths"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    widths = czekitout.convert.to_quadruplet_of_positive_floats(**kwargs)

    return widths



def _pre_serialize_widths(widths):
    obj_to_pre_serialize = widths
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_widths(serializable_rep):
    widths = serializable_rep

    return widths



def _check_and_convert_val_at_center(params):
    obj_name = "val_at_center"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    val_at_center = czekitout.convert.to_float(**kwargs)

    return val_at_center



def _pre_serialize_val_at_center(val_at_center):
    obj_to_pre_serialize = val_at_center
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_val_at_center(serializable_rep):
    val_at_center = serializable_rep

    return val_at_center



def _check_and_convert_functional_form(params):
    obj_name = "functional_form"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    functional_form = czekitout.convert.to_str_from_str_like(**kwargs)

    kwargs["obj"] = functional_form
    kwargs["accepted_strings"] = ("asymmetric_gaussian",
                                  "asymmetric_exponential",
                                  "asymmetric_lorentzian")
    czekitout.check.if_one_of_any_accepted_strings(**kwargs)

    return functional_form



def _pre_serialize_functional_form(functional_form):
    obj_to_pre_serialize = functional_form
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_functional_form(serializable_rep):
    functional_form = serializable_rep

    return functional_form



_default_widths = 4*(0.05,)
_default_val_at_center = 1
_default_functional_form = "asymmetric_gaussian"



class Peak(BaseShape):
    r"""The intensity pattern of a peak.

    Let :math:`\left(u_{x;c;\text{P}},u_{y;c;\text{P}}\right)`,
    :math:`\left(W_{1;1;\text{P}},W_{1;2;\text{P}},
    W_{2;1;\text{P}},W_{2;2;\text{P}}\right)`, and :math:`\theta_{\text{P}}` be
    the center, the widths factors, and the rotation angle of the peak
    respectively. Furthermore, let :math:`A_{\text{P}}` be the value of the
    intensity pattern at the center of the peak. The undistorted intensity
    pattern of the peak is given by:

    .. math ::
        \mathcal{I}_{\text{P}}\left(u_{x},u_{y}\right)=
        A_{\text{P}}F_{\beta;\text{P}}\left(
        \sqrt{\sum_{\alpha=1}^{2}\left\{ \frac{
        z_{\alpha;\text{P}}\left(u_{x},u_{y}\right)}{
        W_{\alpha;\text{P}}\left(u_{x},u_{y}\right)}\right\}^{2}}\right),
        :label: intensity_pattern_of_peak__1

    where :math:`u_{x}` and :math:`u_{y}` are fractional horizontal and vertical
    coordinates of the undistorted intensity pattern of the peak respectively,

    .. math ::
        F_{\beta;\text{P}}\left(\omega\right)=\begin{cases}
        e^{-\frac{1}{2}\omega^{2}}, & \text{if }\beta=\text{A.G.},\\
        e^{-\omega}, & \text{if }\beta=\text{A.E.},\\
        \left(1+\omega^{2}\right)^{-\frac{3}{2}}, & \text{if }\beta=\text{A.L.},
        \end{cases}
        :label: functional_form_of_peak__1

    with A.G., A.E., and A.L. being abbreviations of “asymmetric Gaussian”,
    “asymmetric exponential”, and “asymmetric Lorentzian” respectively, and
    :math:`\beta` specifying the functional form of the intensity pattern;

    .. math ::
        z_{\alpha=1;\text{P}}\left(u_{x},u_{y}\right)=
        \left(u_{x}-u_{x;c;\text{P}}\right)
        \cos\left(\theta_{\text{P}}\right)
        -\left(u_{y}
        -u_{y;c;\text{P}}\right)\sin\left(\theta_{\text{P}}\right);
        :label: z_alpha_peak__1

    .. math ::
        z_{\alpha=2;\text{P}}\left(u_{x},u_{y}\right)=
        \left(u_{x}-u_{x;c;\text{P}}\right)
        \sin\left(\theta_{\text{P}}\right)
        +\left(u_{y}
        -u_{y;c;\text{P}}\right)\cos\left(\theta_{\text{P}}\right);
        :label: z_alpha_peak__2

    and

    .. math ::
        W_{\alpha;\text{P}}\left(u_{x},u_{y}\right)=
        \sum_{\nu=1}^{2}W_{\alpha;\nu;\text{P}}\left[\left\{\nu-1\right\} 
        +\left\{ -1\right\}^{\nu+1}
        \Theta\left(z_{\alpha;\text{P}}\left(u_{x},
        u_{y}\right)\right)\right],
        :label: W_alpha_peak__1

    with :math:`\Theta\left(\cdots\right)` being the Heaviside step function.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(``2``,)), optional
        The center of the peak,
        :math:`\left(u_{x;c;\text{P}},u_{y;c;\text{P}}\right)`.
    widths : `array_like` (`float`, shape=(``4``,)), optional
        The width factors of the peak,
        :math:`\left(W_{1;1;\text{P}},W_{1;2;\text{P}},
        W_{2;1;\text{P}},W_{2;2;\text{P}}\right)`. Must be a quadruplet of
        positive numbers.
    rotation_angle : `float`, optional
        The rotation angle of the peak, :math:`\theta_{\text{P}}`.
    val_at_center : `float`, optional
        The value of the intensity pattern at the center of the peak,
        :math:`A_{\text{P}}`.
    functional_form : ``"asymmetric_gaussian"`` | ``"asymmetric_exponential"`` | ``"asymmetric_lorentzian"``, optional
        The functional form of the peak. If
        ``functional_form==asymmetric_gaussian``, then :math:`\beta`, which
        appears in Eq. :eq:`functional_form_of_peak__1`, is set to "A.G."; else
        if ``functional_form==asymmetric_exponential``, then :math:`\beta` is
        set to "A.E."; else :math:`\beta` is set to "A.L.".
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("center",
                        "widths",
                        "rotation_angle",
                        "val_at_center",
                        "functional_form")
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
                 widths=\
                 _default_widths,
                 rotation_angle=\
                 _default_rotation_angle,
                 val_at_center=\
                 _default_val_at_center,
                 functional_form=\
                 _default_functional_form,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        BaseShape.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        theta = torch.tensor(self._rotation_angle)
        self._cos_theta = torch.cos(theta)
        self._sin_theta = torch.sin(theta)

        functional_form = self._functional_form
        if functional_form == "asymmetric_gaussian":
            self._eval = self._eval_asymmetric_gaussian
        elif functional_form == "asymmetric_exponential":
            self._eval = self._eval_asymmetric_exponential
        else:
            self._eval = self._eval_asymmetric_lorentzian

        return None



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    def _eval_asymmetric_gaussian(self, u_x, u_y):
        u_x_c, u_y_c = self._center
        delta_u_x_c = u_x-u_x_c
        delta_u_y_c = u_y-u_y_c

        cos_theta = self._cos_theta
        sin_theta = self._sin_theta

        A = self._val_at_center
        W_1_1, W_1_2, W_2_1, W_2_2 = self._widths

        z_1 = delta_u_x_c*cos_theta - delta_u_y_c*sin_theta
        mask_1 = (z_1 >= 0)
        W_1 = W_1_1*mask_1 + W_1_2*(~mask_1)
        z_1_over_W_1 = z_1/W_1

        z_2 = delta_u_x_c*sin_theta + delta_u_y_c*cos_theta
        mask_2 = (z_2 >= 0)
        W_2 = W_2_1*mask_2 + W_2_2*(~mask_2)
        z_2_over_W_2 = z_2/W_2

        result = A*torch.exp(-0.5*(z_1_over_W_1*z_1_over_W_1
                                   + z_2_over_W_2*z_2_over_W_2))

        return result



    def _eval_asymmetric_exponential(self, u_x, u_y):
        u_x_c, u_y_c = self._center
        delta_u_x_c = u_x-u_x_c
        delta_u_y_c = u_y-u_y_c

        cos_theta = self._cos_theta
        sin_theta = self._sin_theta

        A = self._val_at_center
        w_1_1, w_1_2, w_2_1, w_2_2 = self._widths

        z_1 = delta_u_x_c*cos_theta - delta_u_y_c*sin_theta
        mask_1 = (z_1 >= 0)
        w_1 = w_1_1*mask_1 + w_1_2*(~mask_1)
        z_1_over_w_1 = z_1/w_1

        z_2 = delta_u_x_c*sin_theta + delta_u_y_c*cos_theta
        mask_2 = (z_2 >= 0)
        w_2 = w_2_1*mask_2 + w_2_2*(~mask_2)
        z_2_over_w_2 = z_2/w_2

        result = A*torch.exp(-torch.sqrt(z_1_over_w_1*z_1_over_w_1
                                         + z_2_over_w_2*z_2_over_w_2))

        return result



    def _eval_asymmetric_lorentzian(self, u_x, u_y):
        u_x_c, u_y_c = self._center
        delta_u_x_c = u_x-u_x_c
        delta_u_y_c = u_y-u_y_c

        cos_theta = self._cos_theta
        sin_theta = self._sin_theta

        A = self._val_at_center
        w_1_1, w_1_2, w_2_1, w_2_2 = self._widths

        z_1 = delta_u_x_c*cos_theta - delta_u_y_c*sin_theta
        mask_1 = (z_1 >= 0)
        w_1 = w_1_1*mask_1 + w_1_2*(~mask_1)
        z_1_over_w_1 = z_1/w_1

        z_2 = delta_u_x_c*sin_theta + delta_u_y_c*cos_theta
        mask_2 = (z_2 >= 0)
        w_2 = w_2_1*mask_2 + w_2_2*(~mask_2)
        z_2_over_w_2 = z_2/w_2

        denom_factor = torch.sqrt(1
                                  + z_1_over_w_1*z_1_over_w_1
                                  + z_2_over_w_2*z_2_over_w_2)

        result = A / denom_factor / denom_factor / denom_factor

        return result



def _check_and_convert_end_pt_1(params):
    obj_name = "end_pt_1"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    end_pt_1 = czekitout.convert.to_pair_of_floats(**kwargs)

    return end_pt_1



def _pre_serialize_end_pt_1(end_pt_1):
    obj_to_pre_serialize = end_pt_1
    serializable_rep = obj_to_pre_serialize
    
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
    obj_to_pre_serialize = end_pt_2
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_end_pt_2(serializable_rep):
    end_pt_2 = serializable_rep

    return end_pt_2



def _check_and_convert_width(params):
    obj_name = "width"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    width = czekitout.convert.to_positive_float(**kwargs)

    return width



def _pre_serialize_width(width):
    obj_to_pre_serialize = width
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_width(serializable_rep):
    width = serializable_rep

    return width



_default_end_pt_1 = (0, 0.5)
_default_end_pt_2 = (1, 0.5)
_default_width = 0.05



class Band(BaseShape):
    r"""The intensity pattern of a band.

    Let :math:`\left(u_{x;\text{B};1},u_{y;\text{B};1}\right)`,
    :math:`\left(u_{x;\text{B};2},u_{y;\text{B};2}\right)`, and
    :math:`W_{\text{B}}` be the first end point, the second end point, and the
    width of the band respectively. Furthermore, let :math:`A_{\text{B}}` be the
    maximum value of the peak. The undistorted intensity pattern of the band is
    given by:

    .. math ::
        \mathcal{I}_{\text{B}}\left(u_{x},u_{y}\right)=
        A_{\text{B}}\Theta\left(\frac{W_{\text{B}}}{2}
        -d_{\text{B};1}\left(u_{x},u_{y}\right)\right)\Theta\left(
        \frac{L_{\text{B}}}{2}
        -d_{\text{B};2}\left(u_{x},u_{y}\right)\right),
        :label: intensity_pattern_of_band__1

    where :math:`u_{x}` and :math:`u_{y}` are fractional horizontal and vertical
    coordinates of the undistorted intensity pattern of the band respectively,

    .. math ::
        \Theta\left(\omega\right)=\begin{cases}
        1, & \text{if }\omega\ge0,\\
        0, & \text{otherwise};
        \end{cases}
        :label: heaviside_step_function__1

    .. math ::
        L_{\text{B}}=\sqrt{\left(u_{x;\text{B};2}
        -u_{x;\text{B};1}\right)^{2}+\left(u_{y;\text{B};2}
        -u_{y;\text{B};1}\right)^{2}};
        :label: length_of_band__1

    .. math ::
        d_{\text{B};1}\left(u_{x},u_{y}\right)=
        \frac{a_{\text{B};1}u_{x}+b_{\text{B};1}u_{y}
        +c_{\text{B};1}}{\sqrt{a_{\text{B};1}^{2}+b_{\text{B};1}^{2}}};
        :label: d_1_of_band__1

    with

    .. math ::
        a_{\text{B};1}=\begin{cases}
        u_{y;\text{B};2}-u_{y;\text{B};1}, 
        & \text{if }u_{x;\text{B};1}\neq u_{x;\text{B};2},\\
        1, & \text{otherwise},
        \end{cases}
        :label: a_1_of_band__1

    .. math ::
        b_{\text{B};1}=u_{x;\text{B};1}-u_{x;\text{B};2},
        :label: b_1_of_band__1

    .. math ::
        c_{\text{B};1}=\begin{cases}
        u_{x;\text{B};2}u_{y;\text{B};1}
        -u_{x;\text{B};1}u_{y;\text{B};2}, 
        & \text{if }u_{x;\text{B};1}\neq u_{x;\text{B};2},\\
        -u_{x;\text{B};1}, & \text{otherwise};
        \end{cases}
        :label: c_1_of_band__1

    .. math ::
        d_{\text{B};2}\left(u_{x},u_{y}\right)=
        \frac{a_{\text{B};2}u_{x}+b_{\text{B};2}u_{y}
        +c_{\text{B};2}}{\sqrt{a_{\text{B};2}^{2}+b_{\text{B};2}^{2}}},
        :label: d_2_of_band__1

    with

    .. math ::
        a_{\text{B};2}=\begin{cases}
        u_{y;\text{B};4}-u_{y;\text{B};3}, 
        & \text{if }u_{x;\text{B};3}\neq u_{x;\text{B};4},\\
        1, & \text{otherwise},
        \end{cases}
        :label: a_2_of_band__1

    .. math ::
        u_{x;\text{B};3}=u_{x;\text{B};1}
        +\frac{L_{\text{B}}}{2}\cos\left(\theta_{\text{B}}\right),
        :label: u_x_3_of_band__1

    .. math ::
        u_{y;\text{B};3}=u_{y;\text{B};1}
        +\frac{L_{\text{B}}}{2}\sin\left(\theta_{\text{B}}\right),
        :label: u_y_3_of_band__1

    .. math ::
        u_{x;\text{B};4}=u_{x;\text{B};3}
        +\frac{L_{\text{B}}}{2}\cos\left(\phi_{\text{B}}\right),
        :label: u_x_4_of_band__1

    .. math ::
        u_{y;\text{B};4}=u_{y;\text{B};3}
        +\frac{L_{\text{B}}}{2}\sin\left(\phi_{\text{B}}\right),
        :label: u_y_4_of_band__1

    .. math ::
        \theta_{\text{B}}=\tan^{-1}\left(\frac{u_{y;\text{B};2}
        -u_{y;\text{B};1}}{u_{x;\text{B};2}-u_{x;\text{B};1}}\right),
        :label: theta_of_band__1

    .. math ::
        \phi_{\text{B}}=\theta_{\text{B}}+\frac{\pi}{2},
        :label: phi_of_band__1

    .. math ::
        b_{\text{B};2}=u_{x;\text{B};3}-u_{x;\text{B};4},
        :label: b_2_of_band__1

    .. math ::
        c_{\text{B};2}=\begin{cases}
        u_{x;\text{B};4}u_{y;\text{B};3}
        -u_{x;\text{B};3}u_{y;\text{B};4}, 
        & \text{if }u_{x;\text{B};3}\neq u_{x;\text{B};4},\\
        -u_{x;\text{B};3}, & \text{otherwise}.
        \end{cases}
        :label: c_2_of_band__1

    Parameters
    ----------
    end_pt_1 : `array_like` (`float`, shape=(``2``,)), optional
        The first end point of the band,
        :math:`\left(u_{x;\text{B};1},u_{y;\text{B};1}\right)`.
    end_pt_2 : `array_like` (`float`, shape=(``2``,)), optional
        The second end point of the band,
        :math:`\left(u_{x;\text{B};2},u_{y;\text{B};2}\right)`.
    width : `float`, optional
        The width of the band, :math:`W_{\text{B}}`. Must be a positve number.
    intra_shape_val : `float`, optional
        The value of the intensity pattern inside the band.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("end_pt_1",
                        "end_pt_2",
                        "width",
                        "intra_shape_val")
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
                 end_pt_1=\
                 _default_end_pt_1,
                 end_pt_2=\
                 _default_end_pt_2,
                 width=\
                 _default_width,
                 intra_shape_val=\
                 _default_intra_shape_val,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        BaseShape.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        u_x_1, u_y_1 = self._end_pt_1
        u_x_2, u_y_2 = self._end_pt_2

        length = np.sqrt((u_x_2-u_x_1)**2 + (u_y_2-u_y_1)**2)
        theta = np.arctan2(u_y_2-u_y_1, u_x_2-u_x_1)
        phi = theta + (np.pi/2)

        u_x_3 = (u_x_1 + (length/2)*np.cos(theta)).item()
        u_y_3 = (u_y_1 + (length/2)*np.sin(theta)).item()
        u_x_4 = (u_x_3 + (length/2)*np.cos(phi)).item()
        u_y_4 = (u_y_3 + (length/2)*np.sin(phi)).item()

        a_1 = u_y_2-u_y_1 if (u_x_1 != u_x_2) else 1
        b_1 = u_x_1-u_x_2
        c_1 = (u_x_2*u_y_1-u_x_1*u_y_2) if (u_x_1 != u_x_2) else -u_x_1

        a_2 = u_y_4-u_y_3 if (u_x_3 != u_x_4) else 1
        b_2 = u_x_3-u_x_4
        c_2 = (u_x_4*u_y_3-u_x_3*u_y_4) if (u_x_3 != u_x_4) else -u_x_3

        self._a_1 = a_1
        self._b_1 = b_1
        self._c_1 = c_1
        self._denom_of_d_1 = np.sqrt(a_1*a_1 + b_1*b_1).item()

        self._a_2 = a_2
        self._b_2 = b_2
        self._c_2 = c_2
        self._denom_of_d_2 = np.sqrt(a_2*a_2 + b_2*b_2).item()

        self._length = length

        return None



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    def _d_1(self, u_x, u_y):
        a_1 = self._a_1
        b_1 = self._b_1
        c_1 = self._c_1
        denom_of_d_1 = self._denom_of_d_1
        
        d_1 = torch.abs(a_1*u_x + b_1*u_y + c_1) / denom_of_d_1

        return d_1



    def _d_2(self, u_x, u_y):
        a_2 = self._a_2
        b_2 = self._b_2
        c_2 = self._c_2
        denom_of_d_2 = self._denom_of_d_2
        
        d_2 = torch.abs(a_2*u_x + b_2*u_y + c_2) / denom_of_d_2

        return d_2



    def _eval(self, u_x, u_y):
        A = self._intra_shape_val
        w_over_2 = self._width/2
        l_over_2 = self._length/2

        d_1 = self._d_1(u_x, u_y)
        d_2 = self._d_2(u_x, u_y)

        one = torch.tensor(1.0, device=d_1.device)
        
        result = (A
                  * torch.heaviside(w_over_2 - d_1, one)
                  * torch.heaviside(l_over_2 - d_2, one))

        return result



def _check_and_convert_amplitude(params):
    obj_name = "amplitude"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    amplitude = czekitout.convert.to_float(**kwargs)

    return amplitude



def _pre_serialize_amplitude(amplitude):
    obj_to_pre_serialize = amplitude
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_amplitude(serializable_rep):
    amplitude = serializable_rep

    return amplitude



def _check_and_convert_wavelength(params):
    obj_name = "wavelength"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    wavelength = czekitout.convert.to_positive_float(**kwargs)

    return wavelength



def _pre_serialize_wavelength(wavelength):
    obj_to_pre_serialize = wavelength
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_wavelength(serializable_rep):
    wavelength = serializable_rep

    return wavelength



def _check_and_convert_propagation_direction(params):
    obj_name = "propagation_direction"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    propagation_direction = czekitout.convert.to_float(**kwargs) % (2*np.pi)

    return propagation_direction



def _pre_serialize_propagation_direction(propagation_direction):
    obj_to_pre_serialize = propagation_direction
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_propagation_direction(serializable_rep):
    propagation_direction = serializable_rep

    return propagation_direction



def _check_and_convert_phase(params):
    obj_name = "phase"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    phase = czekitout.convert.to_float(**kwargs)

    return phase



def _pre_serialize_phase(phase):
    obj_to_pre_serialize = phase
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_phase(serializable_rep):
    phase = serializable_rep

    return phase



_default_amplitude = 1
_default_wavelength = 0.01
_default_propagation_direction = 0
_default_phase = 0



class PlaneWave(BaseShape):
    r"""The intensity pattern of a plane wave.

    Let :math:`A_{\text{PW}}`, :math:`\lambda_{\text{PW}}`,
    :math:`\theta_{\text{PW}}`, and :math:`\phi_{\text{PW}}` be the amplitude,
    the wavelength, the propagation direction, and the phase of the plane wave
    respectively. The undistorted intensity pattern of the plane wave is given
    by:

    .. math ::
        \mathcal{I}_{\text{PW}}\left(u_{x},u_{y}\right)=
        A_{\text{PW}}\cos\left(u_{x}k_{x;\text{PW}}+u_{y}k_{y;\text{PW}}
        +\phi_{\text{PW}}\right),
        :label: intensity_pattern_of_plane_wave__1

    where :math:`u_{x}` and :math:`u_{y}` are fractional horizontal and vertical
    coordinates of the undistorted intensity pattern of the plane wave
    respectively,

    .. math ::
        k_{x;\text{PW}}=\frac{2\pi}{\lambda_{\text{PW}}}
        \cos\left(\theta_{\text{PW}}\right),
        :label: k_x_of_plane_wave__1

    and

    .. math ::
        k_{y;\text{PW}}=\frac{2\pi}{\lambda_{\text{PW}}}
        \sin\left(\theta_{\text{PW}}\right).
        :label: k_y_of_plane_wave__1

    Parameters
    ----------
    amplitude : `float`, optional        
        The amplitude of the plane wave, :math:`A_{\text{PW}}`.
    wavelength : `float`, optional
        The wavelength of the plane wave, :math:`\lambda_{\text{PW}}`. Must be a
        positve number.
    propagation_direction : `float`, optional
        The propagation direction of the plane wave, :math:`\theta_{\text{PW}}`.
    phase : `float`, optional
        The phase of the plane wave, :math:`\phi_{\text{PW}}`.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("amplitude",
                        "wavelength",
                        "propagation_direction",
                        "phase")
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
                 amplitude=\
                 _default_amplitude,
                 wavelength=\
                 _default_wavelength,
                 propagation_direction=\
                 _default_propagation_direction,
                 phase=\
                 _default_phase,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        BaseShape.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        L = self._wavelength
        theta = self._propagation_direction

        self._k_x = (2*np.pi/L)*np.cos(theta).item()
        self._k_y = (2*np.pi/L)*np.sin(theta).item()

        return None



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    def _eval(self, u_x, u_y):
        A = self._amplitude
        phi = self._phase
        k_x = self._k_x
        k_y = self._k_y

        result = A*torch.cos(u_x*k_x+u_y*k_y + phi)

        return result



def _check_and_convert_midpoint_angle(params):
    obj_name = "midpoint_angle"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    midpoint_angle = czekitout.convert.to_float(**kwargs) % (2*np.pi)

    return midpoint_angle



def _pre_serialize_midpoint_angle(midpoint_angle):
    obj_to_pre_serialize = midpoint_angle
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_midpoint_angle(serializable_rep):
    midpoint_angle = serializable_rep

    return midpoint_angle



def _check_and_convert_subtending_angle(params):
    obj_name = "subtending_angle"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    subtending_angle = min(abs(czekitout.convert.to_float(**kwargs)), (2*np.pi))

    return subtending_angle



def _pre_serialize_subtending_angle(subtending_angle):
    obj_to_pre_serialize = subtending_angle
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_subtending_angle(serializable_rep):
    subtending_angle = serializable_rep

    return subtending_angle



def _check_and_convert_radial_range(params):
    obj_name = "radial_range"
    obj = params[obj_name]

    func_alias = czekitout.convert.to_pair_of_positive_floats
    kwargs = {"obj": obj, "obj_name": obj_name}
    radial_range = func_alias(**kwargs)

    current_func_name = "_check_and_convert_radial_range"

    if radial_range[0] >= radial_range[1]:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise ValueError(err_msg)

    return radial_range



def _pre_serialize_radial_range(radial_range):
    obj_to_pre_serialize = radial_range
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_radial_range(serializable_rep):
    radial_range = serializable_rep

    return radial_range



_default_midpoint_angle = 0
_default_subtending_angle = np.pi/4
_default_radial_range = (0.10, 0.15)



class Arc(BaseShape):
    r"""The intensity pattern of a circular arc.

    Let :math:`\left(u_{x;c;\text{A}},u_{y;c;\text{A}}\right)`,
    :math:`\theta_{\text{A}}`, :math:`\phi_{\text{A}}`, and
    :math:`\left(R_{\text{A};1},R_{\text{A};2}\right)` be the circle center, the
    midpoint angle, the subtending angle, and the radial range of the circular
    arc respectively. Furthermore, let :math:`A_{\text{A}}` be the value of the
    intensity pattern inside the arc. The undistorted intensity pattern of the
    circular arc is given by:

    .. math ::
        \mathcal{I}_{\text{A}}\left(u_{x},u_{y}\right)&=
        A_{\text{A}}\\&\quad\mathop{\times}
        \Theta\left(\left|\frac{\phi_{\text{A}}}{2}\right|
        -u_{\theta;\text{A}}\right)\\
        &\quad\mathop{\times}\Theta\left(\left|\frac{\phi_{\text{A}}}{2}\right|
        +u_{\theta;\text{A}}\right)\\
        &\quad\mathop{\times}\Theta\left(u_{r;\text{A}}
        -R_{\text{A};1}\right)\\
        &\quad\mathop{\times}\Theta\left(R_{\text{A};2}
        -u_{r;\text{A}}\right),
        :label: intensity_pattern_of_arc__1

    where :math:`u_{x}` and :math:`u_{y}` are fractional horizontal and vertical
    coordinates of the undistorted intensity pattern of the circular arc
    respectively, :math:`\Theta\left(\cdots\right)` is the Heaviside step
    function,

    .. math ::
        u_{r;\text{A}}=\sqrt{\left(u_{x}-u_{x;c;\text{A}}\right)^{2}
        +\left(u_{y}-u_{y;c;\text{A}}\right)^{2}},
        :label: u_r_UA__1

    and

    .. math ::
        u_{\theta;\text{A}}=\left\{ \tan^{-1}\left(\frac{u_{y}
        -u_{y;c;\text{A}}}{u_{x}-u_{x;c;\text{A}}}\right)
        -\theta_{\text{A}}\right\} \mod 2\pi.
        :label: u_theta_UA__1

    Parameters
    ----------
    center : `array_like` (`float`, shape=(``2``,)), optional
        The circle center, :math:`\left(u_{x;c;\text{A}},
        u_{y;c;\text{A}}\right)`.
    midpoint_angle : `float`, optional
        The midpoint angle of the circular arc, :math:`\theta_{\text{A}}`.
    subtending_angle : `float`, optional
        The subtending angle of the circular arc, :math:`\phi_{\text{A}}`.
    radial_range : `array_like` (`float`, shape=(2,)), optional
        The radial range of the circular arc,
        :math:`\left(R_{\text{A};1},R_{\text{A};2}\right)`, where
        ``radial_range[0]`` and ``radial_range{1]`` are :math:`R_{\text{A};1}`
        and :math:`R_{\text{A};2}` respectively. ``radial_range`` must satisfy
        ``0<radial_range[0]<radial_range[1]``.
    intra_shape_val : `float`, optional
        The value of the intensity pattern inside the circular arc.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("center",
                        "midpoint_angle",
                        "subtending_angle",
                        "radial_range",
                        "intra_shape_val")
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
                 midpoint_angle=\
                 _default_midpoint_angle,
                 subtending_angle=\
                 _default_subtending_angle,
                 radial_range=\
                 _default_radial_range,
                 intra_shape_val=\
                 _default_intra_shape_val,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        BaseShape.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        return None



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    def _eval(self, u_x, u_y):
        u_x_c, u_y_c = self._center
        theta = self._midpoint_angle
        phi = self._subtending_angle
        R_1, R_2 = self._radial_range
        A = self._intra_shape_val

        delta_u_x = u_x-u_x_c
        delta_u_y = u_y-u_y_c

        phi_over_2 = phi/2

        u_r = torch.sqrt(delta_u_x*delta_u_x + delta_u_y*delta_u_y)
        u_theta = ((torch.atan2(delta_u_y, delta_u_x)-theta+phi_over_2)
                   % (2*np.pi))

        one = torch.tensor(1.0, device=u_x.device)

        result = (A
                  * torch.heaviside(phi-u_theta, one)
                  * torch.heaviside(u_r-R_1, one)
                  * torch.heaviside(R_2-u_r, one))

        return result



def _check_and_convert_radial_reference_pt(params):
    obj_name = "radial_reference_pt"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    radial_reference_pt = czekitout.convert.to_pair_of_floats(**kwargs)

    return radial_reference_pt



def _pre_serialize_radial_reference_pt(radial_reference_pt):
    obj_to_pre_serialize = radial_reference_pt
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_radial_reference_pt(serializable_rep):
    radial_reference_pt = serializable_rep

    return radial_reference_pt



def _check_and_convert_radial_amplitudes(params):
    obj_name = "radial_amplitudes"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    func_alias = czekitout.convert.to_tuple_of_nonnegative_floats
    radial_amplitudes = func_alias(**kwargs)

    current_func_name = "_check_and_convert_radial_amplitudes"
    
    num_radial_amplitudes = len(radial_amplitudes)
    if num_radial_amplitudes == 0:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise ValueError(err_msg)

    partial_amplitude_sum = sum(radial_amplitudes[1:])
    if radial_amplitudes[0] <= partial_amplitude_sum:
        err_msg = globals()[current_func_name+"_err_msg_2"]
        raise ValueError(err_msg)

    return radial_amplitudes



def _pre_serialize_radial_amplitudes(radial_amplitudes):
    obj_to_pre_serialize = radial_amplitudes
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_radial_amplitudes(serializable_rep):
    radial_amplitudes = serializable_rep

    return radial_amplitudes



def _check_and_convert_radial_phases(params):
    obj_name = "radial_phases"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    radial_phases = czekitout.convert.to_tuple_of_floats(**kwargs)
    radial_phases = tuple(radial_phase%(2*np.pi)
                          for radial_phase
                          in radial_phases)

    radial_amplitudes = _check_and_convert_radial_amplitudes(params)

    num_radial_phases = len(radial_phases)
    num_radial_amplitudes = len(radial_amplitudes)

    current_func_name = "_check_and_convert_radial_phases"

    if num_radial_phases+1 != num_radial_amplitudes:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise ValueError(err_msg)

    return radial_phases



def _pre_serialize_radial_phases(radial_phases):
    obj_to_pre_serialize = radial_phases
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_radial_phases(serializable_rep):
    radial_phases = serializable_rep

    return radial_phases



_default_radial_reference_pt = (0.5, 0.5)
_default_radial_amplitudes = (0.1,)
_default_radial_phases = tuple()



class GenericBlob(BaseShape):
    r"""The intensity pattern of a generic blob.

    Let :math:`\left(u_{x;c;\text{GB}},u_{y;c;\text{GB}}\right)`,
    :math:`N_{\text{GB}}`, :math:`\left\{ \phi_{\text{GB};n}\right\}
    _{n=0}^{N_{\text{GB}}-1}`, and :math:`\left\{ D_{\text{GB};n}\right\}
    _{n=0}^{N_{\text{GB}}}` be the radial reference point, the number of radial
    phases, the radial phases, and the radial amplitudes of the generic blob
    respectively. Furthermore, let :math:`A_{\text{GB}}` be the value of the
    intensity pattern inside the generic blob. The undistorted intensity pattern
    of the generic blob is given by:

    .. math ::
        \mathcal{I}_{\text{GB}}\left(u_{x},u_{y}\right)=
        A_{\text{GB}}\Theta\left(
        R_{\text{GB}}\left(u_{\theta;\text{UA}}\right)-u_{r;\text{GB}}\right),
        :label: intensity_pattern_of_generic_blob__1

    where :math:`u_{x}` and :math:`u_{y}` are fractional horizontal and vertical
    coordinates of the undistorted intensity pattern of the generic blob
    respectively, :math:`\Theta\left(\cdots\right)` is the Heaviside step
    function,

    .. math ::
        u_{r;\text{GB}}=\sqrt{\left(u_{x}-u_{x;c;\text{GB}}\right)^{2}
        +\left(u_{y}-u_{y;c;\text{GB}}\right)^{2}};
        :label: u_r_UGB__1

    .. math ::
        u_{\theta;\text{GB}}=
        \tan^{-1}\left(\frac{u_{y}
        -u_{y;c;\text{GB}}}{u_{x}-u_{x;c;\text{GB}}}\right);
        :label: u_theta_UGB__1

    and

    .. math ::
        R_{\text{GB}}\left(u_{\theta;\text{GB}}\right)&=
        D_{\text{GB};0}\\&\quad\mathop{+}\min\left(1,N_{\text{GB}}\right)
        \sum_{n=1}^{N_{\text{GB}}}D_{\text{GB};n}
        \cos\left(nu_{\theta;\text{GB}}-\phi_{\text{GB};n-1}\right),
        :label: R_UGB__1

    with

    .. math ::
        D_{\text{GB};n} \ge 0,
        \quad\forall n\in\left\{ 1,\ldots,N_{\text{GB}}\right\},
        :label: D_UGB_n__1

    and

    .. math ::
        D_{\text{GB};0}>\sum_{n=1}^{N_{\text{GB}}}D_{\text{GB};n}.
        :label: D_UGB_n__2

    Parameters
    ----------
    radial_reference_pt : `array_like` (`float`, shape=(``2``,)), optional
        The radial reference point, :math:`\left(u_{x;c;\text{GB}},
        u_{y;c;\text{GB}}\right)`.
    radial_amplitudes : `array_like` (`float`, ndim=1), optional
        The radial amplitudes, 
        :math:`\left\{ D_{\text{GB};n}\right\} _{n=0}^{N_{\text{GB}}}`.
    radial_phases : `array_like` (`float`, shape=(``len(radial_amplitudes)-1``,)), optional
        The radial phases, 
        :math:`\left\{\phi_{\text{GB};n}\right\}_{n=0}^{N_{\text{GB}}-1}`.
    intra_shape_val : `float`, optional
        The value of the intensity pattern inside the generic blob.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("radial_reference_pt",
                        "radial_amplitudes",
                        "radial_phases",
                        "intra_shape_val")
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
                 radial_reference_pt=\
                 _default_radial_reference_pt,
                 radial_amplitudes=\
                 _default_radial_amplitudes,
                 radial_phases=\
                 _default_radial_phases,
                 intra_shape_val=\
                 _default_intra_shape_val,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        BaseShape.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        return None



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    def _eval(self, u_x, u_y):
        u_x_c, u_y_c = self._radial_reference_pt
        D = self._radial_amplitudes
        phi = self._radial_phases
        A = self._intra_shape_val

        delta_u_x = u_x-u_x_c
        delta_u_y = u_y-u_y_c

        u_r = torch.sqrt(delta_u_x*delta_u_x + delta_u_y*delta_u_y)
        u_theta = torch.atan2(delta_u_y, delta_u_x) % (2*np.pi)

        N = len(phi)

        one = torch.tensor(1.0, device=u_x.device)

        R = D[0]*torch.ones_like(u_theta)
        for n in range(1, N+1):
            R += D[n]*torch.cos(n*u_theta - phi[n-1])

        result = A * torch.heaviside(R-u_r, one)

        return result



def _check_and_convert_principal_quantum_number(params):
    obj_name = "principal_quantum_number"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    principal_quantum_number = czekitout.convert.to_positive_int(**kwargs)

    return principal_quantum_number



def _pre_serialize_principal_quantum_number(principal_quantum_number):
    obj_to_pre_serialize = principal_quantum_number
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_principal_quantum_number(serializable_rep):
    principal_quantum_number = serializable_rep

    return principal_quantum_number



def _check_and_convert_azimuthal_quantum_number(params):
    obj_name = "azimuthal_quantum_number"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    azimuthal_quantum_number = czekitout.convert.to_nonnegative_int(**kwargs)

    principal_quantum_number = \
        _check_and_convert_principal_quantum_number(params)

    current_func_name = "_check_and_convert_azimuthal_quantum_number"

    if azimuthal_quantum_number >= principal_quantum_number:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise ValueError(err_msg)

    return azimuthal_quantum_number



def _pre_serialize_azimuthal_quantum_number(azimuthal_quantum_number):
    obj_to_pre_serialize = azimuthal_quantum_number
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_azimuthal_quantum_number(serializable_rep):
    azimuthal_quantum_number = serializable_rep

    return azimuthal_quantum_number



def _check_and_convert_magnetic_quantum_number(params):
    obj_name = "magnetic_quantum_number"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    magnetic_quantum_number = czekitout.convert.to_int(**kwargs)

    azimuthal_quantum_number = \
        _check_and_convert_azimuthal_quantum_number(params)

    current_func_name = "_check_and_convert_magnetic_quantum_number"

    if abs(magnetic_quantum_number) > azimuthal_quantum_number:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise ValueError(err_msg)

    return magnetic_quantum_number



def _pre_serialize_magnetic_quantum_number(magnetic_quantum_number):
    obj_to_pre_serialize = magnetic_quantum_number
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_magnetic_quantum_number(serializable_rep):
    magnetic_quantum_number = serializable_rep

    return magnetic_quantum_number



def _check_and_convert_effective_size(params):
    obj_name = "effective_size"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    effective_size = czekitout.convert.to_positive_float(**kwargs)

    return effective_size



def _pre_serialize_effective_size(effective_size):
    obj_to_pre_serialize = effective_size
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_effective_size(serializable_rep):
    effective_size = serializable_rep

    return effective_size



def _check_and_convert_renormalization_factor(params):
    obj_name = "renormalization_factor"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    renormalization_factor = czekitout.convert.to_float(**kwargs)

    return renormalization_factor



def _pre_serialize_renormalization_factor(renormalization_factor):
    obj_to_pre_serialize = renormalization_factor
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_renormalization_factor(serializable_rep):
    renormalization_factor = serializable_rep

    return renormalization_factor



_default_principal_quantum_number = 1
_default_azimuthal_quantum_number = 0
_default_magnetic_quantum_number = 0
_default_effective_size = 0.1
_default_renormalization_factor = 1.0



class Orbital(BaseShape):
    r"""The intensity pattern of a hydrogen-like atomic orbital.

    Let :math:`\left(u_{x;c;\text{O}},u_{y;c;\text{O}}\right)`,
    :math:`n_{\text{O}}`, :math:`l_{\text{O}}`, :math:`m_{\text{O}}`,
    :math:`a_{0;\text{O}}^{*}`, and :math:`\theta_{\text{O}}` be the center, the
    principal quantum number, the azimuthal quantum number, the magnetic quantum
    number, the effective size, and the rotation angle of the hydrogen-like
    atomic orbital respectively. Furthermore, let :math:`A_{\text{O}}` be the
    renormalization factor of the intensity pattern. The undistorted intensity
    pattern of the orbital is given by:

    .. math ::
        \mathcal{I}_{\text{O}}\left(u_{x},u_{y}\right)=
        A_{\text{O}}\left|\psi_{n_{\text{O}},l_{\text{O}},m_{O}}\left(
        u_{r;\text{O}},u_{\theta;O},0\right)\right|^{2},
        :label: intensity_pattern_of_orbital__1

    where :math:`u_{x}` and :math:`u_{y}` are fractional horizontal and vertical
    coordinates of the undistorted intensity pattern of the orbital
    respectively,

    .. math ::
        u_{r;\text{O}}=\sqrt{\left(u_{x}-u_{x;c;\text{O}}\right)^{2}
        +\left(u_{y}-u_{y;c;\text{O}}\right)^{2}},
        :label: u_r_O__1

    .. math ::
        u_{\theta;\text{O}}=\tan^{-1}\left(\frac{u_{y}-u_{y;c;\text{O}}}{
        u_{x}-u_{x;c;\text{O}}}\right)-\theta_{\text{O}};
        :label: u_theta_O__1

    .. math ::
        \psi_{n_{\text{O}},l_{\text{O}},m_{O}}\left(u_{r;\text{O}},u_{\theta;O},
        u_{\phi;\text{O}}\right)&=\sqrt{\left\{ \frac{2}{n_{\text{O}}
        a_{0;\text{O}}^{*}}\right\} ^{3}\frac{\left(
        n_{\text{O}}-l_{\text{O}}-1\right)!}{2n_{\text{O}}\left(
        n_{\text{O}}+l_{\text{O}}\right)!}}\\&\quad\mathop{\times}
        e^{-u_{\rho;\text{O}}/2}u_{\rho;\text{O}}^{l_{\text{O}}}
        L_{n_{\text{O}}-l_{\text{O}}-1}^{\left(2l_{\text{O}}+1\right)}\left(
        u_{\rho;\text{O}}\right)\\&\quad\mathop{\times}
        Y_{l_{\text{O}}}^{m_{\text{O}}}\left(u_{\theta;\text{O}},
        u_{\phi;\text{O}}\right),
        :label: psi__1

    with

    .. math ::
        u_{\rho;\text{O}}=\frac{2u_{r;\text{O}}}{n_{\text{O}}
        a_{0;\text{O}}^{*}},
        :label: u_rho_O__1

    :math:`L_{n_{\text{O}}-l_{\text{O}}-1}^{2l_{\text{O}}+1}\left(
    u_{\rho;\text{O}}\right)` being the generalized Laguerre polynomial of
    degree :math:`n_{\text{O}}-l_{\text{O}}-1`, and
    :math:`Y_{l_{\text{O}}}^{m_{\text{O}}}\left(u_{\theta;\text{O}},
    u_{\phi;\text{O}}\right)` is the spherical harmonic function of degree
    :math:`l_{\text{O}}` and order :math:`m_{\text{O}}`.

    Parameters
    ----------
    center : `array_like` (`float`, shape=(``2``,)), optional
        The center of the hydrogen-like atomic orbital,
        :math:`\left(u_{x;c;\text{O}}, u_{y;c;\text{O}}\right)`.
    principal_quantum_number : `int`, optional
        The principal quantum number of the hydrogen-like atomic orbital,
        :math:`n_{\text{O}}`. Must be a positve number.
    azimuthal_quantum_number : `int`, optional
        The azimuthal quantum number of the hydrogen-like atomic orbital,
        :math:`l_{\text{O}}`. Must be a nonnegative number satisfying
        ``azimuthal_quantum_number < principal_quantum_number``.
    magnetic_quantum_number : `int`, optional
        The magnetic quantum number of the hydrogen-like atomic orbital,
        :math:`m_{\text{O}}`. Must satisfy ``abs(magnetic_quantum_number) <=
        azimuthal_quantum_number``.
    effective_size : `float`, optional
        The effective size of the hydrogen-like atomic orbital,
        :math:`a_{0;\text{O}}^{*}`. Must be a positive number.
    renormalization_factor : `float`, optional
        The renormalization factor of the hydrogen-like atomic orbital, 
        :math:`A_{\text{O}}`.
    rotation_angle : `float`, optional
        The rotation angle of the hydrogen-like atomic orbital, 
        :math:`\theta_{\text{O}}`.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("center",
                        "principal_quantum_number",
                        "azimuthal_quantum_number",
                        "magnetic_quantum_number",
                        "effective_size",
                        "renormalization_factor",
                        "rotation_angle")
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
                 principal_quantum_number=\
                 _default_principal_quantum_number,
                 azimuthal_quantum_number=\
                 _default_azimuthal_quantum_number,
                 magnetic_quantum_number=\
                 _default_magnetic_quantum_number,
                 effective_size=\
                 _default_effective_size,
                 renormalization_factor=\
                 _default_renormalization_factor,
                 rotation_angle=\
                 _default_rotation_angle,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        BaseShape.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        n = self._principal_quantum_number
        l = self._azimuthal_quantum_number
        m = self._magnetic_quantum_number
        a = self._effective_size
        A = self._renormalization_factor

        self._pre_factor = (A
                            * ((2/(n*a))**3
                               * math.factorial(n-l-1)
                               / (2*n*math.factorial(n+l)))
                            * ((2*l+1)
                               * math.factorial(l-m)
                               / math.factorial(l+m)
                               / (4*np.pi)))

        self._generalized_laguerre_polynomial_coeffs = \
            self._calc_generalized_laguerre_polynomial_coeffs()

        unformatted_method_name = \
            "_calc_generalized_laguerre_polynomial_and_u_rho_to_power_of_2l_v{}"
        if l <= n-l-1:
            method_name = \
                unformatted_method_name.format(1)
        else:
            method_name = \
                unformatted_method_name.format(2)
        method_alias = \
            getattr(self, method_name)
        self._calc_generalized_laguerre_polynomial_and_u_rho_to_power_of_2l = \
            method_alias

        self._associated_legendre_polynomial_coeffs = \
            self._calc_associated_legendre_polynomial_coeffs()

        return None



    def _calc_generalized_laguerre_polynomial_coeffs(self):
        n = self._principal_quantum_number
        l = self._azimuthal_quantum_number

        polynomial_degree = n-l-1
        alpha = 2*l+1

        coeff = 1
        numerator = polynomial_degree+alpha
        denominator = polynomial_degree
        for k in range(polynomial_degree):
            coeff *= (numerator/denominator)
            numerator -= 1
            denominator -= 1

        generalized_laguerre_polynomial_coeffs = (coeff,)
        for i in range(polynomial_degree):
            i_plus_1 = i+1
            coeff = -((polynomial_degree-i)
                      / (i_plus_1 * (alpha+i_plus_1))
                      * generalized_laguerre_polynomial_coeffs[-1])
            generalized_laguerre_polynomial_coeffs += (coeff,)

        return generalized_laguerre_polynomial_coeffs



    def _calc_associated_legendre_polynomial_coeffs(self):
        l = self._azimuthal_quantum_number
        m = self._magnetic_quantum_number
        abs_m = abs(m)
        r = (l-abs_m)//2

        coeff = (-1)**r
        numerator = 2*l - 2*r
        denominator = l - r
        for k in range(0, l-r):
            coeff *= (numerator/denominator)
            numerator -= 1
            denominator -= 1
        denominator = r
        for k in range(l-r, l):
            coeff *= (numerator/denominator)
            numerator -= 1
            denominator -= 1
        denominator = 1
        for k in range(l, 2*l-2*r):
            coeff *= (numerator/denominator)
            numerator -= 1
        coeff /= 2**l

        associated_legendre_polynomial_coeffs = (coeff,)
        temp_1 = r
        temp_2 = l-r+1
        temp_3 = 2*l-2*r+1
        temp_4 = l-abs_m-2*r+1
        for i in range(r):
            coeff = -((temp_1/temp_2)
                      * (temp_3/temp_4)
                      * ((temp_3+1)/(temp_4+1))
                      * associated_legendre_polynomial_coeffs[-1])
            associated_legendre_polynomial_coeffs += (coeff,)
            temp_1 -= 1
            temp_2 += 1
            temp_3 += 2
            temp_4 += 2

        return associated_legendre_polynomial_coeffs



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    def _eval(self, u_x, u_y):
        u_x_c, u_y_c = self._center
        n = self._principal_quantum_number
        l = self._azimuthal_quantum_number
        a = self._effective_size
        theta = self._rotation_angle
        pre_factor = self._pre_factor

        delta_u_x = u_x-u_x_c
        delta_u_y = u_y-u_y_c

        u_r = torch.sqrt(delta_u_x*delta_u_x + delta_u_y*delta_u_y)
        u_rho = (2/n/a)*u_r
        
        u_theta = torch.atan2(delta_u_y, delta_u_x) - theta
        cos_u_theta = torch.cos(u_theta)

        method_name = ("_calc_generalized_laguerre_polynomial"
                       "_and_u_rho_to_power_of_2l")
        method_alias = getattr(self, method_name)
        L, u_rho_to_power_of_2l = method_alias(u_rho)
        L_sq = L*L
        
        method_name = "_calc_associated_legendre_polynomial_sq"
        method_alias = getattr(self, method_name)
        P_sq = method_alias(cos_u_theta)

        result = (pre_factor
                  * torch.exp(-u_rho)
                  * u_rho_to_power_of_2l
                  * L_sq
                  * P_sq)

        return result



    def _calc_generalized_laguerre_polynomial_and_u_rho_to_power_of_2l_v1(
            self, u_rho):
        coeffs = self._generalized_laguerre_polynomial_coeffs

        n = self._principal_quantum_number
        l = self._azimuthal_quantum_number
        polynomial_degree = n-l-1
        
        power_of_u_rho = torch.ones_like(u_rho)
        generalized_laguerre_polynomial = coeffs[0]*power_of_u_rho

        for i in range(1, l+1):
            power_of_u_rho *= u_rho
            generalized_laguerre_polynomial += coeffs[i]*power_of_u_rho

        u_rho_to_power_of_2l = power_of_u_rho*power_of_u_rho

        for i in range(l+1, polynomial_degree+1):
            power_of_u_rho *= u_rho
            generalized_laguerre_polynomial += coeffs[i]*power_of_u_rho

        return generalized_laguerre_polynomial, u_rho_to_power_of_2l



    def _calc_generalized_laguerre_polynomial_and_u_rho_to_power_of_2l_v2(
            self, u_rho):
        coeffs = self._generalized_laguerre_polynomial_coeffs

        n = self._principal_quantum_number
        l = self._azimuthal_quantum_number
        polynomial_degree = n-l-1
        
        power_of_u_rho = torch.ones_like(u_rho)
        generalized_laguerre_polynomial = coeffs[0]*power_of_u_rho

        for i in range(1, polynomial_degree+1):
            power_of_u_rho *= u_rho
            generalized_laguerre_polynomial += coeffs[i]*power_of_u_rho

        for i in range(polynomial_degree+1, l+1):
            power_of_u_rho *= u_rho

        u_rho_to_power_of_2l = power_of_u_rho*power_of_u_rho

        return generalized_laguerre_polynomial, u_rho_to_power_of_2l



    def _calc_associated_legendre_polynomial_sq(self, cos_u_theta):
        coeffs = self._associated_legendre_polynomial_coeffs

        l = self._azimuthal_quantum_number
        m = self._magnetic_quantum_number
        abs_m = abs(m)
        r = (l-abs_m)//2

        x = cos_u_theta
        x_sq = x*x
        power_of_x = torch.ones_like(x)
        for _ in range(l-abs_m-2*r):
            power_of_x *= x

        y_sq = 1-x_sq

        associated_legendre_polynomial_sq = coeffs[0]*power_of_x
        for i in range(1, r+1):
            power_of_x *= x_sq
            associated_legendre_polynomial_sq += coeffs[i]*power_of_x
        associated_legendre_polynomial_sq *= associated_legendre_polynomial_sq
        for i in range(abs_m):
            associated_legendre_polynomial_sq *= y_sq

        return associated_legendre_polynomial_sq



def _check_and_convert_bg_ellipse(params):
    obj_name = "bg_ellipse"
    obj = params[obj_name]

    accepted_types = (Ellipse, Circle, type(None))

    if isinstance(obj, accepted_types[-1]):
        bg_ellipse = accepted_types[0]()
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        bg_ellipse = copy.deepcopy(obj)

    return bg_ellipse



def _pre_serialize_bg_ellipse(bg_ellipse):
    obj_to_pre_serialize = bg_ellipse
    serializable_rep = obj_to_pre_serialize.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_bg_ellipse(serializable_rep):
    if "radius" in serializable_rep:
        bg_ellipse = Circle.de_pre_serialize(serializable_rep)
    else:
        bg_ellipse = Ellipse.de_pre_serialize(serializable_rep)

    return bg_ellipse



def _check_and_convert_fg_ellipse(params):
    obj_name = "fg_ellipse"
    obj = params[obj_name]

    accepted_types = (Ellipse, Circle, type(None))

    if isinstance(obj, accepted_types[-1]):
        fg_ellipse = accepted_types[0]()
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        fg_ellipse = copy.deepcopy(obj)

    return fg_ellipse



def _pre_serialize_fg_ellipse(fg_ellipse):
    obj_to_pre_serialize = fg_ellipse
    serializable_rep = obj_to_pre_serialize.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_fg_ellipse(serializable_rep):
    if "radius" in serializable_rep:
        fg_ellipse = Circle.de_pre_serialize(serializable_rep)
    else:
        fg_ellipse = Ellipse.de_pre_serialize(serializable_rep)

    return fg_ellipse



_default_bg_ellipse = None
_default_fg_ellipse = None



class Lune(BaseShape):
    r"""The intensity pattern of a lune.

    Let :math:`\mathcal{I}_{\text{BE}}\left(u_{x},u_{y}\right)` and
    :math:`\mathcal{I}_{\text{FE}}\left(u_{x},u_{y}\right)` be the intensity
    patterns of the background and the foreground ellipses respectively, the
    latter of which is used to mask the former to form the lune. The undistorted
    intensity pattern of the lune is given by:

    .. math ::
        \mathcal{I}_{\text{L}}\left(u_{x},u_{y}\right)=\begin{cases}
        \mathcal{I}_{\text{BE}}\left(u_{x},u_{y}\right), 
        & \text{if }\mathcal{I}_{\text{FE}}\left(u_{x},u_{y}\right)=0,\\
        0, & \text{otherwise},
        \end{cases}
        :label: intensity_pattern_of_lune__1

    where :math:`u_{x}` and :math:`u_{y}` are fractional horizontal and vertical
    coordinates of the undistorted intensity pattern of the lune respectively.

    Parameters
    ----------
    bg_ellipse : :class:`fakecbed.shapes.Circle` | :class:`fakecbed.shapes.Ellipse` | `None`, optional
        The intensity pattern of the background ellipse,
        :math:`\mathcal{I}_{\text{BE}}\left(u_{x},u_{y}\right)`. If
        ``bg_ellipse`` is set to ``None``, then the parameter will be reassigned
        to the value ``fakecbed.shapes.Ellipse()``.
    fg_ellipse : :class:`fakecbed.shapes.Circle` | :class:`fakecbed.shapes.Ellipse` | `None`, optional
        The intensity pattern of the foreground ellipse,
        :math:`\mathcal{I}_{\text{FE}}\left(u_{x},u_{y}\right)`. If
        ``fg_ellipse`` is set to ``None``, then the parameter will be reassigned
        to the value ``fakecbed.shapes.Ellipse()``.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("bg_ellipse",
                        "fg_ellipse")
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
                 bg_ellipse=\
                 _default_bg_ellipse,
                 fg_ellipse=\
                 _default_fg_ellipse,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        BaseShape.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        return None



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    def _eval(self, u_x, u_y):
        bg_ellipse = self._bg_ellipse
        fg_ellipse = self._fg_ellipse
        result = bg_ellipse.eval(u_x, u_y) * (fg_ellipse.eval(u_x, u_y) == 0)

        return result



def _check_and_convert_support(params):
    obj_name = "support"
    obj = params[obj_name]

    accepted_types = (Circle, Ellipse, Band, Arc, GenericBlob, Lune, type(None))

    if isinstance(obj, accepted_types[-1]):
        support = accepted_types[0]()
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        support = copy.deepcopy(obj)

    return support



def _pre_serialize_support(support):
    obj_to_pre_serialize = support
    serializable_rep = obj_to_pre_serialize.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_support(serializable_rep):
    if "radius" in serializable_rep:
        support = Circle.de_pre_serialize(serializable_rep)
    elif "eccentricity" in serializable_rep:
        support = Ellipse.de_pre_serialize(serializable_rep)
    elif "end_pt_1" in serializable_rep:
        support = Band.de_pre_serialize(serializable_rep)
    elif "subtending_angle" in serializable_rep:
        support = Arc.de_pre_serialize(serializable_rep)
    elif "radial_amplitudes" in serializable_rep:
        support = GenericBlob.de_pre_serialize(serializable_rep)
    else:
        support = Lune.de_pre_serialize(serializable_rep)

    return support



def _check_and_convert_intra_support_shapes(params):
    obj_name = "intra_support_shapes"
    obj = params[obj_name]

    accepted_types = (Circle,
                      Ellipse,
                      Peak,
                      Band,
                      PlaneWave,
                      Arc,
                      GenericBlob,
                      Orbital,
                      Lune,
                      NonuniformBoundedShape)

    current_func_name = "_check_and_convert_intra_support_shapes"

    try:
        for intra_support_shape in obj:
            kwargs = {"obj": intra_support_shape,
                      "obj_name": "intra_support_shape",
                      "accepted_types": accepted_types}
            czekitout.check.if_instance_of_any_accepted_types(**kwargs)
    except:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise TypeError(err_msg)

    intra_support_shapes = copy.deepcopy(obj)

    return intra_support_shapes



def _pre_serialize_intra_support_shapes(intra_support_shapes):
    obj_to_pre_serialize = intra_support_shapes
    serializable_rep = tuple()
    for elem in obj_to_pre_serialize:
        serializable_rep += (elem.pre_serialize(),)
    
    return serializable_rep



def _de_pre_serialize_intra_support_shapes(serializable_rep):
    intra_support_shapes = tuple()

    for pre_serialized_intra_support_shape in serializable_rep:
        if "radius" in pre_serialized_intra_support_shape:
            cls_alias = Circle
        elif "eccentricity" in pre_serialized_intra_support_shape:
            cls_alias = Ellipse
        elif "functional_form" in pre_serialized_intra_support_shape:
            cls_alias = Peak
        elif "end_pt_1" in pre_serialized_intra_support_shape:
            cls_alias = Band
        elif "propagation_direction" in pre_serialized_intra_support_shape:
            cls_alias = PlaneWave
        elif "subtending_angle" in pre_serialized_intra_support_shape:
            cls_alias = Arc
        elif "radial_amplitudes" in pre_serialized_intra_support_shape:
            cls_alias = GenericBlob
        elif "magnetic_quantum_number" in pre_serialized_intra_support_shape:
            cls_alias = Orbital
        elif "bg_ellipse" in pre_serialized_intra_support_shape:
            cls_alias = Lune
        else:
            cls_alias = NonuniformBoundedShape

        intra_support_shape = \
            cls_alias.de_pre_serialize(pre_serialized_intra_support_shape)
        intra_support_shapes += \
            (intra_support_shape,)

    return intra_support_shapes



_default_support = None
_default_intra_support_shapes = tuple()



class NonuniformBoundedShape(BaseShape):
    r"""The intensity pattern of a nonuniform bounded shape.

    Let :math:`\mathcal{I}_{0;\text{NBS}}\left(u_{x},u_{y}\right)`,
    :math:`N_{\text{NBS}}`, and :math:`\left\{
    \mathcal{I}_{k;\text{NBS}}\left(u_{x},u_{y}\right)\right\}
    _{k=1}^{N_{\text{NBS}}}` be the intensity pattern of the uniform bounded
    shape supporting the nonuniform bounded shape, the number of intra-support
    shapes, and the intensity patterns of the intra-support shapes
    respectively. The undistorted intensity pattern of the nonuniform bounded
    shape is given by:

    .. math ::
        \mathcal{I}_{\text{NBS}}\left(u_{x},u_{y}\right)=
        \mathcal{I}_{0;\text{NBS}}\left(u_{x},u_{y}\right)\left|
        \sum_{k=1}^{N_{\text{NBS}}}\mathcal{I}_{k;\text{NBS}}\left(u_{x},
        u_{y}\right)\right|,
        :label: intensity_pattern_of_nonuniform_bounded_shape__1

    where :math:`u_{x}` and :math:`u_{y}` are fractional horizontal and vertical
    coordinates of the undistorted intensity pattern of the nonuniform bounded
    shape respectively.

    Parameters
    ----------
    support : :class:`fakecbed.shapes.Circle` | :class:`fakecbed.shapes.Ellipse` | :class:`fakecbed.shapes.Band` | :class:`fakecbed.shapes.Arc` | :class:`fakecbed.shapes.GenericBlob` | :class:`fakecbed.shapes.Lune` | `None`, optional
        The intensity pattern of the uniform bounded shape supporting the
        nonuniform bounded shape,
        :math:`\mathcal{I}_{0;\text{NBS}}\left(u_{x},u_{y}\right)`. If
        ``support`` is set to ``None``, then the parameter will be reassigned to
        the value ``fakecbed.shapes.Circle()``.
    intra_support_shapes : `array_like` (`any_shape`, ndim=1), optional
        The intensity patterns of the intra-support shapes, :math:`\left\{
        \mathcal{I}_{k;\text{NBS}}\left(u_{x},u_{y}\right)\right\}
        _{k=1}^{N_{\text{NBS}}}`. Note that `any_shape` means any public class
        defined in the module :mod:`fakecbed.shapes` that is a subclass of
        :class:`fakecbed.shapes.BaseShape`.
    skip_validation_and_conversion : `bool`, optional
        Let ``validation_and_conversion_funcs`` and ``core_attrs`` denote the
        attributes :attr:`~fancytypes.Checkable.validation_and_conversion_funcs`
        and :attr:`~fancytypes.Checkable.core_attrs` respectively, both of which
        being `dict` objects.

        Let ``params_to_be_mapped_to_core_attrs`` denote the `dict`
        representation of the constructor parameters excluding the parameter
        ``skip_validation_and_conversion``, where each `dict` key ``key`` is a
        different constructor parameter name, excluding the name
        ``"skip_validation_and_conversion"``, and
        ``params_to_be_mapped_to_core_attrs[key]`` would yield the value of the
        constructor parameter with the name given by ``key``.

        If ``skip_validation_and_conversion`` is set to ``False``, then for each
        key ``key`` in ``params_to_be_mapped_to_core_attrs``,
        ``core_attrs[key]`` is set to ``validation_and_conversion_funcs[key]
        (params_to_be_mapped_to_core_attrs)``.

        Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
        then ``core_attrs`` is set to
        ``params_to_be_mapped_to_core_attrs.copy()``. This option is desired
        primarily when the user wants to avoid potentially expensive deep copies
        and/or conversions of the `dict` values of
        ``params_to_be_mapped_to_core_attrs``, as it is guaranteed that no
        copies or conversions are made in this case.

    """
    ctor_param_names = ("support",
                        "intra_support_shapes")
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
                 support=\
                 _default_support,
                 intra_support_shapes=\
                 _default_intra_support_shapes,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        BaseShape.__init__(self, ctor_params)

        self.execute_post_core_attrs_update_actions()

        return None



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)

        return None



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    def _eval(self, u_x, u_y):
        result = (self._eval_without_support(u_x, u_y)
                  * self._eval_without_intra_support_shapes(u_x, u_y))

        return result



    def _eval_without_intra_support_shapes(self, u_x, u_y):
        support = self._support
        result = support._eval(u_x, u_y)

        return result



    def _eval_without_support(self, u_x, u_y):
        intra_support_shapes = self._intra_support_shapes
        
        result = torch.zeros_like(u_x)
        for intra_support_shape in intra_support_shapes:
            result += intra_support_shape._eval(u_x, u_y)
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

_base_shape_err_msg_1 = \
    ("Cannot construct instances of the class `fakecbed.shapes.BaseShape`, "
     "only subclasses of itself defined in the `fakecbed` library.")

_check_and_convert_eccentricity_err_msg_1 = \
    ("The object ``eccentricity`` must be a nonnegative number less than or "
     "equal to unity.")

_check_and_convert_radial_range_err_msg_1 = \
    ("The object ``radial_range`` must be a pair of positive real numbers "
     "satisfying ``radial_range[0]<radial_range[1]``.")

_check_and_convert_radial_amplitudes_err_msg_1 = \
    ("The object ``radial_amplitudes`` must be a non-empty sequence of "
     "nonnegative real numbers.")
_check_and_convert_radial_amplitudes_err_msg_2 = \
    ("The object ``radial_amplitudes`` must satisfy ``radial_amplitudes[0] > "
     "sum(radial_amplitudes[1:])``.")

_check_and_convert_radial_phases_err_msg_1 = \
    ("The objects ``radial_phases`` and ``radial_amplitudes``  must satisfy "
     "``len(radial_phases)+1 == len(radial_amplitudes)``.")

_check_and_convert_azimuthal_quantum_number_err_msg_1 = \
    ("The objects ``azimuthal_quantum_number`` and "
     "``principal_quantum_number``  must satisfy "
     "``azimuthal_quantum_number < principal_quantum_number``.")

_check_and_convert_magnetic_quantum_number_err_msg_1 = \
    ("The objects ``magnetic_quantum_number`` and "
     "``azimuthal_quantum_number``  must satisfy "
     "``abs(magnetic_quantum_number) <= azimuthal_quantum_number``.")

_check_and_convert_intra_support_shapes_err_msg_1 = \
    ("The object ``intra_support_shapes`` must be a sequence of objects of any "
     "of the following types: ("
     "`fakecbed.shapes.Circle`, "
     "`fakecbed.shapes.Ellipse`, "
     "`fakecbed.shapes.Peak`, "
     "`fakecbed.shapes.Band`, "
     "`fakecbed.shapes.PlaneWave`, "
     "`fakecbed.shapes.Arc`, "
     "`fakecbed.shapes.GenericBlob`, "
     "`fakecbed.shapes.Orbital`, "
     "`fakecbed.shapes.Lune`, "
     "`fakecbed.shapes.NonuniformBoundedShape`).")
