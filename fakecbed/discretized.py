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
"""For creating discretized fake CBED patterns.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

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

# For creating hyperspy signals and axes.
import hyperspy.signals
import hyperspy.axes

# For creating distortion models.
import distoptica

# For inpainting images.
import skimage.restoration

# For cropping hyperspy signals.
import empix



# For creating undistorted geometric shapes.
import fakecbed.shapes

# For creating undistorted thermal diffuse models.
import fakecbed.tds



##################################
## Define classes and functions ##
##################################

# List of public objects in module.
__all__ = ["CBEDPattern",
           "CroppedCBEDPattern"]



def _check_and_convert_undistorted_tds_model(params):
    obj_name = "undistorted_tds_model"
    obj = params[obj_name]

    accepted_types = (fakecbed.tds.Model, type(None))

    if isinstance(obj, accepted_types[-1]):
        undistorted_tds_model = accepted_types[0]()
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        undistorted_tds_model = copy.deepcopy(obj)

    return undistorted_tds_model



def _pre_serialize_undistorted_tds_model(undistorted_tds_model):
    obj_to_pre_serialize = undistorted_tds_model
    serializable_rep = obj_to_pre_serialize.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_undistorted_tds_model(serializable_rep):
    undistorted_tds_model = \
        fakecbed.tds.Model.de_pre_serialize(serializable_rep)

    return undistorted_tds_model



def _check_and_convert_undistorted_disks(params):
    obj_name = "undistorted_disks"
    obj = params[obj_name]

    current_func_name = "_check_and_convert_undistorted_disks"

    try:
        for undistorted_disk in obj:
            accepted_types = (fakecbed.shapes.NonuniformBoundedShape,)
            
            kwargs = {"obj": undistorted_disk,
                      "obj_name": "undistorted_disk",
                      "accepted_types": accepted_types}
            czekitout.check.if_instance_of_any_accepted_types(**kwargs)

            accepted_types = (fakecbed.shapes.Circle, fakecbed.shapes.Ellipse)

            undistorted_disk_core_attrs = \
                undistorted_disk.get_core_attrs(deep_copy=False)
            undistorted_disk_support = \
                undistorted_disk_core_attrs["support"]

            kwargs = {"obj": undistorted_disk_support,
                      "obj_name": "undistorted_disk_support",
                      "accepted_types": accepted_types}
            czekitout.check.if_instance_of_any_accepted_types(**kwargs)
    except:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise TypeError(err_msg)

    undistorted_disks = copy.deepcopy(obj)

    return undistorted_disks



def _pre_serialize_undistorted_disks(undistorted_disks):
    obj_to_pre_serialize = undistorted_disks
    serializable_rep = tuple()
    for elem in obj_to_pre_serialize:
        serializable_rep += (elem.pre_serialize(),)
    
    return serializable_rep



def _de_pre_serialize_undistorted_disks(serializable_rep):
    undistorted_disks = \
        tuple()
    for pre_serialized_undistorted_disk in serializable_rep:
        cls_alias = \
            fakecbed.shapes.NonuniformBoundedShape
        undistorted_disk = \
            cls_alias.de_pre_serialize(pre_serialized_undistorted_disk)
        undistorted_disks += \
            (undistorted_disk,)

    return undistorted_disks



def _check_and_convert_undistorted_misc_shapes(params):
    obj_name = "undistorted_misc_shapes"
    obj = params[obj_name]

    accepted_types = (fakecbed.shapes.Circle,
                      fakecbed.shapes.Ellipse,
                      fakecbed.shapes.Peak,
                      fakecbed.shapes.Band,
                      fakecbed.shapes.PlaneWave,
                      fakecbed.shapes.Arc,
                      fakecbed.shapes.GenericBlob,
                      fakecbed.shapes.Orbital,
                      fakecbed.shapes.Lune,
                      fakecbed.shapes.NonuniformBoundedShape)

    current_func_name = "_check_and_convert_undistorted_misc_shapes"

    try:
        for undistorted_misc_shape in obj:
            kwargs = {"obj": undistorted_misc_shape,
                      "obj_name": "undistorted_misc_shape",
                      "accepted_types": accepted_types}
            czekitout.check.if_instance_of_any_accepted_types(**kwargs)
    except:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise TypeError(err_msg)

    undistorted_misc_shapes = copy.deepcopy(obj)

    return undistorted_misc_shapes



def _pre_serialize_undistorted_misc_shapes(undistorted_misc_shapes):
    obj_to_pre_serialize = undistorted_misc_shapes
    serializable_rep = tuple()
    for elem in obj_to_pre_serialize:
        serializable_rep += (elem.pre_serialize(),)
    
    return serializable_rep



def _de_pre_serialize_undistorted_misc_shapes(serializable_rep):
    undistorted_misc_shapes = tuple()

    for pre_serialized_undistorted_misc_shape in serializable_rep:
        if "radius" in pre_serialized_undistorted_misc_shape:
            cls_alias = fakecbed.shapes.Circle
        elif "eccentricity" in pre_serialized_undistorted_misc_shape:
            cls_alias = fakecbed.shapes.Ellipse
        elif "functional_form" in pre_serialized_undistorted_misc_shape:
            cls_alias = fakecbed.shapes.Peak
        elif "end_pt_1" in pre_serialized_undistorted_misc_shape:
            cls_alias = fakecbed.shapes.Band
        elif "propagation_direction" in pre_serialized_undistorted_misc_shape:
            cls_alias = fakecbed.shapes.PlaneWave
        elif "subtending_angle" in pre_serialized_undistorted_misc_shape:
            cls_alias = fakecbed.shapes.Arc
        elif "radial_amplitudes" in pre_serialized_undistorted_misc_shape:
            cls_alias = fakecbed.shapes.GenericBlob
        elif "magnetic_quantum_number" in pre_serialized_undistorted_misc_shape:
            cls_alias = fakecbed.shapes.Orbital
        elif "bg_ellipse" in pre_serialized_undistorted_misc_shape:
            cls_alias = fakecbed.shapes.Lune
        else:
            cls_alias = fakecbed.shapes.NonuniformBoundedShape

        undistorted_misc_shape = \
            cls_alias.de_pre_serialize(pre_serialized_undistorted_misc_shape)
        undistorted_misc_shapes += \
            (undistorted_misc_shape,)

    return undistorted_misc_shapes



def _check_and_convert_undistorted_outer_illumination_shape(params):
    obj_name = "undistorted_outer_illumination_shape"
    obj = params[obj_name]

    accepted_types = (fakecbed.shapes.Circle,
                      fakecbed.shapes.Ellipse,
                      fakecbed.shapes.GenericBlob,
                      type(None))

    if isinstance(obj, accepted_types[-1]):
        kwargs = {"radius": np.inf}
        undistorted_outer_illumination_shape = accepted_types[0](**kwargs)
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        undistorted_outer_illumination_shape = copy.deepcopy(obj)

    return undistorted_outer_illumination_shape



def _pre_serialize_undistorted_outer_illumination_shape(
        undistorted_outer_illumination_shape):
    obj_to_pre_serialize = undistorted_outer_illumination_shape
    serializable_rep = obj_to_pre_serialize.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_undistorted_outer_illumination_shape(serializable_rep):
    if "radius" in serializable_rep:
        undistorted_outer_illumination_shape = \
            fakecbed.shapes.Circle.de_pre_serialize(serializable_rep)
    elif "eccentricity" in serializable_rep:
        undistorted_outer_illumination_shape = \
            fakecbed.shapes.Ellipse.de_pre_serialize(serializable_rep)
    else:
        undistorted_outer_illumination_shape = \
            fakecbed.shapes.GenericBlob.de_pre_serialize(serializable_rep)

    return undistorted_outer_illumination_shape



def _check_and_convert_gaussian_filter_std_dev(params):
    obj_name = "gaussian_filter_std_dev"
    func_alias = czekitout.convert.to_nonnegative_float
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    gaussian_filter_std_dev = func_alias(**kwargs)

    return gaussian_filter_std_dev



def _pre_serialize_gaussian_filter_std_dev(gaussian_filter_std_dev):
    obj_to_pre_serialize = gaussian_filter_std_dev
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_gaussian_filter_std_dev(serializable_rep):
    gaussian_filter_std_dev = serializable_rep

    return gaussian_filter_std_dev



def _check_and_convert_distortion_model(params):
    obj_name = "distortion_model"
    obj = params[obj_name]

    num_pixels_across_pattern = \
        _check_and_convert_num_pixels_across_pattern(params)

    accepted_types = (distoptica.DistortionModel, type(None))

    if isinstance(obj, accepted_types[-1]):
        sampling_grid_dims_in_pixels = 2*(num_pixels_across_pattern,)
        kwargs = {"sampling_grid_dims_in_pixels": sampling_grid_dims_in_pixels}
        distortion_model = accepted_types[0](**kwargs)
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        distortion_model = copy.deepcopy(obj)

    distortion_model_core_attrs = \
        distortion_model.get_core_attrs(deep_copy=False)
    sampling_grid_dims_in_pixels = \
        distortion_model_core_attrs["sampling_grid_dims_in_pixels"]

    current_func_name = "_check_and_convert_distortion_model"

    if ((sampling_grid_dims_in_pixels[0]%num_pixels_across_pattern != 0)
        or (sampling_grid_dims_in_pixels[1]%num_pixels_across_pattern != 0)):
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise ValueError(err_msg)

    return distortion_model



def _pre_serialize_distortion_model(distortion_model):
    obj_to_pre_serialize = distortion_model
    serializable_rep = obj_to_pre_serialize.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_distortion_model(serializable_rep):
    distortion_model = \
        distoptica.DistortionModel.de_pre_serialize(serializable_rep)

    return distortion_model



def _check_and_convert_num_pixels_across_pattern(params):
    obj_name = "num_pixels_across_pattern"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    num_pixels_across_pattern = czekitout.convert.to_positive_int(**kwargs)

    return num_pixels_across_pattern



def _pre_serialize_num_pixels_across_pattern(num_pixels_across_pattern):
    obj_to_pre_serialize = num_pixels_across_pattern
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_num_pixels_across_pattern(serializable_rep):
    num_pixels_across_pattern = serializable_rep

    return num_pixels_across_pattern



def _check_and_convert_apply_shot_noise(params):
    obj_name = "apply_shot_noise"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    apply_shot_noise = czekitout.convert.to_bool(**kwargs)

    return apply_shot_noise



def _pre_serialize_apply_shot_noise(apply_shot_noise):
    obj_to_pre_serialize = apply_shot_noise
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_apply_shot_noise(serializable_rep):
    apply_shot_noise = serializable_rep

    return apply_shot_noise



def _check_and_convert_rng_seed(params):
    obj_name = "rng_seed"
    obj = params[obj_name]

    current_func_name = "_check_and_convert_rng_seed"
    
    if obj is not None:
        kwargs = {"obj": obj, "obj_name": obj_name}
        try:
            rng_seed = czekitout.convert.to_nonnegative_int(**kwargs)
        except:
            err_msg = globals()[current_func_name+"_err_msg_1"]
            raise TypeError(err_msg)
    else:
        rng_seed = obj

    return rng_seed



def _pre_serialize_rng_seed(rng_seed):
    obj_to_pre_serialize = rng_seed
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_rng_seed(serializable_rep):
    rng_seed = serializable_rep

    return rng_seed



def _check_and_convert_detector_partition_width_in_pixels(params):
    obj_name = "detector_partition_width_in_pixels"
    func_alias = czekitout.convert.to_nonnegative_int
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    detector_partition_width_in_pixels = func_alias(**kwargs)

    return detector_partition_width_in_pixels



def _pre_serialize_detector_partition_width_in_pixels(
        detector_partition_width_in_pixels):
    obj_to_pre_serialize = detector_partition_width_in_pixels
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_detector_partition_width_in_pixels(serializable_rep):
    detector_partition_width_in_pixels = serializable_rep

    return detector_partition_width_in_pixels



def _check_and_convert_cold_pixels(params):
    obj_name = "cold_pixels"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    cold_pixels = czekitout.convert.to_pairs_of_ints(**kwargs)

    num_pixels_across_pattern = \
        _check_and_convert_num_pixels_across_pattern(params)

    current_func_name = "_check_and_convert_cold_pixels"

    coords_of_cold_pixels = cold_pixels
    for coords_of_cold_pixel in coords_of_cold_pixels:
        row, col = coords_of_cold_pixel
        if ((row < -num_pixels_across_pattern)
            or (num_pixels_across_pattern <= row)
            or (col < -num_pixels_across_pattern)
            or (num_pixels_across_pattern <= col)):
            err_msg = globals()[current_func_name+"_err_msg_1"]
            raise TypeError(err_msg)

    return cold_pixels



def _pre_serialize_cold_pixels(cold_pixels):
    serializable_rep = cold_pixels
    
    return serializable_rep



def _de_pre_serialize_cold_pixels(serializable_rep):
    cold_pixels = serializable_rep

    return cold_pixels



def _check_and_convert_mask_frame(params):
    obj_name = "mask_frame"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    mask_frame = czekitout.convert.to_quadruplet_of_nonnegative_ints(**kwargs)

    return mask_frame



def _pre_serialize_mask_frame(mask_frame):
    obj_to_pre_serialize = mask_frame
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_mask_frame(serializable_rep):
    mask_frame = serializable_rep

    return mask_frame



def _check_and_convert_deep_copy(params):
    obj_name = "deep_copy"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    deep_copy = czekitout.convert.to_bool(**kwargs)

    return deep_copy



def _check_and_convert_overriding_image(params):
    obj_name = "overriding_image"
    obj = params[obj_name]

    func_alias = fakecbed.shapes._check_and_convert_real_torch_matrix
    params["real_torch_matrix"] = obj
    params["name_of_alias_of_real_torch_matrix"] = obj_name
    overriding_image = func_alias(params)

    del params["real_torch_matrix"]
    del params["name_of_alias_of_real_torch_matrix"]

    num_pixels_across_pattern = params["num_pixels_across_pattern"]
    expected_image_dims_in_pixels = 2*(num_pixels_across_pattern,)

    current_func_name = "_check_and_convert_overriding_image"

    if overriding_image.shape != expected_image_dims_in_pixels:
        unformatted_err_msg = globals()[current_func_name+"_err_msg_1"]
        args = expected_image_dims_in_pixels
        err_msg = unformatted_err_msg.format(*args)
        raise ValueError(err_msg)

    return overriding_image



def _check_and_convert_skip_validation_and_conversion(params):
    func_alias = \
        fakecbed.shapes._check_and_convert_skip_validation_and_conversion
    skip_validation_and_conversion = \
        func_alias(params)

    return skip_validation_and_conversion



_default_undistorted_outer_illumination_shape = \
    None
_default_undistorted_tds_model = \
    None
_default_undistorted_disks = \
    tuple()
_default_undistorted_misc_shapes = \
    tuple()
_default_gaussian_filter_std_dev = \
    0
_default_distortion_model = \
    None
_default_num_pixels_across_pattern = \
    512
_default_apply_shot_noise = \
    False
_default_rng_seed = \
    None
_default_detector_partition_width_in_pixels = \
    0
_default_cold_pixels = \
    tuple()
_default_skip_validation_and_conversion = \
    fakecbed.shapes._default_skip_validation_and_conversion
_default_deep_copy = \
    True
_default_mask_frame = \
    4*(0,)



class CBEDPattern(fancytypes.PreSerializableAndUpdatable):
    r"""A discretized fake convergent beam electron diffraction (CBED) pattern.

    A series of parameters need to be specified in order to create an image of a
    fake CBED pattern, with the most important parameters being: the set of
    intensity patterns of undistorted shapes that determine the undistorted
    noiseless non-blurred uncorrupted (UNNBU) fake CBED pattern; and a
    distortion model which transforms the UNNBU fake CBED pattern into a
    distorted noiseless non-blurred uncorrupted (DNNBU) fake CBED pattern. The
    remaining parameters determine whether additional images effects are
    applied, like e.g. shot noise or blur effects. Note that in the case of the
    aforementioned shapes, we expand the notion of intensity patterns to mean a
    2D real-valued function, i.e. it can be negative. To be clear, we do not
    apply this generalized notion of intensity patterns to the fake CBED
    patterns: in such cases intensity patterns mean 2D real-valued functions
    that are strictly nonnegative.

    Let :math:`u_{x}` and :math:`u_{y}` be the fractional horizontal and
    vertical coordinates, respectively, of a point in an undistorted image,
    where :math:`\left(u_{x},u_{y}\right)=\left(0,0\right)` is the bottom left
    corner of the image. Secondly, let :math:`q_{x}` and :math:`q_{y}` be the
    fractional horizontal and vertical coordinates, respectively, of a point in
    a distorted image, where :math:`\left(q_{x},q_{y}\right)=\left(0,0\right)`
    is the bottom left corner of the image. When users specify the distortion
    model, represented by an instance of the class
    :class:`distoptica.DistortionModel`, they also specify a coordinate
    transformation, :math:`\left(T_{⌑;x}\left(u_{x},u_{y}\right),
    T_{⌑;y}\left(u_{x},u_{y}\right)\right)`, which maps a given coordinate pair
    :math:`\left(u_{x},u_{y}\right)` to a corresponding coordinate pair
    :math:`\left(q_{x},q_{y}\right)`, and implicitly a right-inverse to said
    coordinate transformation,
    :math:`\left(T_{\square;x}\left(q_{x},q_{y}\right),
    T_{\square;y}\left(q_{x},q_{y}\right)\right)`, that maps a coordinate pair
    :math:`\left(q_{x},q_{y}\right)` to a corresponding coordinate pair
    :math:`\left(u_{x},u_{y}\right)`, when such a relationship exists for
    :math:`\left(q_{x},q_{y}\right)`.

    The calculation of the image of the target fake CBED pattern involves
    calculating various intermediate images which are subsequently combined to
    yield the target image. These intermediate images share the same horizontal
    and vertical dimensions in units of pixels, which may differ from those of
    the image of the target fake CBED pattern. Let :math:`N_{\mathcal{I};x}` and
    :math:`N_{\mathcal{I};y}` be the number of pixels in the image of the target
    fake CBED pattern from left to right and top to bottom respectively, and let
    :math:`N_{\mathring{\mathcal{I}};x}` and
    :math:`N_{\mathring{\mathcal{I}};y}` be the number of pixels in each of the
    aforementioned intermediate images from left to right and top to bottom
    respectively. In :mod:`fakecbed`, we assume that

    .. math ::
        N_{\mathcal{I};x}=N_{\mathcal{I};y},
        :label: N_I_x_eq_N_I_y__1

    .. math ::
        N_{\mathring{\mathcal{I}};x}\ge N_{\mathcal{I};x},
        :label: N_ring_I_x_ge_N_I_x__1

    and

    .. math ::
        N_{\mathring{\mathcal{I}};y}\ge N_{\mathcal{I};y}.
        :label: N_ring_I_y_ge_N_I_y__1

    The integer :math:`N_{\mathcal{I};x}` is specified by the parameter
    ``num_pixels_across_pattern``. The integers
    :math:`N_{\mathring{\mathcal{I}};x}` and
    :math:`N_{\mathring{\mathcal{I}};y}` are specified indirectly by the
    parameter ``distortion_model``. The parameter ``distortion_model`` specifies
    the distortion model, which as mentioned above is represented by an instance
    of the class :class:`distoptica.DistortionModel`. One of the parameters of
    said distortion model is the integer pair
    ``sampling_grid_dims_in_pixels``. In the current context,
    ``sampling_grid_dims_in_pixels[0]`` and ``sampling_grid_dims_in_pixels[1]``
    are equal to :math:`N_{\mathring{\mathcal{I}};x}` and
    :math:`N_{\mathring{\mathcal{I}};y}` respectively.

    As mentioned above, a set of intensity patterns need to be specified in
    order to create the target fake CBED pattern. The first of these is the
    intensity pattern of an undistorted thermal diffuse scattering (TDS) model,
    :math:`\mathcal{I}_{\text{TDS}}\left(u_{x},u_{y}\right)`, which is specified
    by the parameter ``undistorted_tds_model``. The second of these intensity
    patterns is that of the undistorted outer illumination shape,
    :math:`\mathcal{I}_{\text{OI}}\left(u_{x},u_{y}\right)`, which is specified
    by the parameter ``undistorted_outer_illumination_shape``.
    :math:`\mathcal{I}_{\text{OI}}\left(u_{x},u_{y}\right)` is defined such that
    for every coordinate pair :math:`\left(u_{x},u_{y}\right)`, if
    :math:`\mathcal{I}_{\text{OI}}\left(u_{x},u_{y}\right)=0` then the value of
    the UNNBU fake CBED pattern is also equal to 0. A separate subset of the
    intensity patterns that need to be specified are :math:`N_{\text{D}}`
    intensity patterns of undistorted nonuniform circles and/or ellipses,
    :math:`\left\{ \mathcal{I}_{k;\text{D}}\left(u_{x},u_{y}\right)\right\}
    _{k=0}^{N_{\text{D}}-1}`, which is specified by the parameter
    ``undistorted_disks``. Each intensity pattern
    :math:`\mathcal{I}_{k;\text{D}}\left(u_{x},u_{y}\right)` is suppose to
    depict one of the CBED disks in the fake CBED pattern in the absence of the
    intensity background. Moreover, each intensity pattern
    :math:`\mathcal{I}_{k;\text{D}}\left(u_{x},u_{y}\right)` has a corresponding
    supporting intensity pattern
    :math:`\mathcal{I}_{k;\text{DS}}\left(u_{x},u_{y}\right)`, which is defined
    such that for every coordinate pair :math:`\left(u_{x},u_{y}\right)`, if
    :math:`\mathcal{I}_{k;\text{DS}}\left(u_{x},u_{y}\right)=0` then
    :math:`\mathcal{I}_{k;\text{D}}\left(u_{x},u_{y}\right)=0`. The remaining
    intensity patterns that need to be specified, of which there are
    :math:`N_{\text{M}}`, are intensity patterns of undistorted miscellaneous
    shapes, :math:`\left\{
    \mathcal{I}_{k;\text{M}}\left(u_{x},u_{y}\right)\right\}
    _{k=0}^{N_{\text{M}}-1}`, which is specified by the parameter
    ``undistorted_misc_shapes``. These patterns, along with that of the TDS
    model, contribute to the intensity background.

    To add blur effects, users can specify a nonzero standard deviation
    :math:`\sigma_{\text{blur}}` of the Gaussian filter used to yield such blur
    effects on the target fake CBED pattern. The value of
    :math:`\sigma_{\text{blur}}` is specified by the parameter
    ``gaussian_filter_std_dev``.

    To add shot noise effects to the image of the target fake CBED pattern, the
    parameter ``apply_shot_noise`` needs to be set to ``True``.

    For some pixelated electron detectors, the pixels of a number
    :math:`N_{\text{DPW}}` of contiguous rows and an equal number of contiguous
    columns will not measure or readout incident electron counts. Instead, the
    final intensity values measured are inpainted according to the final
    intensity values of the other pixels in the detector. The intersection of
    the aforementioned contiguous block of rows and the aforementioned
    contiguous block of columns is located within one pixel of the center of the
    detector. The integer :math:`N_{\text{DPW}}`, which we call the detector
    partition width in units of pixels, is specified by the parameter
    ``detector_partition_width_in_pixels``.

    Cold pixels, which are individual zero-valued pixels in the image of the
    target fake CBED pattern, are specified by the parameter
    ``cold_pixels``. Let :math:`N_{\text{CP}}` be the number of cold pixels in
    the image of the target fake CBED pattern. Furthermore, let :math:`\left\{
    n_{k;\text{CP}}\right\} _{k=0}^{N_{\text{CP}}-1}` and :math:`\left\{
    m_{k;\text{CP}}\right\} _{k=0}^{N_{\text{CP}}-1}` be integer sequences
    respectively, where :math:`n_{k;\text{CP}}` and :math:`m_{k;\text{CP}}` are
    the row and column indices respectively of the :math:`k^{\text{th}}` cold
    pixel. For every nonnegative integer ``k`` less than :math:`N_{\text{CP}}`,
    ``cold_pixels[k][0]`` and ````cold_pixels[k][1]`` are
    :math:`n_{k;\text{CP}}` and :math:`m_{k;\text{CP}}` respectively, with the
    integer :math:`k` being equal to the value of ``k``.

    The mask frame of the image of the target fake CBED pattern is specified by
    the parameter ``mask_frame``, which is expected to be a 4-element tuple,
    :math:`\left(L, R, B, T\right)`, of nonnegative integers. ``mask_frame[0]``,
    ``mask_frame[1]``, ``mask_frame[2]``, and ``mask_frame[3]`` are the widths,
    in units of pixels, of the left, right, bottom, and top sides of the mask
    frame respectively. If all elements of ``mask_frame`` are zero, then no
    pixels in the image of the target fake CBED pattern are masked by the mask
    frame. 

    Below we describe in more detail how various attributes of the current class
    are effectively calculated. Before doing so, we need to introduce a few more
    quantities:

    .. math ::
        j\in\left\{ j^{\prime}\right\}_{j^{\prime}=0}^{
        N_{\mathcal{\mathring{I}};x}-1},
        :label: j_range__1

    .. math ::
        i\in\left\{ i^{\prime}\right\} _{i^{\prime}=0}^{
        N_{\mathcal{\mathring{I}};y}-1}
        :label: i_range__1

    .. math ::
        q_{\mathcal{\mathring{I}};x;j}=\left(j+\frac{1}{2}\right)
        \Delta q_{\mathcal{\mathring{I}};x},
        :label: q_I_circ_x_j__1

    .. math ::
        q_{\mathcal{\mathring{I}};y;i}=1-\left(i+\frac{1}{2}\right)
        \Delta q_{\mathcal{\mathring{I}};y},
        :label: q_I_circ_y_i__1

    .. math ::
        \Delta q_{\mathcal{\mathring{I}};x}=
        \frac{1}{N_{\mathcal{\mathring{I}};x}},
        :label: Delta_q_I_circ_x__1

    .. math ::
        \Delta q_{\mathcal{\mathring{I}};y}=
        \frac{1}{N_{\mathcal{\mathring{I}};y}}.
        :label: Delta_q_I_circ_y__1

    .. math ::
        m\in\left\{ m^{\prime}\right\} _{m^{\prime}=0}^{N_{\mathcal{I};x}-1},
        :label: m_range__1

    .. math ::
        n\in\left\{ n^{\prime}\right\} _{n^{\prime}=0}^{N_{\mathcal{I};y}-1},
        :label: n_range__1

    and

    .. math ::
        \mathbf{J}_{\square}\left(q_{x},q_{y}\right)=
        \begin{pmatrix}\frac{\partial T_{\square;x}}{\partial q_{x}} 
        & \frac{\partial T_{\square;x}}{\partial q_{y}}\\
        \frac{\partial T_{\square;y}}{\partial q_{x}} 
        & \frac{\partial T_{\square;y}}{\partial q_{y}}
        \end{pmatrix},
        :label: J_sq__1

    where the derivatives in Eq. :eq:`J_sq__1` are calculated numerically using
    the second-order accurate central differences method. The aforementioned
    attributes of the current class are effectively calculated by executing the
    following steps:

    1. Calculate

    .. math ::
        \mathring{\mathcal{I}}_{\text{OI};⌑;i,j}\leftarrow
        \mathcal{I}_{\text{OI}}\left(
        T_{\square;x}\left(q_{\mathring{\mathcal{I}};x;j},
        q_{\mathring{\mathcal{I}};y;i}\right),
        T_{\square;y}\left(q_{\mathring{\mathcal{I}};x;j},
        q_{\mathring{\mathcal{I}};y;i}\right)\right),
        :label: HD_I_OI__1

    .. math ::
        \mathring{\mathcal{I}}_{\text{OI};⌑;i,j}\leftarrow\begin{cases}
        \text{True}, & \text{if }\mathring{\mathcal{I}}_{\text{OI};⌑;i,j}
        \neq0,\\
        \text{False}, & \text{otherwise},
        \end{cases}
        :label: HD_I_OI__2

    and then apply max pooling to
    :math:`\mathring{\mathcal{I}}_{\text{OI};⌑;i,j}` with a kernel of dimensions
    :math:`\left(N_{\mathring{\mathcal{I}};y}/N_{\mathcal{I};y},
    N_{\mathring{\mathcal{I}};x}/N_{\mathcal{I};x}\right)`
    and store the result in :math:`\mathcal{I}_{\text{OI};⌑;n,m}`.

    2. Calculate

    .. math ::
        \mathcal{I}_{\text{DOM};⌑;n,m}\leftarrow0.
        :label: LD_I_DOM__1

    3. Calculate

    .. math ::
        \mathcal{I}_{\text{MF};⌑;n,m}\leftarrow\begin{cases}
        \text{True}, & \text{if }L\le m<N_{\mathcal{I};x}-R
        \text{ and }T\le n<N_{\mathcal{I};y}-B,\\
        \text{False}, & \text{otherwise}.
        \end{cases}
        :label: LD_I_MF__1

    4. For :math:`0\le k<N_{\text{D}}`, calculate

    .. math ::
        \mathring{\mathcal{I}}_{\text{CBED};⌑;i,j}&
        \leftarrow\mathring{\mathcal{I}}_{\text{CBED};⌑;i,j}\\&
        \quad\quad\mathop{+}\mathcal{I}_{k;\text{D}}\left(T_{\square;x}\left(
        q_{\mathring{\mathcal{I}};x;j},q_{\mathring{\mathcal{I}};y;i}\right),
        T_{\square;y}\left(q_{\mathring{\mathcal{I}};x;j},
        q_{\mathring{\mathcal{I}};y;i}\right)\right),
        :label: HD_I_CBED__1

    .. math ::
        \mathring{\mathcal{I}}_{k;\text{DS};⌑;i,j}\leftarrow
        \mathcal{I}_{k;\text{DS}}\left(T_{\square;x}\left(
        q_{\mathring{\mathcal{I}};x;j},q_{\mathring{\mathcal{I}};y;i}\right),
        T_{\square;y}\left(q_{\mathring{\mathcal{I}};x;j},
        q_{\mathring{\mathcal{I}};y;i}\right)\right),
        :label: HD_I_k_DS__1

    .. math ::
        \mathring{\mathcal{I}}_{k;\text{DS};⌑;i,j}\leftarrow\begin{cases}
        \text{True}, & \text{if }\mathring{\mathcal{I}}_{k;\text{DS};⌑;i,j}
        \neq0,\\
        \text{False}, & \text{otherwise},
        \end{cases}
        :label: HD_I_k_DS__2

    then apply max pooling to :math:`\mathring{\mathcal{I}}_{k;\text{DS};⌑;i,j}`
    with a kernel of dimensions
    :math:`\left(N_{\mathring{\mathcal{I}};y}/N_{\mathcal{I};y},
    N_{\mathring{\mathcal{I}};x}/N_{\mathcal{I};x}\right)` and store the result
    in :math:`\mathcal{I}_{k;\text{DS};⌑;n,m}`, and calculate

    .. math ::
        \mathcal{I}_{\text{DOM};⌑;n,m}\leftarrow
        \mathcal{I}_{\text{DOM};⌑;n,m}+\mathcal{I}_{k;\text{DS};⌑;n,m}.
        :label: LD_I_DOM__2

    5. Calculate

    .. math ::
        \mathcal{I}_{\text{DOM};⌑;n,m}\leftarrow
        \mathcal{I}_{\text{MF};⌑;n,m}
        \mathcal{I}_{\text{OI};⌑;n,m}
        \mathcal{I}_{\text{DOM};⌑;n,m}.
        :label: LD_I_DOM__3

    6. For :math:`0\le k<N_{\text{M}}`, calculate

    .. math ::
        \mathring{\mathcal{I}}_{\text{CBED};⌑;i,j}&
        \leftarrow\mathring{\mathcal{I}}_{\text{CBED};⌑;i,j}\\&
        \quad\quad\mathop{+}\mathcal{I}_{k;\text{M}}\left(T_{\square;x}\left(
        q_{\mathring{\mathcal{I}};x;j},q_{\mathring{\mathcal{I}};y;i}\right),
        T_{\square;y}\left(q_{\mathring{\mathcal{I}};x;j},
        q_{\mathring{\mathcal{I}};y;i}\right)\right).
        :label: HD_I_CBED__2

    7. Calculate

    .. math ::
        \mathring{\mathcal{I}}_{\text{CBED};⌑;i,j}\leftarrow
        \text{det}\left(\mathbf{J}_{\square}\left(
        q_{\mathring{\mathcal{I}};x;j},
        q_{\mathring{\mathcal{I}};y;i}\right)\right)
        \left|\mathring{\mathcal{I}}_{\text{CBED};⌑;i,j}\right|.
        :label: HD_I_CBED__3

    8. Apply average pooling to
    :math:`\mathring{\mathcal{I}}_{\text{CBED};⌑;i,j}` with a kernel of
    dimensions :math:`\left(N_{\mathring{\mathcal{I}};y}/N_{\mathcal{I};y},
    N_{\mathring{\mathcal{I}};x}/N_{\mathcal{I};x}\right)`, and store the result
    in :math:`\mathcal{I}_{\text{CBED};⌑;n,m}`.

    9. Apply a Gaussian filter to :math:`\mathcal{I}_{\text{CBED};⌑;n,m}` that
    is identical in outcome to that implemented by the function
    :func:`scipy.ndimage.gaussian_filter`, with ``sigma`` set to
    ``gaussian_filter_std_dev`` and ``truncate`` set to ``4``, and store the
    result in :math:`\mathcal{I}_{\text{CBED};⌑;n,m}`.

    10. Calculate

    .. math ::
        k_{\text{I};1}\leftarrow
        \left\lfloor \frac{N_{\mathcal{I};x}-1}{2}\right\rfloor 
        -\left\lfloor \frac{N_{\text{DPW}}}{2}\right\rfloor ,
        :label: k_I_1__1

    and

    .. math ::
        k_{\text{I};2}\leftarrow k_{\text{I};1}+N_{\text{DPW}}-1.
        :label: k_I_2__1

    11. If ``apply_shot_noise`` is set to ``True``, then apply shot/Poisson
    noise to :math:`\mathcal{I}_{\text{CBED};⌑;n,m}`, and store the result in
    :math:`\mathcal{I}_{\text{CBED};⌑;n,m}`.

    12. If :math:`N_{\text{DPW}}>0`, then inpaint the pixels in the rows indexed
    from :math:`k_{\text{I};1}` to :math:`k_{\text{I};2}` and the columns
    indexed from :math:`k_{\text{I};1}` to :math:`k_{\text{I};2}` of the image
    :math:`\mathcal{I}_{\text{CBED};⌑;n,m}` using the function
    :func:`skimage.restoration.inpaint_biharmonic`, and store the result in
    :math:`\mathcal{I}_{\text{CBED};⌑;n,m}`.

    13. Calculate

    .. math ::
        \mathcal{I}_{\text{CBED};⌑;n,m}\leftarrow
        \mathcal{I}_{\text{MF};⌑;n,m}
        \mathcal{I}_{\text{OI};⌑;n,m}
        \mathcal{I}_{\text{CBED};⌑;n,m}.
        :label: LD_I_CBED__1

    14. Update pixels of :math:`\mathcal{I}_{\text{CBED};⌑;n,m}` at pixel
    locations specified by ``cold_pixels`` to the value of zero.

    15. Apply min-max normalization of :math:`\mathcal{I}_{\text{CBED};⌑;n,m}`,
    and store result in :math:`\mathcal{I}_{\text{CBED};⌑;n,m}`.

    16. Calculate

    .. math ::
        \mathcal{I}_{\text{CS};⌑;n,m}\leftarrow1
        -\mathcal{I}_{\text{MF};⌑;n,m}\mathcal{I}_{\text{OI};⌑;n,m}.
        :label: LD_I_CS__1

    17. Convolve a :math:`3 \times 3` filter of ones over a symmetrically unity-padded
    :math:`\mathcal{I}_{\text{CS};⌑;n,m}` to yield an output matrix with the
    same dimensions of :math:`\mathcal{I}_{\text{CBED};⌑;n,m}`, and store said
    output matrix in :math:`\mathcal{I}_{\text{CS};⌑;n,m}`.

    18. For :math:`0\le k<N_{\text{D}}`, calculate

    .. math ::
        \mathcal{I}_{k;\text{DCM};⌑;n,m}\leftarrow
        \mathcal{I}_{\text{CS};⌑;n,m}\mathcal{I}_{k;\text{DS};⌑;n,m},
        :label: LD_I_DCM__1

    .. math ::
        \Omega_{k;\text{DCR};⌑}\leftarrow\begin{cases}
        \text{True}, & \text{if }
        \left(\sum_{n,m}\mathcal{I}_{k;\text{DCM};⌑;n,m}\neq0\right)\bigparallel
        \left(\sum_{n,m}\mathcal{I}_{k;\text{DS};⌑;n,m}=0\right),\\
        \text{False}, & \text{otherwise},
        \end{cases}
        :label: Omega_k_DCR__1

    and

    .. math ::
        \Omega_{k;\text{DAR};⌑}\leftarrow\begin{cases}
        \text{True}, & \text{if }\sum_{n,m}\mathcal{I}_{k;\text{DS};⌑;n,m}=0,\\
        \text{False}, & \text{otherwise}.
        \end{cases}
        :label: Omega_k_DAR__1

    We refer to :math:`\mathcal{I}_{\text{CBED};⌑;n,m}` as the image of the
    target fake CBED pattern, :math:`\mathcal{I}_{\text{OI};⌑;n,m}` as the image
    of the illumination support, :math:`\mathcal{I}_{\text{DOM};⌑;n,m}` as the
    image of the disk overlap map, :math:`\mathcal{I}_{k;\text{DS};⌑;n,m}` as
    the image of the support of the :math:`k^{\text{th}}` CBED disk,
    :math:`\left\{ \Omega_{k;\text{DCR};⌑}\right\}_{k=0}^{N_{\text{D}}-1}` as
    the disk clipping registry, and :math:`\left\{
    \Omega_{k;\text{DAR};⌑}\right\}_{k=0}^{N_{\text{D}}-1}` as the disk absence
    registry.

    Parameters
    ----------
    undistorted_tds_model : :class:`fakecbed.tds.Model` | `None`, optional
        The intensity pattern of the undistorted TDS model,
        :math:`\mathcal{I}_{\text{TDS}}\left(u_{x},u_{y}\right)`. If
        ``undistorted_tds_model`` is set to ``None``, then the parameter will be
        reassigned to the value ``fakecbed.tds.Model()``.
    undistorted_disks : `array_like` (:class:`fakecbed.shapes.NonuniformBoundedShape`, ndim=1), optional
        The intensity patterns of the undistorted fake CBED disks,
        :math:`\left\{ \mathcal{I}_{k;\text{D}}\left(u_{x},u_{y}\right)\right\}
        _{k=0}^{N_{\text{D}}-1}`. For every nonnegative integer ``k`` less than
        :math:`N_{\text{D}}`, ``undistorted_disks[k]`` is
        :math:`\mathcal{I}_{k;\text{D}}\left(u_{x},u_{y}\right)`, with the
        integer :math:`k` being equal to the value of ``k``
    undistorted_misc_shapes : `array_like` (`any_shape`, ndim=1), optional
        The intensity patterns of the undistorted miscellaneous shapes,
        :math:`\left\{ \mathcal{I}_{k;\text{M}}\left(u_{x},u_{y}\right)\right\}
        _{k=0}^{N_{\text{M}}-1}`. Note that `any_shape` means any public class
        defined in the module :mod:`fakecbed.shapes` that is a subclass of
        :class:`fakecbed.shapes.BaseShape`.
    undistorted_outer_illumination_shape : :class:`fakecbed.shapes.Circle` | :class:`fakecbed.shapes.Ellipse` | :class:`fakecbed.shapes.GenericBlob` | `None`, optional
        The intensity pattern of the undistorted outer illumination shape,
        :math:`\mathcal{I}_{\text{OI}}\left(u_{x},u_{y}\right)`. If
        ``undistorted_outer_illumination_shape`` is set to ``None``, then
        :math:`\mathcal{I}_{\text{OI}}\left(u_{x},u_{y}\right)` will equal unity
        for all :math:`u_{x}` and :math:`u_{y}`.
    gaussian_filter_std_dev : `float`, optional
        The standard deviation :math:`\sigma_{\text{blur}}` of the Gaussian
        filter used to yield such blur effects on the target fake CBED
        pattern. Must be nonnegative.
    num_pixels_across_pattern : `int`, optional
        The number of pixels across the image of the fake CBED pattern,
        :math:`N_{\mathcal{I};x}`. Must be positive.
    distortion_model : :class:`distoptica.DistortionModel` | `None`, optional
        The distortion model. If ``distortion_model`` is set to ``None``, then
        the parameter will be reassigned to the value
        ``distoptica.DistortionModel(sampling_grid_dims_in_pixels=(N_x, N_x))``,
        Where ``N_x`` is equal to ``num_pixels_across_pattern``.
    apply_shot_noise : `bool`, optional
        If ``apply_shot_noise`` is set to ``True``, then shot noise is applied
        to the image of the fake CBED pattern. Otherwise, no shot noise is
        applied.
    rng_seed : `int` | `None`, optional
        ``rng_seed`` specifies the seed used in the random number generator used
        to apply shot noise.
    detector_partition_width_in_pixels : `int`, optional
        The detector partition width in units of pixels,
        :math:`N_{\text{DPW}}`. Must be nonnegative.
    cold_pixels : `array_like` (`int`, ndim=2), optional
        The pixel coordinates of the cold pixels.
    mask_frame : `array_like` (`int`, shape=(4,)), optional
        ``mask_frame`` specifies the mask frame of the image of the target fake
        CBED pattern. ``mask_frame[0]``, ``mask_frame[1]``, ``mask_frame[2]``,
        and ``mask_frame[3]`` are the widths, in units of pixels, of the left,
        right, bottom, and top sides of the mask frame respectively. If all
        elements of ``mask_frame`` are zero, then no pixels in the image of the
        target fake CBED pattern are masked by the mask frame.
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
    ctor_param_names = ("undistorted_tds_model",
                        "undistorted_disks",
                        "undistorted_misc_shapes",
                        "undistorted_outer_illumination_shape",
                        "gaussian_filter_std_dev",
                        "num_pixels_across_pattern",
                        "distortion_model",
                        "apply_shot_noise",
                        "rng_seed",
                        "detector_partition_width_in_pixels",
                        "cold_pixels",
                        "mask_frame")
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
                 undistorted_tds_model=\
                 _default_undistorted_tds_model,
                 undistorted_disks=\
                 _default_undistorted_disks,
                 undistorted_misc_shapes=\
                 _default_undistorted_misc_shapes,
                 undistorted_outer_illumination_shape=\
                 _default_undistorted_outer_illumination_shape,
                 gaussian_filter_std_dev=\
                 _default_gaussian_filter_std_dev,
                 num_pixels_across_pattern=\
                 _default_num_pixels_across_pattern,
                 distortion_model=\
                 _default_distortion_model,
                 apply_shot_noise=\
                 _default_apply_shot_noise,
                 rng_seed=\
                 _default_rng_seed,
                 detector_partition_width_in_pixels=\
                 _default_detector_partition_width_in_pixels,
                 cold_pixels=\
                 _default_cold_pixels,
                 mask_frame=\
                 _default_mask_frame,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        kwargs = ctor_params
        kwargs["skip_cls_tests"] = True
        fancytypes.PreSerializableAndUpdatable.__init__(self, **kwargs)

        self.execute_post_core_attrs_update_actions()

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



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)
        
        self._num_disks = len(self._undistorted_disks)
        self._device = self._distortion_model.device

        self._illumination_support = None
        self._image = None
        self._image_has_been_overridden = False        
        self._signal = None
        self._disk_clipping_registry = None
        self._disk_supports = None
        self._disk_absence_registry = None
        self._disk_overlap_map = None

        return None



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    @property
    def num_disks(self):
        r"""`int`: The total number of CBED disks defined, :math:`N_{\text{D}}`.

        See the summary documentation for the class
        :class:`fakecbed.discretized.CBEDPattern` for additional context.

        Let ``core_attrs`` denote the attribute
        :attr:`~fancytypes.Checkable.core_attrs`. ``num_disks`` is equal to
        ``len(core_attrs["undistorted_disks"])``.

        Note that ``num_disks`` should be considered **read-only**.

        """
        result = self._num_disks
        
        return result



    @property
    def device(self):
        r"""`torch.device`: The device on which computationally intensive 
        PyTorch operations are performed and attributes of the type 
        :class:`torch.Tensor` are stored.

        Note that ``device`` should be considered **read-only**.

        """
        result = copy.deepcopy(self._device)

        return result



    def override_image_then_reapply_mask(
            self,
            overriding_image,
            skip_validation_and_conversion=\
            _default_skip_validation_and_conversion):
        r"""Override the target fake CBED pattern image and reapply masking.

        See the summary documentation for the class
        :class:`fakecbed.discretized.CBEDPattern` for additional context.

        Let ``image``, ``illumination_support``, and ``core_attrs`` denote the
        attributes :attr:`fakecbed.discretized.CBEDPattern.image`,
        :attr:`fakecbed.discretized.CBEDPattern.illumination_support`, and
        :attr:`~fancytypes.Checkable.core_attrs`. ``overriding_image`` is the
        overriding image.

        Upon calling the method ``override_image_then_reapply_mask``, the
        attribute ``image`` is updated effectively by:

        .. code-block:: python

           coords_of_cold_pixels = core_attrs["cold_pixels"]
           L, R, B, T = core_attrs["mask_frame"]
           N_I_x = core_attrs["num_pixels_across_pattern"]
           N_I_y = N_I_x

           image = (overriding_image * illumination_support).clip(min=0)
           image[:T, :] = 0
           image[max(N_I_y-B, 0):, :] = 0
           image[:, :L] = 0
           image[:, max(N_I_x-R, 0):] = 0
           for coords_of_cold_pixel in coords_of_cold_pixels:
               image[coords_of_cold_pixel] = 0

        and then finally min-max normalization is applied to ``image``.

        Parameters
        ----------
        overriding_image : `array_like` (`float`, shape=image.shape)
            The overriding image.
        skip_validation_and_conversion : `bool`, optional
            If ``skip_validation_and_conversion`` is set to ``False``, then
            validations and conversions are performed on the above parameters.

            Otherwise, if ``skip_validation_and_conversion`` is set to ``True``,
            no validations and conversions are performed on the above
            parameters. This option is desired primarily when the user wants to
            avoid potentially expensive validation and/or conversion operations.

        """
        params = locals()
        
        func_alias = _check_and_convert_skip_validation_and_conversion
        skip_validation_and_conversion = func_alias(params)

        if (skip_validation_and_conversion == False):
            params = {"overriding_image": \
                      overriding_image,
                      "num_pixels_across_pattern": \
                      self._num_pixels_across_pattern,
                      "device": \
                      self._device}
            overriding_image = _check_and_convert_overriding_image(params)
        
        self._override_image_then_reapply_mask(overriding_image)

        return None



    def _override_image_then_reapply_mask(self, overriding_image):
        if self._illumination_support is None:
            u_x, u_y = self._calc_u_x_and_u_y()
            method_name = "_calc_illumination_support"
            method_alias = getattr(self, method_name)
            self._illumination_support = method_alias(u_x, u_y)
        illumination_support = self._illumination_support

        coords_of_cold_pixels = self._cold_pixels
        L, R, B, T = self._mask_frame
        N_I_x = self._num_pixels_across_pattern
        N_I_y = N_I_x

        image = overriding_image*illumination_support
        image[:T, :] = 0
        image[max(N_I_y-B, 0):, :] = 0
        image[:, :L] = 0
        image[:, max(N_I_x-R, 0):] = 0
        for coords_of_cold_pixel in coords_of_cold_pixels:
            image[coords_of_cold_pixel] = 0

        kwargs = {"input_matrix": image}
        image = self._normalize_matrix(**kwargs)

        self._image = image
        self._image_has_been_overridden = True

        if self._signal is not None:
            self._signal.data[0] = image.numpy(force=True)

        return None



    def _calc_u_x_and_u_y(self):
        distortion_model = self._distortion_model

        method_alias = distortion_model.get_sampling_grid
        sampling_grid = method_alias(deep_copy=False)

        try:
            method_alias = \
                distortion_model.get_flow_field_of_coord_transform_right_inverse
            flow_field_of_coord_transform_right_inverse = \
                method_alias(deep_copy=False)
        except:
            err_msg = _cbed_pattern_err_msg_1
            raise RuntimeError(err_msg)

        u_x = sampling_grid[0] + flow_field_of_coord_transform_right_inverse[0]
        u_y = sampling_grid[1] + flow_field_of_coord_transform_right_inverse[1]

        return u_x, u_y



    def _calc_illumination_support(self, u_x, u_y):
        shape = self._undistorted_outer_illumination_shape

        pooler_kernel_size = self._calc_pooler_kernel_size()
        pooler = torch.nn.MaxPool2d(kernel_size=pooler_kernel_size)

        illumination_support = (shape._eval(u_x, u_y) != 0)
        illumination_support = torch.unsqueeze(illumination_support, dim=0)
        illumination_support = torch.unsqueeze(illumination_support, dim=0)
        illumination_support = illumination_support.to(dtype=u_x.dtype)
        illumination_support = pooler(illumination_support)[0, 0]
        illumination_support = illumination_support.to(dtype=torch.bool)

        return illumination_support



    def _calc_pooler_kernel_size(self):
        distortion_model = self._distortion_model
        num_pixels_across_pattern = self._num_pixels_across_pattern

        distortion_model_core_attrs = \
            distortion_model.get_core_attrs(deep_copy=False)
        sampling_grid_dims_in_pixels = \
            distortion_model_core_attrs["sampling_grid_dims_in_pixels"]

        pooler_kernel_size = (sampling_grid_dims_in_pixels[1]
                              // num_pixels_across_pattern,
                              sampling_grid_dims_in_pixels[0]
                              // num_pixels_across_pattern)

        return pooler_kernel_size



    def _normalize_matrix(self, input_matrix):
        if input_matrix.max()-input_matrix.min() > 0:
            normalization_weight = 1 / (input_matrix.max()-input_matrix.min())
            normalization_bias = -normalization_weight*input_matrix.min()
            output_matrix = (input_matrix*normalization_weight
                             + normalization_bias).clip(min=0, max=1)
        else:
            output_matrix = torch.zeros_like(input_matrix)

        return output_matrix



    def get_signal(self, deep_copy=_default_deep_copy):
        r"""Return the hyperspy signal representation of the fake CBED pattern.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``signal`` denote the attribute
            :attr:`fakecbed.discretized.CBEDPattern.signal`.

            If ``deep_copy`` is set to ``True``, then a deep copy of ``signal``
            is returned.  Otherwise, a reference to ``signal`` is returned.

        Returns
        -------
        signal : :class:`hyperspy._signals.signal2d.Signal2D`
            The attribute :attr:`fakecbed.discretized.CBEDPattern.signal`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._signal is None:
            u_x, u_y = self._calc_u_x_and_u_y()
            method_name = "_calc_signal_and_cache_select_intermediates"
            method_alias = getattr(self, method_name)
            self._signal = method_alias(u_x, u_y)

        signal = (copy.deepcopy(self._signal)
                  if (deep_copy == True)
                  else self._signal)

        return signal



    def _calc_signal_and_cache_select_intermediates(self, u_x, u_y):
        method_name = "_calc_signal_metadata_and_cache_select_intermediates"
        method_alias = getattr(self, method_name)
        signal_metadata = method_alias(u_x, u_y)

        if self._image is None:
            method_name = "_calc_image_and_cache_select_intermediates"
            method_alias = getattr(self, method_name)
            self._image = method_alias(u_x, u_y)
        image = self._image.numpy(force=True)
        
        if self._disk_overlap_map is None:
            method_name = ("_calc_disk_overlap_map"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._disk_overlap_map = method_alias(u_x, u_y)
        disk_overlap_map = self._disk_overlap_map.numpy(force=True)

        illumination_support = self._illumination_support.cpu().detach().clone()
        illumination_support = illumination_support.numpy(force=True)

        if self._disk_supports is None:
            method_name = ("_calc_disk_supports"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._disk_supports = method_alias(u_x, u_y)
        disk_supports = self._disk_supports.numpy(force=True)

        num_disks = self._num_disks
        L, R, B, T = self._mask_frame
        N_I_x = self._num_pixels_across_pattern
        N_I_y = N_I_x

        signal_data_shape = (num_disks+3,) + image.shape

        signal_data = np.zeros(signal_data_shape, dtype=image.dtype)
        signal_data[0] = image
        signal_data[1] = illumination_support
        signal_data[1, :T, :] = 0
        signal_data[1, max(N_I_y-B, 0):, :] = 0
        signal_data[1, :, :L] = 0
        signal_data[1, :, max(N_I_x-R, 0):] = 0
        signal_data[2] = disk_overlap_map
        signal_data[3:] = disk_supports

        signal = hyperspy.signals.Signal2D(data=signal_data,
                                           metadata=signal_metadata)
        self._update_signal_axes(signal)

        return signal



    def _calc_signal_metadata_and_cache_select_intermediates(self, u_x, u_y):
        distortion_model = self._distortion_model

        title = ("Fake Undistorted CBED Intensity Pattern"
                 if distortion_model.is_trivial
                 else "Fake Distorted CBED Intensity Pattern")
        
        pre_serialized_core_attrs = self.pre_serialize()

        if self._disk_clipping_registry is None:
            method_name = ("_calc_disk_clipping_registry"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._disk_clipping_registry = method_alias(u_x, u_y)
        disk_clipping_registry = self._disk_clipping_registry.cpu()
        disk_clipping_registry = disk_clipping_registry.detach().clone()
        disk_clipping_registry = tuple(disk_clipping_registry.tolist())

        if self._disk_absence_registry is None:
            method_name = ("_calc_disk_absence_registry"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._disk_absence_registry = method_alias(u_x, u_y)
        disk_absence_registry = self._disk_absence_registry.cpu()
        disk_absence_registry = disk_absence_registry.detach().clone()
        disk_absence_registry = tuple(disk_absence_registry.tolist())

        fakecbed_metadata = {"num_disks": \
                             self._num_disks,
                             "disk_clipping_registry": \
                             disk_clipping_registry,
                             "disk_absence_registry": \
                             disk_absence_registry,
                             "pre_serialized_core_attrs": \
                             pre_serialized_core_attrs,
                             "cbed_pattern_image_has_been_overridden": \
                             self._image_has_been_overridden}
        
        signal_metadata = {"General": {"title": title},
                           "Signal": {"pixel value units": "dimensionless"},
                           "FakeCBED": fakecbed_metadata}

        return signal_metadata



    def _calc_disk_clipping_registry_and_cache_select_intermediates(self,
                                                                    u_x,
                                                                    u_y):
        undistorted_disks = self._undistorted_disks
        num_disks = len(undistorted_disks)

        if num_disks > 0:
            if self._illumination_support is None:
                method_name = "_calc_illumination_support"
                method_alias = getattr(self, method_name)
                self._illumination_support = method_alias(u_x, u_y)
            illumination_support = self._illumination_support

            if self._disk_supports is None:
                method_name = ("_calc_disk_supports"
                               "_and_cache_select_intermediates")
                method_alias = getattr(self, method_name)
                self._disk_supports = method_alias(u_x, u_y)
            disk_supports = self._disk_supports

            clip_support = self._calc_clip_support(illumination_support)

            disk_clipping_map = disk_supports*clip_support[None, :, :]
    
            disk_clipping_registry = ((disk_clipping_map.sum(dim=(1, 2)) != 0)
                                      + (disk_supports.sum(dim=(1, 2)) == 0))
        else:
            disk_clipping_registry = torch.zeros((num_disks,),
                                                 device=u_x.device,
                                                 dtype=torch.bool)

        return disk_clipping_registry



    def _calc_disk_supports_and_cache_select_intermediates(self, u_x, u_y):
        undistorted_disks = self._undistorted_disks
        num_disks = len(undistorted_disks)

        if num_disks > 0:
            L, R, B, T = self._mask_frame
            N_I_x = self._num_pixels_across_pattern
            N_I_y = N_I_x

            if self._illumination_support is None:
                method_name = "_calc_illumination_support"
                method_alias = getattr(self, method_name)
                self._illumination_support = method_alias(u_x, u_y)
            illumination_support = self._illumination_support

            pooler_kernel_size = self._calc_pooler_kernel_size()
            pooler = torch.nn.MaxPool2d(kernel_size=pooler_kernel_size)

            disk_supports_shape = (num_disks,)+u_x.shape
    
            disk_supports = torch.zeros(disk_supports_shape,
                                        device=u_x.device)
            for disk_idx, undistorted_disk in enumerate(undistorted_disks):
                method_name = "_eval_without_intra_support_shapes"
                method_alias = getattr(undistorted_disk, method_name)
                disk_supports[disk_idx] = (method_alias(u_x, u_y) != 0)
            disk_supports = torch.unsqueeze(disk_supports, dim=0)
            disk_supports = disk_supports.to(dtype=u_x.dtype)
            disk_supports = pooler(disk_supports)[0]
            disk_supports = disk_supports.to(dtype=torch.bool)
            disk_supports[:, :, :] *= illumination_support[None, :, :]
            disk_supports[:, :T, :] = 0
            disk_supports[:, max(N_I_y-B, 0):, :] = 0
            disk_supports[:, :, :L] = 0
            disk_supports[:, :, max(N_I_x-R, 0):] = 0
        else:
            num_pixels_across_pattern = self._num_pixels_across_pattern
            
            disk_supports_shape = (num_disks,
                                   num_pixels_across_pattern,
                                   num_pixels_across_pattern)
            
            disk_supports = torch.zeros(disk_supports_shape,
                                        device=u_x.device,
                                        dtype=torch.bool)

        return disk_supports



    def _calc_clip_support(self, illumination_support):
        L, R, B, T = self._mask_frame
        N_I_x = self._num_pixels_across_pattern
        N_I_y = N_I_x

        clip_support = ~illumination_support
        for _ in range(2):
            clip_support = torch.unsqueeze(clip_support, dim=0)
        clip_support = clip_support.to(dtype=torch.float)

        conv_weights = torch.ones((1, 1, 3, 3),
                                  device=illumination_support.device)

        kwargs = {"input": clip_support,
                  "weight": conv_weights,
                  "padding": "same"}
        clip_support = (torch.nn.functional.conv2d(**kwargs) != 0)[0, 0]        
        clip_support[:T+1, :] = True
        clip_support[max(N_I_y-B-1, 0):, :] = True
        clip_support[:, :L+1] = True
        clip_support[:, max(N_I_x-R-1, 0):] = True

        return clip_support



    def _calc_disk_absence_registry_and_cache_select_intermediates(self,
                                                                   u_x,
                                                                   u_y):
        undistorted_disks = self._undistorted_disks
        num_disks = len(undistorted_disks)

        if num_disks > 0:
            if self._disk_supports is None:
                method_name = ("_calc_disk_supports"
                               "_and_cache_select_intermediates")
                method_alias = getattr(self, method_name)
                self._disk_supports = method_alias(u_x, u_y)
            disk_supports = self._disk_supports

            disk_absence_registry = (disk_supports.sum(dim=(1, 2)) == 0)
        else:
            disk_absence_registry = torch.zeros((num_disks,),
                                                 device=u_x.device,
                                                 dtype=torch.bool)
    
        return disk_absence_registry



    def _calc_image_and_cache_select_intermediates(self, u_x, u_y):
        method_name = ("_calc_maskless_and_noiseless_image"
                       "_and_cache_select_intermediates")
        method_alias = getattr(self, method_name)
        image = method_alias(u_x, u_y)

        apply_shot_noise = self._apply_shot_noise
        if apply_shot_noise == True:
            # ``torch.poisson`` was occasionally causing CUDA errors.
            rng = np.random.default_rng(self._rng_seed)
            image_dtype = image.dtype
            image = image.numpy(force=True).clip(min=0, max=1.0e15)
            image = rng.poisson(image)
            image = torch.from_numpy(image)
            image = image.to(device=self._device, dtype=image_dtype)

        image = self._apply_detector_partition_inpainting(input_image=image)
            
        if self._illumination_support is None:
            method_name = "_calc_illumination_support"
            method_alias = getattr(self, method_name)
            self._illumination_support = method_alias(u_x, u_y)
        illumination_support = self._illumination_support

        image = image*illumination_support

        coords_of_cold_pixels = self._cold_pixels
        L, R, B, T = self._mask_frame
        N_I_x = self._num_pixels_across_pattern
        N_I_y = N_I_x

        image[:T, :] = 0
        image[max(N_I_y-B, 0):, :] = 0
        image[:, :L] = 0
        image[:, max(N_I_x-R, 0):] = 0
        for coords_of_cold_pixel in coords_of_cold_pixels:
            image[coords_of_cold_pixel] = 0

        image = self._normalize_matrix(input_matrix=image)

        return image



    def _calc_maskless_and_noiseless_image_and_cache_select_intermediates(self,
                                                                          u_x,
                                                                          u_y):
        jacobian_weights = self._calc_jacobian_weights(u_x, u_y)

        bg = self._calc_bg(u_x, u_y, jacobian_weights)

        if self._disk_supports is None:
            method_name = ("_calc_disk_supports"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._disk_supports = method_alias(u_x, u_y)
        disk_supports = self._disk_supports

        intra_disk_shapes = self._calc_intra_disk_shapes(u_x,
                                                         u_y,
                                                         jacobian_weights)

        gaussian_filter_std_dev = self._gaussian_filter_std_dev

        maskless_and_noiseless_image = (bg
                                        + (disk_supports
                                           * intra_disk_shapes).sum(dim=0))

        maskless_and_noiseless_image = torch.abs(maskless_and_noiseless_image)

        kwargs = {"input_matrix": maskless_and_noiseless_image,
                  "truncate": 4}
        maskless_and_noiseless_image = self._apply_2d_guassian_filter(**kwargs)

        maskless_and_noiseless_image = torch.clip(maskless_and_noiseless_image,
                                                  min=0)
        
        return maskless_and_noiseless_image



    def _calc_jacobian_weights(self, u_x, u_y):
        distortion_model = self._distortion_model

        sampling_grid = distortion_model.get_sampling_grid(deep_copy=False)

        spacing = (sampling_grid[1][:, 0], sampling_grid[0][0, :])

        kwargs = {"input": u_x,
                  "spacing": spacing,
                  "dim": None,
                  "edge_order": 2}
        d_u_x_over_d_q_y, d_u_x_over_d_q_x = torch.gradient(**kwargs)

        kwargs["input"] = u_y
        d_u_y_over_d_q_y, d_u_y_over_d_q_x = torch.gradient(**kwargs)

        jacobian_weights = torch.abs(d_u_x_over_d_q_x*d_u_y_over_d_q_y
                                     - d_u_x_over_d_q_y*d_u_y_over_d_q_x)

        return jacobian_weights



    def _calc_bg(self, u_x, u_y, jacobian_weights):
        undistorted_tds_model = self._undistorted_tds_model
        undistorted_misc_shapes = self._undistorted_misc_shapes

        pooler_kernel_size = self._calc_pooler_kernel_size()
        pooler = torch.nn.AvgPool2d(kernel_size=pooler_kernel_size)

        bg = undistorted_tds_model._eval(u_x, u_y)
        for undistorted_misc_shape in undistorted_misc_shapes:
            bg[:, :] += undistorted_misc_shape._eval(u_x, u_y)[:, :]
        bg[:, :] *= jacobian_weights[:, :]
        bg = torch.unsqueeze(bg, dim=0)
        bg = torch.unsqueeze(bg, dim=0)
        bg = pooler(bg)[0, 0]

        return bg



    def _calc_intra_disk_shapes(self, u_x, u_y, jacobian_weights):
        undistorted_disks = self._undistorted_disks
        num_disks = len(undistorted_disks)

        if num_disks > 0:
            pooler_kernel_size = self._calc_pooler_kernel_size()
            pooler = torch.nn.AvgPool2d(kernel_size=pooler_kernel_size)

            intra_disk_shapes_shape = (num_disks,)+u_x.shape
    
            intra_disk_shapes = torch.zeros(intra_disk_shapes_shape,
                                            device=u_x.device)
            for disk_idx, undistorted_disk in enumerate(undistorted_disks):
                method_alias = undistorted_disk._eval_without_support
                intra_disk_shapes[disk_idx] = method_alias(u_x, u_y)
            intra_disk_shapes[:, :, :] *= jacobian_weights[None, :, :]
            intra_disk_shapes = torch.unsqueeze(intra_disk_shapes, dim=0)
            intra_disk_shapes = pooler(intra_disk_shapes)[0]
        else:
            num_pixels_across_pattern = self._num_pixels_across_pattern
            
            intra_disk_shapes_shape = (num_disks,
                                       num_pixels_across_pattern,
                                       num_pixels_across_pattern)
            
            intra_disk_shapes = torch.zeros(intra_disk_shapes_shape,
                                            device=u_x.device)

        return intra_disk_shapes



    def _apply_2d_guassian_filter(self, input_matrix, truncate):
        intermediate_tensor = input_matrix
        for axis_idx in range(2):
            kwargs = {"input_matrix": intermediate_tensor,
                      "truncate": truncate,
                      "axis_idx": axis_idx}
            intermediate_tensor = self._apply_1d_guassian_filter(**kwargs)
        output_matrix = intermediate_tensor

        return output_matrix



    def _apply_1d_guassian_filter(self, input_matrix, truncate, axis_idx):
        intermediate_tensor = torch.unsqueeze(input_matrix, dim=0)
        intermediate_tensor = torch.unsqueeze(intermediate_tensor, dim=0)

        sigma = self._gaussian_filter_std_dev

        if sigma > 0:
            radius = int(truncate*sigma + 0.5)
            coords = torch.arange(-radius, radius+1, device=input_matrix.device)

            weights = torch.exp(-(coords/sigma)*(coords/sigma)/2)
            weights /= torch.sum(weights)
            weights = torch.unsqueeze(weights, dim=axis_idx)
            weights = torch.unsqueeze(weights, dim=0)
            weights = torch.unsqueeze(weights, dim=0)

            kwargs = {"input": intermediate_tensor,
                      "weight": weights,
                      "padding": "same"}
            output_matrix = torch.nn.functional.conv2d(**kwargs)[0, 0]
        else:
            output_matrix = input_matrix

        return output_matrix



    def _apply_detector_partition_inpainting(self, input_image):
        N_DPW = self._detector_partition_width_in_pixels

        k_I_1 = ((input_image.shape[1]-1)//2) - (N_DPW//2)
        k_I_2 = k_I_1 + N_DPW - 1

        inpainting_mask = np.zeros(input_image.shape, dtype=bool)
        inpainting_mask[k_I_1:k_I_2+1, :] = True
        inpainting_mask[:, k_I_1:k_I_2+1] = True

        kwargs = {"image": input_image.numpy(force=True),
                  "mask": inpainting_mask}
        output_image = skimage.restoration.inpaint_biharmonic(**kwargs)
        output_image = torch.from_numpy(output_image)
        output_image = output_image.to(device=input_image.device,
                                       dtype=input_image.dtype)

        return output_image



    def _calc_disk_overlap_map_and_cache_select_intermediates(self, u_x, u_y):
        undistorted_disks = self._undistorted_disks
        num_disks = len(undistorted_disks)

        if num_disks > 0:
            if self._illumination_support is None:
                method_name = "_calc_illumination_support"
                method_alias = getattr(self, method_name)
                self._illumination_support = method_alias(u_x, u_y)
            illumination_support = self._illumination_support

            if self._disk_supports is None:
                method_name = ("_calc_disk_supports"
                               "_and_cache_select_intermediates")
                method_alias = getattr(self, method_name)
                self._disk_supports = method_alias(u_x, u_y)
            disk_supports = self._disk_supports

            L, R, B, T = self._mask_frame
            N_I_x = self._num_pixels_across_pattern
            N_I_y = N_I_x

            disk_overlap_map = (illumination_support
                                * torch.sum(disk_supports, dim=0))
            disk_overlap_map[:T, :] = 0
            disk_overlap_map[max(N_I_y-B, 0):, :] = 0
            disk_overlap_map[:, :L] = 0
            disk_overlap_map[:, max(N_I_x-R, 0):] = 0
        else:
            num_pixels_across_pattern = self._num_pixels_across_pattern
            
            disk_overlap_map_shape = 2*(num_pixels_across_pattern,)
            
            disk_overlap_map = torch.zeros(disk_overlap_map_shape,
                                           device=u_x.device,
                                           dtype=torch.int)
    
        return disk_overlap_map



    def _update_signal_axes(self, signal):
        num_pixels_across_pattern = signal.axes_manager.signal_shape[0]

        sizes = (signal.axes_manager.navigation_shape
                 + signal.axes_manager.signal_shape)
        scales = (1,
                  1/num_pixels_across_pattern,
                  -1/num_pixels_across_pattern)
        offsets = (0,
                   0.5/num_pixels_across_pattern,
                   1-(1-0.5)/num_pixels_across_pattern)
        axes_labels = (r"fake CBED pattern attribute",
                       r"fractional horizontal coordinate",
                       r"fractional vertical coordinate")
        units = ("dimensionless",)*3

        num_axes = len(units)

        for axis_idx in range(num_axes):
            axis = hyperspy.axes.UniformDataAxis(size=sizes[axis_idx],
                                                 scale=scales[axis_idx],
                                                 offset=offsets[axis_idx],
                                                 units=units[axis_idx],
                                                 name=axes_labels[axis_idx])
            signal.axes_manager[axis_idx].update_from(axis)
            signal.axes_manager[axis_idx].name = axis.name

        return None



    @property
    def signal(self):
        r"""`hyperspy._signals.signal2d.Signal2D`: The hyperspy signal 
        representation of the fake CBED pattern.

        See the summary documentation for the class
        :class:`fakecbed.discretized.CBEDPattern` for additional context.

        Let ``image``, ``illumination_support``, ``disk_overlap_map``,
        ``disk_supports``, ``disk_clipping_registry``,
        ``image_has_been_overridden``, ``num_disks``, ``disk_absence_registry``,
        and ``core_attrs`` denote the attributes
        :attr:`fakecbed.discretized.CBEDPattern.image`,
        :attr:`fakecbed.discretized.CBEDPattern.illumination_support`,
        :attr:`fakecbed.discretized.CBEDPattern.disk_overlap_map`,
        :attr:`fakecbed.discretized.CBEDPattern.disk_supports`,
        :attr:`fakecbed.discretized.CBEDPattern.disk_clipping_registry`,
        :attr:`fakecbed.discretized.CBEDPattern.disk_absence_registry`,
        :attr:`fakecbed.discretized.CBEDPattern.image_has_been_overridden`,
        :attr:`fakecbed.discretized.CBEDPattern.num_disks`, and
        :attr:`~fancytypes.Checkable.core_attrs`. Furthermore, let
        ``pre_serialize`` denote the method
        :meth:`~fancytypes.PreSerializable.pre_serialize`.

        The signal data, ``signal.data``, is a NumPy array having a shape equal
        to ``(num_disks+3,)+2*(core_attrs["num_pixels_across_pattern"],)``. The
        elements of ``signal.data`` are set effectively by:

        .. code-block:: python

           L, R, B, T = core_attrs["mask_frame"]
           N_I_x = core_attrs["num_pixels_across_pattern"]
           N_I_y = N_I_x

           signal.data[0] = image.numpy(force=True)
           signal.data[1] = illumination_support.numpy(force=True)
           signal.data[1, :T, :] = 0
           signal.data[1, max(N_I_y-B, 0):, :] = 0
           signal.data[1, :, :L] = 0
           signal.data[1, :, max(N_I_x-R, 0):] = 0
           signal.data[2] = disk_overlap_map.numpy(force=True)
           signal.data[3:] = disk_supports.numpy(force=True)

        The signal metadata, ``signal.metadata``, stores serializable forms of
        several instance attributes, in addition to other items of
        metadata. ``signal.metadata.as_dictionary()`` yields a dictionary
        ``signal_metadata`` that is calculated effectively by:

        .. code-block:: python

           distortion_model = core_attrs["distortion_model"]

           title = ("Fake Undistorted CBED Intensity Pattern"
                    if distortion_model.is_trivial
                    else "Fake Distorted CBED Intensity Pattern")

           pre_serialized_core_attrs = pre_serialize()

           fakecbed_metadata = {"num_disks": \
                                num_disks, 
                                "disk_clipping_registry": \
                                disk_clipping_registry.numpy(force=True),
                                "disk_absence_registry": \
                                disk_absence_registry.numpy(force=True),
                                "pre_serialized_core_attrs": \
                                pre_serialized_core_attrs, 
                                "cbed_pattern_image_has_been_overridden": \
                                image_has_been_overridden}

           signal_metadata = {"General": {"title": title},
                              "Signal": {"pixel value units": "dimensionless"},
                              "FakeCBED": fakecbed_metadata}

        Note that ``signal`` should be considered **read-only**.

        """
        result = self.get_signal(deep_copy=True)
        
        return result



    def get_image(self, deep_copy=_default_deep_copy):
        r"""Return the image of the target fake CBED pattern,
        :math:`\mathcal{I}_{\text{CBED};⌑;n,m}`.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``image`` denote the attribute
            :attr:`fakecbed.discretized.CBEDPattern.image`.

            If ``deep_copy`` is set to ``True``, then a deep copy of ``image``
            is returned.  Otherwise, a reference to ``image`` is returned.

        Returns
        -------
        image : `torch.Tensor` (`float`, ndim=2)
            The attribute :attr:`fakecbed.discretized.CBEDPattern.image`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._image is None:
            u_x, u_y = self._calc_u_x_and_u_y()
            method_name = "_calc_image_and_cache_select_intermediates"
            method_alias = getattr(self, method_name)
            self._image = method_alias(u_x, u_y)

        image = (self._image.detach().clone()
                 if (deep_copy == True)
                 else self._image)

        return image



    @property
    def image(self):
        r"""`torch.Tensor`: The image of the target fake CBED pattern, 
        :math:`\mathcal{I}_{\text{CBED};⌑;n,m}`.

        See the summary documentation for the class
        :class:`fakecbed.discretized.CBEDPattern` for additional context, in
        particular a description of the calculation of
        :math:`\mathcal{I}_{\text{CBED};⌑;n,m}`.

        Let ``core_attrs`` denote the attribute
        :attr:`~fancytypes.Checkable.core_attrs`.

        ``image`` is a PyTorch tensor having a shape equal to
        ``2*(core_attrs["num_pixels_across_pattern"],)``.

        For every pair of nonnegative integers ``(n, m)`` that does not raise an
        ``IndexError`` exception upon calling ``image[n, m]``, ``image[n, m]``
        is equal to :math:`\mathcal{I}_{\text{CBED};⌑;n,m}`, with the integers
        :math:`n` and :math:`m` being equal to the values of ``n`` and ``m``
        respectively.

        Note that ``image`` should be considered **read-only**.

        """
        result = self.get_image(deep_copy=True)

        return result



    def get_illumination_support(self, deep_copy=_default_deep_copy):
        r"""Return the image of the illumination support,
        :math:`\mathcal{I}_{\text{OI};⌑;n,m}`.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``illumination_support`` denote the attribute
            :attr:`fakecbed.discretized.CBEDPattern.illumination_support`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``illumination_support`` is returned.  Otherwise, a reference to
            ``illumination_support`` is returned.

        Returns
        -------
        illumination_support : `torch.Tensor` (`bool`, ndim=2)
            The attribute
            :attr:`fakecbed.discretized.CBEDPattern.illumination_support`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._illumination_support is None:
            u_x, u_y = self._calc_u_x_and_u_y()
            method_name = "_calc_illumination_support"
            method_alias = getattr(self, method_name)
            self._illumination_support = method_alias(u_x, u_y)

        illumination_support = (self._illumination_support.detach().clone()
                                if (deep_copy == True)
                                else self._illumination_support)

        return illumination_support



    @property
    def illumination_support(self):
        r"""`torch.Tensor`: The image of the illumination support,
        :math:`\mathcal{I}_{\text{OI};⌑;n,m}`.

        See the summary documentation for the class
        :class:`fakecbed.discretized.CBEDPattern` for additional context, in
        particular a description of the calculation of
        :math:`\mathcal{I}_{\text{OI};⌑;n,m}`.

        Note that :math:`\mathcal{I}_{\text{CBED};⌑;n,m}` is the image of the
        target fake CBED pattern, which is stored in the attribute
        :attr:`fakecbed.discretized.CBEDPattern.image`

        Let ``core_attrs`` denote the attribute
        :attr:`~fancytypes.Checkable.core_attrs`.

        ``illumination_support`` is a PyTorch tensor having a shape equal to
        ``2*(core_attrs["num_pixels_across_pattern"],)``.

        For every pair of nonnegative integers ``(n, m)`` that does not raise an
        ``IndexError`` exception upon calling ``illumination_support[n, m]``,
        ``illumination_support[n, m]`` is equal to
        :math:`\mathcal{I}_{\text{OI};⌑;n,m}`, with the integers :math:`n` and
        :math:`m` being equal to the values of ``n`` and ``m``
        respectively. Furthermore, for each such pair of integers ``(n, m)``, if
        ``illumination_support[n, m]`` equals zero, then the corresponding pixel
        of the image of the target fake CBED pattern is also zero.

        Note that ``illumination_support`` should be considered **read-only**.

        """
        result = self.get_illumination_support(deep_copy=True)

        return result



    def get_disk_supports(self, deep_copy=_default_deep_copy):
        r"""Return the image stack of the disk supports,
        :math:`\left\{\mathcal{I}_{k;
        \text{DS};⌑;n,m}\right\}_{k=0}^{N_{\text{D}}-1}`.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``disk_supports`` denote the attribute
            :attr:`fakecbed.discretized.CBEDPattern.disk_supports`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``disk_supports`` is returned.  Otherwise, a reference to
            ``disk_supports`` is returned.

        Returns
        -------
        disk_supports : `torch.Tensor` (`bool`, ndim=3)
            The attribute
            :attr:`fakecbed.discretized.CBEDPattern.disk_supports`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._disk_supports is None:
            u_x, u_y = self._calc_u_x_and_u_y()
            method_name = "_calc_disk_supports_and_cache_select_intermediates"
            method_alias = getattr(self, method_name)
            self._disk_supports = method_alias(u_x, u_y)

        disk_supports = (self._disk_supports.detach().clone()
                         if (deep_copy == True)
                         else self._disk_supports)

        return disk_supports



    @property
    def disk_supports(self):
        r"""`torch.Tensor`: The image stack of the disk supports,
        :math:`\left\{\mathcal{I}_{k;
        \text{DS};⌑;n,m}\right\}_{k=0}^{N_{\text{D}}-1}`.

        See the summary documentation for the class
        :class:`fakecbed.discretized.CBEDPattern` for additional context, in
        particular a description of the calculation of
        :math:`\left\{\mathcal{I}_{k;
        \text{DS};⌑;n,m}\right\}_{k=0}^{N_{\text{D}}-1}`.

        Let ``core_attrs`` and ``num_disks`` denote the attributes
        :attr:`~fancytypes.Checkable.core_attrs` and
        :attr:`fakecbed.discretized.CBEDPattern.num_disks` respectively.

        ``disk_supports`` is a PyTorch tensor having a shape equal to
        ``(num_disks,) + 2*(core_attrs["num_pixels_across_pattern"],)``.

        For every pair of nonnegative integers ``(k, n, m)`` that does not raise
        an ``IndexError`` exception upon calling ``disk_supports[k, n, m]``,
        ``disk_supports[k, n, m]`` is equal to
        :math:`\mathcal{I}_{k;\text{DS};⌑;n,m}`, with the integers :math:`k`,
        :math:`n`, and :math:`m` being equal to the values of ``k``, ``n``, and
        ``m`` respectively. Furthermore, for each such triplet of integers ``(k,
        n, m)``, if ``disk_supports[k, n, m]`` equals zero, then the
        :math:`k^{\text{th}}` distorted CBED disk is not supported at the pixel
        of the image of the target fake CBED pattern specified by ``(n, m)``.

        Note that ``disk_supports`` should be considered **read-only**.

        """
        result = self.get_disk_supports(deep_copy=True)

        return result



    def get_disk_overlap_map(self, deep_copy=_default_deep_copy):
        r"""Return the image of the disk overlap map,
        :math:`\mathcal{I}_{\text{DOM};⌑;n,m}`.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``disk_overlap_map`` denote the attribute
            :attr:`fakecbed.discretized.CBEDPattern.disk_overlap_map`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``disk_overlap_map`` is returned.  Otherwise, a reference to
            ``disk_overlap_map`` is returned.

        Returns
        -------
        disk_overlap_map : `torch.Tensor` (`int`, ndim=2)
            The attribute
            :attr:`fakecbed.discretized.CBEDPattern.disk_overlap_map`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._disk_overlap_map is None:
            u_x, u_y = self._calc_u_x_and_u_y()
            method_name = ("_calc_disk_overlap_map"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._disk_overlap_map = method_alias(u_x, u_y)

        disk_overlap_map = (self._disk_overlap_map.detach().clone()
                            if (deep_copy == True)
                            else self._disk_overlap_map)

        return disk_overlap_map



    @property
    def disk_overlap_map(self):
        r"""`torch.Tensor`: The image of the disk overlap map,
        :math:`\mathcal{I}_{\text{DOM};⌑;n,m}`.

        See the summary documentation for the class
        :class:`fakecbed.discretized.CBEDPattern` for additional context, in
        particular a description of the calculation of
        :math:`\mathcal{I}_{\text{DOM};⌑;n,m}`.

        Note that :math:`\mathcal{I}_{\text{CBED};⌑;n,m}` is the image of the
        target fake CBED pattern, which is stored in the attribute
        :attr:`fakecbed.discretized.CBEDPattern.image`

        Let ``core_attrs`` denote the attribute
        :attr:`~fancytypes.Checkable.core_attrs`.

        ``disk_overlap_map`` is a PyTorch tensor having a shape equal to
        ``2*(core_attrs["num_pixels_across_pattern"],)``.

        For every pair of nonnegative integers ``(n, m)`` that does not raise an
        ``IndexError`` exception upon calling ``disk_overlap_map[n, m]``,
        ``disk_overlap_map[n, m]`` is equal to
        :math:`\mathcal{I}_{\text{DOM};⌑;n,m}`, with the integers :math:`n` and
        :math:`m` being equal to the values of ``n`` and ``m`` respectively. In
        other words, for each such pair of integers ``(n, m)``,
        ``disk_overlap_map[n, m]`` is equal to the number of imaged CBED disks
        that overlap at the corresponding pixel of the image of the target fake
        CBED pattern.

        Note that ``disk_overlap_map`` should be considered **read-only**.

        """
        result = self.get_disk_overlap_map(deep_copy=True)

        return result



    def get_disk_clipping_registry(self, deep_copy=_default_deep_copy):
        r"""Return the disk clipping registry,
        :math:`\left\{\Omega_{k;\text{DCR};⌑}\right\}_{k=0}^{N_{\text{D}}-1}`.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``disk_clipping_registry`` denote the attribute
            :attr:`fakecbed.discretized.CBEDPattern.disk_clipping_registry`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``disk_clipping_registry`` is returned.  Otherwise, a reference to
            ``disk_clipping_registry`` is returned.

        Returns
        -------
        disk_clipping_registry : `torch.Tensor` (`bool`, ndim=1)
            The attribute
            :attr:`fakecbed.discretized.CBEDPattern.disk_clipping_registry`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._disk_clipping_registry is None:
            u_x, u_y = self._calc_u_x_and_u_y()
            method_name = ("_calc_disk_clipping_registry"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._disk_clipping_registry = method_alias(u_x, u_y)

        disk_clipping_registry = (self._disk_clipping_registry.detach().clone()
                                  if (deep_copy == True)
                                  else self._disk_clipping_registry)

        return disk_clipping_registry


    
    @property
    def disk_clipping_registry(self):
        r"""`torch.Tensor`: The disk clipping registry, 
        :math:`\left\{\Omega_{k;\text{DCR};⌑}\right\}_{k=0}^{N_{\text{D}}-1}`.

        See the summary documentation for the class
        :class:`fakecbed.discretized.CBEDPattern` for additional context, in
        particular a description of the calculation of
        :math:`\left\{\Omega_{k;\text{DCR};⌑}\right\}_{k=0}^{N_{\text{D}}-1}`.

        Note that :math:`N_{\text{D}}` is equal to the value of the attribute
        :attr:`fakecbed.discretized.CBEDPattern.num_disks`,
        :math:`\mathcal{I}_{\text{OI};⌑;n,m}` is the image of the illumination
        support, and :math:`\mathcal{I}_{k;\text{DS};⌑;n,m}` is the image of the
        support of the :math:`k^{\text{th}}` distorted CBED disk.

        ``disk_clipping_registry`` is a one-dimensional PyTorch tensor of length
        equal to :math:`N_{\text{D}}`. For every nonnegative integer ``k`` less
        than :math:`N_{\text{D}}`, ``disk_clipping_registry[k]`` is
        :math:`\Omega_{k;\text{DCR};⌑}`, with the integer :math:`k` being equal
        to the value of ``k``. If ``disk_clipping_registry[k]`` is equal to
        ``False``, then the image of the support of the :math:`k^{\text{th}}`
        distorted CBED disk has at least one nonzero pixel, and that the
        position of every nonzero pixel of said image is at least two pixels
        away from (i.e. at least pixel-wise next-nearest neighbours to) the
        position of every zero-valued pixel of the image of the illumination
        support, at least two pixels away from the position of every pixel
        inside the mask frame, and at least one pixel away from every pixel
        bordering the image of the illumination support. Otherwise, if
        ``disk_clipping_registry[k]`` is equal to ``True``, then the opposite of
        the above scenario is true.

        Note that ``disk_clipping_registry`` should be considered **read-only**.

        """
        result = self.get_disk_clipping_registry(deep_copy=True)

        return result



    def get_disk_absence_registry(self, deep_copy=_default_deep_copy):
        r"""Return the disk absence registry,
        :math:`\left\{\Omega_{k;\text{DAR};⌑}\right\}_{k=0}^{N_{\text{D}}-1}`.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``disk_absence_registry`` denote the attribute
            :attr:`fakecbed.discretized.CBEDPattern.disk_absence_registry`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``disk_absence_registry`` is returned.  Otherwise, a reference to
            ``disk_absence_registry`` is returned.

        Returns
        -------
        disk_absence_registry : `torch.Tensor` (`bool`, ndim=1)
            The attribute
            :attr:`fakecbed.discretized.CBEDPattern.disk_absence_registry`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._disk_absence_registry is None:
            u_x, u_y = self._calc_u_x_and_u_y()
            method_name = ("_calc_disk_absence_registry"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._disk_absence_registry = method_alias(u_x, u_y)

        disk_absence_registry = (self._disk_absence_registry.detach().clone()
                                 if (deep_copy == True)
                                 else self._disk_absence_registry)

        return disk_absence_registry


    
    @property
    def disk_absence_registry(self):
        r"""`torch.Tensor`: The disk absence registry, 
        :math:`\left\{\Omega_{k;\text{DAR};⌑}\right\}_{k=0}^{N_{\text{D}}-1}`.

        See the summary documentation for the class
        :class:`fakecbed.discretized.CBEDPattern` for additional context, in
        particular a description of the calculation of
        :math:`\left\{\Omega_{k;\text{DAR};⌑}\right\}_{k=0}^{N_{\text{D}}-1}`.

        Note that :math:`N_{\text{D}}` is equal to the value of the attribute
        :attr:`fakecbed.discretized.CBEDPattern.num_disks`, and
        :math:`\mathcal{I}_{k;\text{DS};⌑;n,m}` is the image of the support of
        the :math:`k^{\text{th}}` distorted CBED disk.

        ``disk_absence_registry`` is a one-dimensional PyTorch tensor of length
        equal to :math:`N_{\text{D}}`. For every nonnegative integer ``k`` less
        than :math:`N_{\text{D}}`, ``disk_absence_registry[k]`` is
        :math:`\Omega_{k;\text{DAR};⌑}`, with the integer :math:`k` being equal
        to the value of ``k``. If ``disk_absence_registry[k]`` is equal to
        ``False``, then the image of the support of the :math:`k^{\text{th}}`
        distorted CBED disk has at least one nonzero pixel. Otherwise, if
        ``disk_absence_registry[k]`` is equal to ``True``, then the opposite of
        the above scenario is true.

        Note that ``disk_absence_registry`` should be considered **read-only**.

        """
        result = self.get_disk_absence_registry(deep_copy=True)

        return result



    @property
    def image_has_been_overridden(self):
        r"""`bool`: Equals ``True`` if the image of the target fake CBED pattern
        has been overridden.

        Let ``override_image_then_reapply_mask``, and ``update`` denote the
        methods
        :meth:`fakecbed.discretized.CBEDPattern.override_image_then_reapply_mask`,
        and :meth:`~fancytypes.Updatable.update`
        respectively. ``image_has_been_overridden`` equals ``True`` if the
        method ``override_image_then_reapply_mask`` has been called without
        raising an exception after either instance update via the method
        ``update``, or instance construction. Otherwise,
        ``image_has_been_overridden`` equals ``False``.

        Note that ``image_has_been_overridden`` should be considered
        **read-only**.

        """
        result = self._image_has_been_overridden
        
        return result



def _check_and_convert_cbed_pattern(params):
    obj_name = "cbed_pattern"
    obj = params[obj_name]

    accepted_types = (fakecbed.discretized.CBEDPattern, type(None))

    if isinstance(obj, accepted_types[-1]):
        cbed_pattern = accepted_types[0]()
    else:
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        cbed_pattern = copy.deepcopy(obj)

    return cbed_pattern



def _pre_serialize_cbed_pattern(cbed_pattern):
    obj_to_pre_serialize = cbed_pattern
    serializable_rep = obj_to_pre_serialize.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_cbed_pattern(serializable_rep):
    cbed_pattern = \
        fakecbed.discretized.CBEDPattern.de_pre_serialize(serializable_rep)

    return cbed_pattern



def _check_and_convert_cropping_window_center(params):
    obj_name = "cropping_window_center"
    obj = params[obj_name]

    current_func_name = "_check_and_convert_cropping_window_center"
    
    if obj is not None:
        try:
            func_alias = czekitout.convert.to_pair_of_floats
            kwargs = {"obj": obj, "obj_name": obj_name}
            cropping_window_center = func_alias(**kwargs)
        except:
            err_msg = globals()[current_func_name+"_err_msg_1"]
            raise TypeError(err_msg)
    else:
        cls_alias = fakecbed.discretized.CBEDPattern
        cbed_pattern = (params["cbed_pattern"]
                        if isinstance(params["cbed_pattern"], cls_alias)
                        else _check_and_convert_cbed_pattern(params))
        params["cbed_pattern"] = cbed_pattern

        cbed_pattern_core_attrs = cbed_pattern.get_core_attrs(deep_copy=False)
        N_W_h = cbed_pattern_core_attrs["num_pixels_across_pattern"]
        N_W_v = N_W_h

        h_scale = 1/N_W_h
        v_scale = -1/N_W_v

        h_offset = 0.5/N_W_h
        v_offset = 1 - (1-0.5)/N_W_v

        cropping_window_center = (h_offset + h_scale*((N_W_h-1)//2),
                                  v_offset + v_scale*((N_W_v-1)//2))

    return cropping_window_center



def _pre_serialize_cropping_window_center(cropping_window_center):
    obj_to_pre_serialize = cropping_window_center
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_cropping_window_center(serializable_rep):
    cropping_window_center = serializable_rep

    return cropping_window_center



def _check_and_convert_cropping_window_dims_in_pixels(params):
    obj_name = "cropping_window_dims_in_pixels"
    obj = params[obj_name]

    current_func_name = "_check_and_convert_cropping_window_dims_in_pixels"

    if obj is not None:
        try:
            kwargs = {"obj": obj, "obj_name": obj_name}
            func_alias = czekitout.convert.to_pair_of_positive_ints
            cropping_window_dims_in_pixels = func_alias(**kwargs)
        except:
            err_msg = globals()[current_func_name+"_err_msg_1"]
            raise TypeError(err_msg)
    else:
        cls_alias = fakecbed.discretized.CBEDPattern
        cbed_pattern = (params["cbed_pattern"]
                        if isinstance(params["cbed_pattern"], cls_alias)
                        else _check_and_convert_cbed_pattern(params))
        params["cbed_pattern"] = cbed_pattern

        cbed_pattern_core_attrs = cbed_pattern.get_core_attrs(deep_copy=False)
        N_W_h = cbed_pattern_core_attrs["num_pixels_across_pattern"]
        N_W_v = N_W_h

        cropping_window_dims_in_pixels = (N_W_h, N_W_v)

    return cropping_window_dims_in_pixels



def _pre_serialize_cropping_window_dims_in_pixels(
        cropping_window_dims_in_pixels):
    obj_to_pre_serialize = cropping_window_dims_in_pixels
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_cropping_window_dims_in_pixels(serializable_rep):
    cropping_window_dims_in_pixels = serializable_rep

    return cropping_window_dims_in_pixels



def _check_and_convert_principal_disk_idx(params):
    obj_name = "principal_disk_idx"
    func_alias = czekitout.convert.to_nonnegative_int
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    principal_disk_idx = func_alias(**kwargs)

    return principal_disk_idx



def _pre_serialize_principal_disk_idx(principal_disk_idx):
    obj_to_pre_serialize = principal_disk_idx
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_principal_disk_idx(serializable_rep):
    principal_disk_idx = serializable_rep

    return principal_disk_idx



def _check_and_convert_disk_boundary_sample_size(params):
    obj_name = "disk_boundary_sample_size"
    func_alias = czekitout.convert.to_positive_int
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    disk_boundary_sample_size = func_alias(**kwargs)

    return disk_boundary_sample_size



def _pre_serialize_disk_boundary_sample_size(disk_boundary_sample_size):
    obj_to_pre_serialize = disk_boundary_sample_size
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_disk_boundary_sample_size(serializable_rep):
    disk_boundary_sample_size = serializable_rep

    return disk_boundary_sample_size



_default_cbed_pattern = None
_default_cropping_window_center = None
_default_cropping_window_dims_in_pixels = None
_default_principal_disk_idx = 0
_default_disk_boundary_sample_size = 512



class CroppedCBEDPattern(fancytypes.PreSerializableAndUpdatable):
    r"""A discretized cropped fake convergent beam electron diffraction (CBED)
    pattern.

    A series of parameters need to be specified in order to create an image of a
    cropped fake CBED pattern, which are: the set of parameters of a discretized
    fake CBED pattern; the spatial dimensions and the target center of the
    cropping window to be applied to said pattern; an index specifying the CBED
    disk, should it exist, for which to analyze in further detail than any other
    CBED disk in the pattern; the number of points to sample from the boundary
    of the unclipped support of the CBED disk specified by the aforementioned
    index, again should the disk exist; and the mask frame to be applied after
    cropping the fake CBED pattern.

    Parameters
    ----------
    cbed_pattern : :class:`fakecbed.discretized.CBEDPattern` | `None`, optional
        The discretized fake CBED pattern for which to crop. If ``cbed_pattern``
        is set to ``None``, then the parameter will be reassigned to the value
        ``fakecbed.discretized.CBEDPattern()``.
    cropping_window_center : `array_like` (`float`, shape=(2,)) | `None`, optional
        If ``cropping_window_center`` is set to ``None``, then the center of the
        cropping window is set to the fractional coordinates of the discretized
        fake CBED pattern corresponding to the pixel that is ``(h_dim+1)//2-1``
        pixels to the right of the upper left corner pixel, and
        ``(v_dim+1)//2-1`` pixels below the same corner, where ``h_dim`` and
        ``v_dim`` are the horizontal and vertical dimensions of the discretized
        fake CBED pattern in units of pixels. Otherwise, if
        ``cropping_window_center`` is set to a pair of floating-point numbers,
        then ``cropping_window_center[0]`` and ``cropping_window_center[1]``
        specify the fractional horizontal and vertical coordinates,
        respectively, of the center of the cropping window prior to the subpixel
        shift to the nearest pixel. Note that the crop is applied after the
        subpixel shift to the nearest pixel. Moreover, note that the fractional
        coordinate pair :math:`\left(0, 0\right)` corresponds to the bottom left
        corner of the fake CBED pattern.

        We define the center of the cropping window to be ``(N_W_h+1)//2 - 1``
        pixels to the right of the upper left corner of the cropping window, and
        ``(N_W_v+1)//2 - 1`` pixels below the same corner, where ``N_W_h`` and
        ``N_W_v`` are the horizontal and vertical dimensions of the cropping
        window in units of pixels.
    cropping_window_dims_in_pixels : `array_like` (`int`, shape=(2,)) | `None`, optional
        If ``cropping_window_dims_in_pixels`` is set to ``None``, then the
        dimensions of the cropping window are set to the dimensions of
        discretized fake CBED pattern.  Otherwise, if
        ``cropping_window_dims_in_pixels`` is set to a pair of positive
        integers, then ``cropping_window_dims_in_pixels[0]`` and
        ``cropping_window_dims_in_pixels[1]`` specify the horizontal and
        vertical dimensions of the cropping window in units of pixels.
    principal_disk_idx : `int`, optional
        A nonnegative integer specifying the CBED disk, should it exist, for
        which to analyze in further detail than any other CBED disk in the
        pattern. Assuming that ``cbed_pattern`` has already been reassigned if
        necessary, and that ``principal_disk_idx <
        len(cbed_pattern.core_attrs["undistorted_disks"])``, then
        ``principal_disk_idx`` specifies the CBED disk constructed from the
        undistorted intensity pattern stored in
        ``cbed_pattern.core_attrs["undistorted_disks"][prinicpal_disk_idx]``. If
        ``principal_disk_idx >=
        len(cbed_pattern.core_attrs["undistorted_disks"])``, then
        ``principal_disk_idx`` specifies a nonexistent CBED disk. See the
        summary documentation for the class
        :class:`fakecbed.discretized.CBEDPattern`, in particular the description
        of the construction parameter ``undistorted_disks`` therein, for
        additional context.
    disk_boundary_sample_size : `int`, optional
        The number of points to sample from the boundary of the unclipped
        support of the CBED disk specified by ``principal_disk_idx``, should the
        disk exist, upon calling the method
        :class:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_cropped_image_fractional_coords`.
        Must be a positive integer.
    mask_frame : `array_like` (`int`, shape=(4,)), optional
        ``mask_frame`` specifies the mask frame to be applied after cropping the
        fake CBED pattern. ``mask_frame[0]``, ``mask_frame[1]``,
        ``mask_frame[2]``, and ``mask_frame[3]`` are the widths, in units of
        pixels, of the left, right, bottom, and top sides of the mask frame
        respectively. 
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
    ctor_param_names = ("cbed_pattern",
                        "cropping_window_center",
                        "cropping_window_dims_in_pixels",
                        "principal_disk_idx",
                        "disk_boundary_sample_size",
                        "mask_frame")
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
                 cbed_pattern=\
                 _default_cbed_pattern,
                 cropping_window_center=\
                 _default_cropping_window_center,
                 cropping_window_dims_in_pixels=\
                 _default_cropping_window_dims_in_pixels,
                 principal_disk_idx=\
                 _default_principal_disk_idx,
                 disk_boundary_sample_size=\
                 _default_disk_boundary_sample_size,
                 mask_frame=\
                 _default_mask_frame,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        kwargs = ctor_params
        kwargs["skip_cls_tests"] = True
        fancytypes.PreSerializableAndUpdatable.__init__(self, **kwargs)

        self.execute_post_core_attrs_update_actions()

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



    def execute_post_core_attrs_update_actions(self):
        self_core_attrs = self.get_core_attrs(deep_copy=False)
        for self_core_attr_name in self_core_attrs:
            attr_name = "_"+self_core_attr_name
            attr = self_core_attrs[self_core_attr_name]
            setattr(self, attr_name, attr)
        
        self._num_disks = \
            self._cbed_pattern._num_disks
        self._device = \
            self._cbed_pattern._device
        self._image_has_been_overridden = \
            self._cbed_pattern._image_has_been_overridden

        cropped_blank_signal = self._generate_cropped_blank_signal()

        self._cropped_blank_signal_offsets = \
            (cropped_blank_signal.axes_manager[0].offset,
             cropped_blank_signal.axes_manager[1].offset,
             cropped_blank_signal.axes_manager[2].offset)

        attr_name_subset = ("_illumination_support",
                            "_image",
                            "_signal",
                            "_disk_clipping_registry",
                            "_disk_supports",
                            "_disk_absence_registry",
                            "_principal_disk_is_clipped",
                            "_principal_disk_is_absent",
                            "_disk_overlap_map",
                            "_principal_disk_analysis_results",
                            "_principal_disk_bounding_box"
                            "_in_uncropped_image_fractional_coords",
                            "_principal_disk_bounding_box"
                            "_in_cropped_image_fractional_coords",
                            "_principal_disk_boundary_pts"
                            "_in_uncropped_image_fractional_coords",
                            "_principal_disk_boundary_pts"
                            "_in_cropped_image_fractional_coords")

        for attr_name in attr_name_subset:
            setattr(self, attr_name, None)

        return None



    def _generate_cropped_blank_signal(self):
        uncropped_blank_signal = self._generate_uncropped_blank_signal()

        optional_params = self._generate_optional_cropping_params()

        kwargs = {"input_signal": uncropped_blank_signal,
                  "optional_params": optional_params}
        cropped_blank_signal = empix.crop(**kwargs)

        return cropped_blank_signal



    def _generate_uncropped_blank_signal(self):
        cbed_pattern_core_attrs = \
            self._cbed_pattern.get_core_attrs(deep_copy=False)

        num_disks = self._num_disks
        N_x = cbed_pattern_core_attrs["num_pixels_across_pattern"]

        uncropped_blank_signal_data = np.zeros((3+num_disks, N_x, N_x),
                                               dtype=float)
        kwargs = {"data": uncropped_blank_signal_data}
        uncropped_blank_signal = hyperspy.signals.Signal2D(**kwargs)

        kwargs = {"signal": uncropped_blank_signal}
        self._cbed_pattern._update_signal_axes(**kwargs)

        return uncropped_blank_signal



    def _generate_optional_cropping_params(self):
        kwargs = {"center": self._cropping_window_center,
                  "window_dims": self._cropping_window_dims_in_pixels,
                  "pad_mode": "zeros",
                  "apply_symmetric_mask": False,
                  "title": "",
                  "skip_validation_and_conversion": True}
        optional_cropping_params = empix.OptionalCroppingParams(**kwargs)

        return optional_cropping_params



    def update(self,
               new_core_attr_subset_candidate,
               skip_validation_and_conversion=\
               _default_skip_validation_and_conversion):
        super().update(new_core_attr_subset_candidate,
                       skip_validation_and_conversion)
        self.execute_post_core_attrs_update_actions()

        return None



    @property
    def num_disks(self):
        r"""`int`: The total number of CBED disks defined, :math:`N_{\text{D}}`.

        See the summary documentation for the class
        :class:`fakecbed.discretized.CBEDPattern` for additional context.

        Let ``core_attrs`` denote the attribute
        :attr:`~fancytypes.Checkable.core_attrs`. ``num_disks`` is equal to
        ``core_attrs["cbed_pattern"].num_disks``.

        Note that ``num_disks`` should be considered **read-only**.

        """
        result = self._num_disks
        
        return result



    @property
    def device(self):
        r"""`torch.device`: The device on which computationally intensive 
        PyTorch operations are performed and attributes of the type 
        :class:`torch.Tensor` are stored.

        Note that ``device`` should be considered **read-only**.

        """
        result = copy.deepcopy(self._device)

        return result



    def get_principal_disk_boundary_pts_in_uncropped_image_fractional_coords(
            self, deep_copy=_default_deep_copy):
        r"""Return the sample of points on the boundary of the "principal" CBED 
        disk, should it exist, in fractional coordinates of the uncropped CBED
        pattern.

        Let ``core_attrs`` denote the attribute
        :attr:`~fancytypes.Checkable.core_attrs`.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let
            ``principal_disk_boundary_pts_in_uncropped_image_fractional_coords``
            denote the attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_uncropped_image_fractional_coords`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``principal_disk_boundary_pts_in_uncropped_image_fractional_coords``
            is returned.  Otherwise, a reference to
            ``principal_disk_boundary_pts_in_uncropped_image_fractional_coords``
            is returned.

        Returns
        -------
        principal_disk_boundary_pts_in_uncropped_image_fractional_coords : `torch.Tensor` (`float`, shape=(core_attrs["disk_boundary_sample_size"], 2)) | `None`
            The attribute 
            :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_uncropped_image_fractional_coords`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        attr_name = ("_principal_disk_boundary_pts"
                     "_in_uncropped_image_fractional_coords")
        attr = getattr(self, attr_name)

        if attr is None:
            method_name = ("_calc_principal_disk_analysis_results"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._principal_disk_analysis_results = method_alias()

            attr = getattr(self, attr_name)

        principal_disk_boundary_pts_in_uncropped_image_fractional_coords = \
            (attr
             if ((deep_copy == False) or (attr is None))
             else attr.detach().clone())

        return principal_disk_boundary_pts_in_uncropped_image_fractional_coords



    def _calc_principal_disk_analysis_results_and_cache_select_intermediates(
            self):
        principal_disk_analysis_results = tuple()
        
        method_name_set = \
            ("_calc_disk_boundary_pts_in_uncropped_image_fractional_coords",
             "_calc_disk_bounding_box_in_uncropped_image_fractional_coords",
             "_calc_disk_boundary_pts_in_cropped_image_fractional_coords",
             "_calc_disk_bounding_box_in_cropped_image_fractional_coords")

        for method_alias_idx, method_name in enumerate(method_name_set):
            method_alias = getattr(self, method_name)
            
            if method_alias_idx in (0,):
                kwargs = \
                    {"disk_idx": \
                     self._principal_disk_idx}
            elif method_alias_idx in (1, 2):
                kwargs = \
                    {"disk_boundary_pts_in_uncropped_image_fractional_coords": \
                     principal_disk_analysis_results[0]}
            else:
                kwargs = \
                    {"disk_boundary_pts_in_cropped_image_fractional_coords": \
                     principal_disk_analysis_results[2]}

            principal_disk_analysis_results += (method_alias(**kwargs),)

            attr_name = "_principal" + method_name[5:]
            setattr(self, attr_name, principal_disk_analysis_results[-1])

        return principal_disk_analysis_results



    def _calc_disk_boundary_pts_in_uncropped_image_fractional_coords(self,
                                                                     disk_idx):
        cbed_pattern_core_attrs = \
            self._cbed_pattern.get_core_attrs(deep_copy=False)

        undistorted_disks = cbed_pattern_core_attrs["undistorted_disks"]

        if disk_idx < len(undistorted_disks):
            undistorted_disk = undistorted_disks[disk_idx]

            num_pixels_across_uncropped_pattern = \
                cbed_pattern_core_attrs["num_pixels_across_pattern"]

            method_name = \
                "_calc_undistorted_disk_boundary_starting_angular_coord"
            method_alias = \
                getattr(self, method_name)
            kwargs = \
                {"undistorted_disk": \
                 undistorted_disk,
                 "num_pixels_across_uncropped_pattern": \
                 num_pixels_across_uncropped_pattern}
            undistorted_disk_boundary_starting_angular_coord = \
                method_alias(**kwargs)

            kwargs = {"steps": self._disk_boundary_sample_size}
            kwargs["start"] = undistorted_disk_boundary_starting_angular_coord
            kwargs["end"] = (kwargs["start"]
                             + (kwargs["steps"]-1)*(2*np.pi/kwargs["steps"]))
            undistorted_disk_boundary_angular_coords = torch.linspace(**kwargs)

            method_name = ("_calc_disk_boundary_pts_and_diffs"
                           "_in_uncropped_image_fractional_coords")
            method_alias = getattr(self, method_name)
            kwargs = {"undistorted_disk": \
                      undistorted_disk,
                      "undistorted_disk_boundary_angular_coords": \
                      undistorted_disk_boundary_angular_coords}
            pts_and_diffs = method_alias(**kwargs)

            disk_boundary_pts_in_uncropped_image_fractional_coords = \
                pts_and_diffs[0]
        else:
            disk_boundary_pts_in_uncropped_image_fractional_coords = \
                None

        return disk_boundary_pts_in_uncropped_image_fractional_coords



    def _calc_undistorted_disk_boundary_starting_angular_coord(
            self, undistorted_disk, num_pixels_across_uncropped_pattern):
        max_num_iterations = 30
        iteration_idx = 0
        num_pts = self._disk_boundary_sample_size
        search_algorithm_has_not_finished = True

        kwargs = {"steps": num_pts}
        kwargs["start"] = 0
        kwargs["end"] = (kwargs["start"]
                         + (kwargs["steps"]-1)*(2*np.pi/kwargs["steps"]))

        while search_algorithm_has_not_finished:
            angular_coords = torch.linspace(**kwargs)

            method_name = ("_calc_disk_boundary_pts_and_diffs"
                           "_in_uncropped_image_fractional_coords")
            method_alias = getattr(self, method_name)
            kwargs = {"undistorted_disk": \
                      undistorted_disk,
                      "undistorted_disk_boundary_angular_coords": \
                      angular_coords}
            pts, diffs = method_alias(**kwargs)
            
            angular_coord_idx = torch.argmax(pts[:, 0])
            diff_subset = diffs if (iteration_idx==0) else diffs[:-1]
            N_x = num_pixels_across_uncropped_pattern

            if iteration_idx == 0:
                min_angle = angular_coords[(angular_coord_idx-16)%num_pts]
                max_angle = angular_coords[(angular_coord_idx+16)%num_pts]
            else:
                min_angle = angular_coords[max(angular_coord_idx-16, 0)]
                max_angle = angular_coords[min(angular_coord_idx+16, num_pts-1)]
                
            kwargs = {"steps": num_pts}
            kwargs["start"] = min_angle
            kwargs["end"] = max_angle + (min_angle > max_angle)*(2*np.pi)

            search_algorithm_has_not_finished = \
                ((100*N_x*torch.amax(diff_subset) > 1)
                 and (iteration_idx < max_num_iterations-1))

            iteration_idx += 1
            
        undistorted_disk_boundary_starting_angular_coord = \
            angular_coords[angular_coord_idx].item() % (2*np.pi)

        return undistorted_disk_boundary_starting_angular_coord



    def _calc_disk_boundary_pts_and_diffs_in_uncropped_image_fractional_coords(
            self, undistorted_disk, undistorted_disk_boundary_angular_coords):
        method_name = \
            "_calc_shape_boundary_pts_in_uncropped_image_fractional_coords"
        method_alias = \
            getattr(self, method_name)
        kwargs = \
            {"shape": \
             undistorted_disk,
             "shape_boundary_angular_coords": \
             undistorted_disk_boundary_angular_coords}
        shape_boundary_pts_in_uncropped_image_fractional_coords = \
            method_alias(**kwargs)

        u_x = shape_boundary_pts_in_uncropped_image_fractional_coords[:, 0:1]
        u_y = shape_boundary_pts_in_uncropped_image_fractional_coords[:, 1:2]

        cbed_pattern_core_attrs = \
            self._cbed_pattern.get_core_attrs(deep_copy=False)
        distortion_model = \
            cbed_pattern_core_attrs["distortion_model"]

        distortion_model_core_attrs = \
            distortion_model.get_core_attrs(deep_copy=False)
        coord_transform_params = \
            distortion_model_core_attrs["coord_transform_params"]

        kwargs = {"u_x": u_x,
                  "u_y": u_y,
                  "coord_transform_params": coord_transform_params,
                  "device": self._device,
                  "skip_validation_and_conversion": True}
        q_x, q_y = distoptica.apply_coord_transform(**kwargs)

        num_pts = self._disk_boundary_sample_size
        pts = torch.zeros((num_pts, 2), device=self._device)
        pts[:, 0] = q_x[:, 0]
        pts[:, 1] = q_y[:, 0]

        diffs = torch.zeros_like(undistorted_disk_boundary_angular_coords)
        for angular_coord_idx in range(num_pts):
            pt_1 = pts[(angular_coord_idx)%num_pts]
            pt_2 = pts[(angular_coord_idx+1)%num_pts]
            diffs[angular_coord_idx] = torch.linalg.norm(pt_2-pt_1)

        disk_boundary_pts_and_diffs_in_uncropped_image_fractional_coords = \
            (pts, diffs)

        return disk_boundary_pts_and_diffs_in_uncropped_image_fractional_coords



    def _calc_shape_boundary_pts_in_uncropped_image_fractional_coords(
            self, shape, shape_boundary_angular_coords):
        shape_core_attrs = shape.get_core_attrs(deep_copy=False)
        shape_support = shape_core_attrs["support"]

        shape_support_core_attrs = shape_support.get_core_attrs(deep_copy=False)
        u_x_c, u_y_c = shape_support_core_attrs["center"]
        a = (shape_support_core_attrs["semi_major_axis"]
             if ("semi_major_axis" in shape_support_core_attrs)
             else shape_support_core_attrs["radius"])
        e = shape_support_core_attrs.get("eccentricity", 0)
        theta = shape_support_core_attrs.get("rotation_angle", 0)

        b = a*np.sqrt(1-e*e).item()
        phi = shape_boundary_angular_coords

        u_x = (u_x_c + a*torch.cos(phi+theta))[:, None]
        u_y = (u_y_c + b*torch.sin(phi+theta))[:, None]

        num_pts = shape_boundary_angular_coords.numel()
        pts = torch.zeros((num_pts, 2), device=self._device)
        pts[:, 0] = u_x[:, 0]
        pts[:, 1] = u_y[:, 0]

        shape_boundary_pts_in_uncropped_image_fractional_coords = pts
        
        return shape_boundary_pts_in_uncropped_image_fractional_coords



    def _calc_disk_bounding_box_in_uncropped_image_fractional_coords(
            self, disk_boundary_pts_in_uncropped_image_fractional_coords):
        if disk_boundary_pts_in_uncropped_image_fractional_coords is not None:
            pts = disk_boundary_pts_in_uncropped_image_fractional_coords

            disk_bounding_box_in_uncropped_image_fractional_coords = \
                (torch.amin(pts[:, 0]).item(),
                 torch.amax(pts[:, 0]).item(),
                 torch.amin(pts[:, 1]).item(),
                 torch.amax(pts[:, 1]).item())
        else:
            disk_bounding_box_in_uncropped_image_fractional_coords = None

        return disk_bounding_box_in_uncropped_image_fractional_coords



    def _calc_disk_boundary_pts_in_cropped_image_fractional_coords(
            self, disk_boundary_pts_in_uncropped_image_fractional_coords):
        if disk_boundary_pts_in_uncropped_image_fractional_coords is not None:
            cbed_pattern_core_attrs = \
                self._cbed_pattern.get_core_attrs(deep_copy=False)

            N_x = cbed_pattern_core_attrs["num_pixels_across_pattern"]
            N_y = N_x

            N_W_h, N_W_v = self._cropping_window_dims_in_pixels

            kwargs = \
                {"input": \
                 disk_boundary_pts_in_uncropped_image_fractional_coords}
            disk_boundary_pts_in_cropped_image_fractional_coords = \
                torch.zeros_like(**kwargs)

            disk_boundary_pts_in_cropped_image_fractional_coords[:, 0] = \
                ((disk_boundary_pts_in_uncropped_image_fractional_coords[:, 0]
                  - self._cropped_blank_signal_offsets[1]) * (N_x/N_W_h)
                 + (0.5/N_W_h))
            disk_boundary_pts_in_cropped_image_fractional_coords[:, 1] = \
                ((disk_boundary_pts_in_uncropped_image_fractional_coords[:, 1]
                  - self._cropped_blank_signal_offsets[2]) * (N_y/N_W_v)
                 + (1-(1-0.5)/N_W_v))
        else:
            disk_boundary_pts_in_cropped_image_fractional_coords = None

        return disk_boundary_pts_in_cropped_image_fractional_coords



    def _calc_disk_bounding_box_in_cropped_image_fractional_coords(
            self, disk_boundary_pts_in_cropped_image_fractional_coords):
        if disk_boundary_pts_in_cropped_image_fractional_coords is not None:
            pts = disk_boundary_pts_in_cropped_image_fractional_coords

            disk_bounding_box_in_cropped_image_fractional_coords = \
                (torch.amin(pts[:, 0]).item(),
                 torch.amax(pts[:, 0]).item(),
                 torch.amin(pts[:, 1]).item(),
                 torch.amax(pts[:, 1]).item())
        else:
            disk_bounding_box_in_cropped_image_fractional_coords = None

        return disk_bounding_box_in_cropped_image_fractional_coords



    @property
    def principal_disk_boundary_pts_in_uncropped_image_fractional_coords(self):
        r"""`torch.Tensor` | `None`: The sample of points on the boundary of the
        unclipped support of the "principal" CBED disk, should it exist, in 
        fractional coordinates of the uncropped CBED pattern.

        See the summary documentation for the classes
        :class:`fakecbed.discretized.CBEDPattern` and
        :class:`fakecbed.discretized.CroppedCBEDPattern` for additional context.

        Let ``core_attrs`` denote the attribute
        :attr:`~fancytypes.Checkable.core_attrs`. Furthermore, let
        ``cbed_pattern``, ``disk_boundary_sample_size``, and
        ``principal_disk_idx`` denote ``core_attrs["cbed_pattern"]``,
        ``core_attrs["disk_boundary_sample_size"]``, and
        ``core_attrs["principal_disk_idx"]`` respectively.

        The "principal" CBED disk refers to the CBED disk, should it exist, that
        is specified by the nonnegative integer ``principal_disk_idx``. If
        ``principal_disk_idx <
        len(cbed_pattern.core_attrs["undistorted_disks"])``, then
        ``principal_disk_idx`` specifies the CBED disk constructed from the
        undistorted intensity pattern stored in
        ``cbed_pattern.core_attrs["undistorted_disks"][prinicpal_disk_idx]``. If
        ``principal_disk_idx >=
        len(cbed_pattern.core_attrs["undistorted_disks"])``, then
        ``principal_disk_idx`` specifies a nonexistent CBED disk.

        If ``principal_disk_idx`` specifies a nonexistent CBED disk, then
        ``principal_disk_boundary_pts_in_uncropped_image_fractional_coords`` is
        set to ``None``. 

        The remainder of the documentation for the current attribute describes
        how said attribute is calculated, assuming that the principal CBED disk
        exists. In :mod:`fakecbed`, the undistorted shape of the support of
        every CBED disk is assumed to be an ellipse, which incidentally can be a
        circle if the eccentricity is zero. The definitions of the center, the
        semi-major axis, the eccentricity, and the rotation angle of a ellipse
        that are adopted in :mod:`fakecbed` are given implictly in the
        documentation for the class :class:`fakecbed.shapes.Ellipse`. Let
        :math:`\left(u_{x;c;\text{PDS}},u_{y;c;\text{PDS}}\right)`,
        :math:`a_{\text{PDS}}`, :math:`e_{\text{PDS}}`, and
        :math:`\theta_{\text{PDS}}` be the center, the semi-major axis, the
        eccentricity, and the rotation angle of the principal CBED disk support.

        Let :math:`u_{x}` and :math:`u_{y}` be the fractional horizontal and
        vertical coordinates, respectively, of a point in an undistorted image,
        where :math:`\left(u_{x},u_{y}\right)=\left(0,0\right)` is the bottom
        left corner of the image. Similarly, let :math:`q_{x}` and :math:`q_{y}`
        be the fractional horizontal and vertical coordinates, respectively, of
        a point in a distorted image, where
        :math:`\left(q_{x},q_{y}\right)=\left(0,0\right)` is the bottom left
        corner of the image. The core attribute ``cbed_pattern`` specifies a
        distortion model, which can be accessed via
        ``cbed_pattern.core_attrs["distortion_model"]``. The distortion model
        specifies a coordinate transformation,
        :math:`\left(T_{⌑;x}\left(u_{x},u_{y}\right),
        T_{⌑;x}\left(u_{x},u_{y}\right)\right)`, which maps a given coordinate
        pair :math:`\left(u_{x},u_{y}\right)` to a corresponding coordinate pair
        :math:`\left(q_{x},q_{y}\right)`.

        Next, let :math:`N_{\mathcal{I};x} and N_{\mathcal{I};y}` be the number
        of pixels in the image of the uncropped fake CBED pattern, where we
        assume that

        .. math ::
            N_{\mathcal{I};x}=N_{\mathcal{I};y},
            :label: N_I_x_eq_N_I_y__2

        Furthermore, let :math:`N_{\text{max-iter}}=30, and let N_{\text{SS}}`
        be ``disk_boundary_sample_size``.

        Next, let :math:`\partial S_{\text{PDS};x;l}` and :math:`\partial
        S_{\text{PDS};y;l}` denote
        ``principal_disk_boundary_pts_in_uncropped_image_fractional_coords[l,
        0]`` and
        ``principal_disk_boundary_pts_in_uncropped_image_fractional_coords[l,
        1]`` respectively, where
        ````principal_disk_boundary_pts_in_uncropped_image_fractional_coords``
        is the current attribute, and ``l`` is equal to the value of
        :math:`l`. The current attribute is calculated effectively by executing
        the following steps:

        1. Calculate

        .. math ::
            X_{1} \leftarrow
            \left\{ l^{\prime}\right\}_{l^{\prime}=0}^{N_{\text{SS}}-1}.
            :label: X_1__1

        2. Calculate

        .. math ::
            X_{2} \leftarrow
            \left\{ l^{\prime}\right\}_{l^{\prime}=0}^{N_{\text{SS}}-2}
            :label: X_2__1

        3. Calculate

        .. math ::
            m \leftarrow 0.
            :label: m__1

        4. Calculate
        
        .. math ::
            N_{\text{iter}} \leftarrow 0.
            :label: N_iter__1

        5. Calculate 
        
        .. math ::
            \phi_{l}\leftarrow\frac{2\pi l}{N_{\text{SS}}}
            \text{ for }l\in X_{1}.
            :label: phi_l__1

        6. Calculate
        
        .. math ::
            j\leftarrow\min\left(\left\{ N_{\text{iter}}+1,1\right\} \right).
            :label: j__1

        7. Calculate

        .. math ::
            u_{x;l}\leftarrow u_{x;c;\text{PDS}}
            +a_{\text{PDS}}\cos\left(\phi_{l}+\theta_{\text{PDS}}\right)
            \text{ for }l\in X_{1},
            :label: u_x_l__1

        and

        .. math ::
            u_{y;l}\leftarrow u_{y;c;\text{PDS}}
            +a_{\text{PDS}}\sqrt{1-e_{\text{PDS}}^{2}}
            \sin\left(\phi_{l}+\theta_{\text{PDS}}\right)\text{ for }l\in X_{1}.
            :label: u_y_l__1

        8. Calculate
        
        .. math ::
            q_{x;l}\leftarrow T_{⌑;x}\left(u_{x;l},u_{y;l}\right)
            \text{ for }l\in X_{1},
            :label: q_x_l__1

        and

        .. math ::
            q_{y;l}\leftarrow T_{⌑;y}\left(u_{x;l},u_{y;l}\right)
            \text{ for }l\in X_{1}.
            :label: q_y_l__1

        9. If :math:`m=0`, then go to step 10, otherwise go to step 21.

        10. Calculate

        .. math ::
            n\leftarrow\text{argmax}_{l\in X_{1}}\left(u_{x;l}\right).
            :label: n__1

        11. Calculate

        .. math ::
            \Delta_{q;l}\leftarrow\sqrt{\sum_{\alpha\in\left\{ x,y\right\} }
            \left[q_{\alpha;l\mod N_{\text{SS}}}
            -q_{\alpha;\left\{ l+1\right\} \mod N_{\text{SS}}}\right]^{2}}
            \text{ for }l\in X_{j}.
            :label: Delta_q_l__1

        12. If :math:`N_{\text{iter}}=0` then go to step 13, otherwise go to 
        step 15.

        13. Calculate

        .. math ::
            \phi_{\vdash}\leftarrow\phi_{\left(n-16\right)\mod N_{\text{SS}}},
            :label: phi_vdash__1

        and

        .. math ::
            \phi_{\dashv}\leftarrow\phi_{\left(n+16\right)\mod N_{\text{SS}}}.
            :label: phi_dashv__1

        14. Go to step 16.

        15. Calculate

        .. math ::
            \phi_{\vdash}\leftarrow
            \phi_{\max\left(\left\{n-16,0\right\}\right)},
            :label: phi_vdash__2

        and

        .. math ::
            \phi_{\dashv}\leftarrow
            \phi_{\min\left(\left\{n+16,N_{\text{SS}}-1\right\}\right)}.
            :label: phi_dashv__2

        16. If :math:`\phi_{\vdash}>\phi_{\dashv}`, then calculate
        
        .. math ::
            \phi_{\dashv}\leftarrow\phi_{\dashv}+2\pi.
            :label: phi_dashv__3

        17. Calculate

        .. math ::
            \phi_{l}\leftarrow
            \phi_{\vdash}+l\frac{\phi_{\dashv}-\phi_{\vdash}}{N_{\text{SS}}-1}
            \text{ for }l\in X_{1}.
            :label: phi_l__2

        18. If :math:`100 
        N_{\mathcal{I};x}\max\left(
        \left\{\Delta_{q;l}\right\}_{l\in S_{m}}\right) \le 1` or 
        :math:`N_{\text{iter}}=N_{\text{max-iter}}`, then calculate
        

        .. math ::
            m \leftarrow 1,
            :label: m__2

        .. math ::
            \phi_{\vdash}\leftarrow\phi_{n}\mod\left\{ 2\pi\right\},
            :label: phi_vdash__3

        and

        .. math ::
            \phi_{\dashv}\leftarrow\phi_{\vdash}+2\pi.
            :label: phi_dashv__4
            

        19. Calculate

        .. math ::
            N_{\text{iter}}\leftarrow N_{\text{iter}}+1.
            :label: N_iter__2

        20. Go to step 6.

        21. Calculate

        .. math ::
            \partial S_{\text{PDS};x;l}\leftarrow q_{x;l}\text{ for }l\in X_{1},
            :label: partial_S_PDS_x_l__1

        and

        .. math ::
            \partial S_{\text{PDS};y;l}\leftarrow q_{y;l}\text{ for }l\in X_{1}.
            :label: partial_S_PDS_y_l__1

        A consequence of the above algorithm is that :math:`\left(\partial
        S_{\text{PDS};x;0},\partial S_{\text{PDS};y;0}\right)` is the right-most
        point of the entire sample of points :math:`\left\{ \left(\partial
        S_{\text{PDS};x;l},\partial S_{\text{PDS};y;l}\right)\right\} _{l\in
        S_{1}}` on the boundary of the principal CBED disk.

        Note that
        ``principal_disk_boundary_pts_in_uncropped_image_fractional_coords``
        should be considered **read-only**.

        """
        method_name = ("get_principal_disk_boundary_pts"
                       "_in_uncropped_image_fractional_coords")
        method_alias = getattr(self, method_name)
        result = method_alias(deep_copy=True)

        return result



    def get_principal_disk_bounding_box_in_uncropped_image_fractional_coords(
            self, deep_copy=_default_deep_copy):
        r"""Return the bounding box of the "principal" CBED disk, should it 
        exist, in fractional coordinates of the uncropped CBED pattern.

        Let ``core_attrs`` denote the attribute
        :attr:`~fancytypes.Checkable.core_attrs`.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let
            ``principal_disk_bounding_box_in_uncropped_image_fractional_coords``
            denote the attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_bounding_box_in_uncropped_image_fractional_coords`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``principal_disk_bounding_box_in_uncropped_image_fractional_coords``
            is returned.  Otherwise, a reference to
            ``principal_disk_bounding_box_in_uncropped_image_fractional_coords``
            is returned.

        Returns
        -------
        principal_disk_bounding_box_in_uncropped_image_fractional_coords : `torch.Tensor` (`float`, shape=(core_attrs["disk_boundary_sample_size"], 2)) | `None`
            The attribute 
            :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_bounding_box_in_uncropped_image_fractional_coords`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        attr_name = ("_principal_disk_bounding_box"
                     "_in_uncropped_image_fractional_coords")
        attr = getattr(self, attr_name)

        if attr is None:
            method_name = ("_calc_principal_disk_analysis_results"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._principal_disk_analysis_results = method_alias()

            attr = getattr(self, attr_name)

        principal_disk_bounding_box_in_uncropped_image_fractional_coords = \
            copy.deepcopy(attr) if (deep_copy == True) else attr

        return principal_disk_bounding_box_in_uncropped_image_fractional_coords



    @property
    def principal_disk_bounding_box_in_uncropped_image_fractional_coords(self):
        r"""`tuple` | `None`: The bounding box of the unclipped support of the 
        "principal" CBED disk, should it exist, in fractional coordinates of the
        uncropped CBED pattern.

        See the documentation for the attribute
        :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_uncropped_image_fractional_coords`
        for additional context.

        Let ``principal_disk_boundary_pts_in_uncropped_image_fractional_coords``
        denote the attribute
        :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_uncropped_image_fractional_coords`.

        ``principal_disk_bounding_box_in_uncropped_image_fractional_coords`` is
        calculated effectively by:

        .. code-block:: python

           import torch
           
           pts = \
               principal_disk_boundary_pts_in_uncropped_image_fractional_coords

           if pts is not None:
               bounding_box = (torch.amin(pts[:, 0]).item(),
                               torch.amax(pts[:, 0]).item(),
                               torch.amin(pts[:, 1]).item(),
                               torch.amax(pts[:, 1]).item())
           else:
               bounding_box = None

           principal_disk_bounding_box_in_uncropped_image_fractional_coords = \
               bounding_box

        Note that
        ``principal_disk_bounding_box_in_uncropped_image_fractional_coords``
        should be considered **read-only**.

        """
        method_name = ("get_principal_disk_bounding_box"
                       "_in_uncropped_image_fractional_coords")
        method_alias = getattr(self, method_name)
        result = method_alias(deep_copy=True)

        return result



    def get_principal_disk_boundary_pts_in_cropped_image_fractional_coords(
            self, deep_copy=_default_deep_copy):
        r"""Return the sample of points on the boundary of the "principal" CBED 
        disk, should it exist, in fractional coordinates of the cropped CBED
        pattern.

        Let ``core_attrs`` denote the attribute
        :attr:`~fancytypes.Checkable.core_attrs`.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let
            ``principal_disk_boundary_pts_in_cropped_image_fractional_coords``
            denote the attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_cropped_image_fractional_coords`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``principal_disk_boundary_pts_in_cropped_image_fractional_coords``
            is returned.  Otherwise, a reference to
            ``principal_disk_boundary_pts_in_cropped_image_fractional_coords``
            is returned.

        Returns
        -------
        principal_disk_boundary_pts_in_cropped_image_fractional_coords : `torch.Tensor` (`float`, shape=(core_attrs["disk_boundary_sample_size"], 2)) | `None`
            The attribute 
            :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_cropped_image_fractional_coords`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        attr_name = ("_principal_disk_boundary_pts"
                     "_in_cropped_image_fractional_coords")
        attr = getattr(self, attr_name)

        if attr is None:
            method_name = ("_calc_principal_disk_analysis_results"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._principal_disk_analysis_results = method_alias()

            attr = getattr(self, attr_name)

        principal_disk_boundary_pts_in_cropped_image_fractional_coords = \
            (attr
             if ((deep_copy == False) or (attr is None))
             else attr.detach().clone())

        return principal_disk_boundary_pts_in_cropped_image_fractional_coords



    @property
    def principal_disk_boundary_pts_in_cropped_image_fractional_coords(self):
        r"""`torch.Tensor` | `None`: The sample of points on the boundary of the
        unclipped support of the "principal" CBED disk, should it exist, in 
        fractional coordinates of the cropped CBED pattern.

        See the documentation for the attribute
        :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_uncropped_image_fractional_coords`
        for additional context.

        Let ``core_attrs``,
        ``principal_disk_boundary_pts_in_uncropped_image_fractional_coords``,
        and ``signal``, denote the attributes
        :attr:`~fancytypes.Checkable.core_attrs`
        :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_uncropped_image_fractional_coords`,
        and :attr:`fakecbed.discretized.CroppedCBEDPattern.signal` respectively.

        ``principal_disk_boundary_pts_in_cropped_image_fractional_coords`` is
        calculated effectively by:

        .. code-block:: python

           import torch

           cbed_pattern = core_attrs["cbed_pattern"]
           N_x = cbed_pattern.core_attrs["num_pixels_across_pattern"]
           N_y = N_x

           N_W_h, N_W_v = core_attrs["cropping_window_dims_in_pixels"]

           boundary_pts_in_uncropped_image_fractional_coords = \
               principal_disk_boundary_pts_in_uncropped_image_fractional_coords

           if boundary_pts_in_uncropped_image_fractional_coords is not None:

               kwargs = \
                   {"input": boundary_pts_in_uncropped_image_fractional_coords}
               boundary_pts_in_cropped_image_fractional_coords = \
                   torch.zeros_like(**kwargs)

               boundary_pts_in_cropped_image_fractional_coords[:, 0] = \
                   ((boundary_pts_in_uncropped_image_fractional_coords[:, 0]
                     - signal.axes_manager[1].offset) * (N_x/N_W_h)
                    + (0.5/N_W_h))
               boundary_pts_in_cropped_image_fractional_coords[:, 1] = \
                   ((boundary_pts_in_uncropped_image_fractional_coords[:, 1]
                     - signal.axes_manager[2].offset) * (N_y/N_W_v)
                    + (1-(1-0.5)/N_W_v))
           else:
               boundary_pts_in_cropped_image_fractional_coords = \
                   None

           principal_disk_boundary_pts_in_cropped_image_fractional_coords = \
               boundary_pts_in_cropped_image_fractional_coords

        Note that
        ``principal_disk_boundary_pts_in_cropped_image_fractional_coords``
        should be considered **read-only**.

        """
        method_name = ("get_principal_disk_boundary_pts"
                       "_in_cropped_image_fractional_coords")
        method_alias = getattr(self, method_name)
        result = method_alias(deep_copy=True)

        return result



    def get_principal_disk_bounding_box_in_cropped_image_fractional_coords(
            self, deep_copy=_default_deep_copy):
        r"""Return the bounding box of the "principal" CBED disk, should it 
        exist, in fractional coordinates of the cropped CBED pattern.

        Let ``core_attrs`` denote the attribute
        :attr:`~fancytypes.Checkable.core_attrs`.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let
            ``principal_disk_bounding_box_in_cropped_image_fractional_coords``
            denote the attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_bounding_box_in_cropped_image_fractional_coords`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``principal_disk_bounding_box_in_cropped_image_fractional_coords``
            is returned.  Otherwise, a reference to
            ``principal_disk_bounding_box_in_cropped_image_fractional_coords``
            is returned.

        Returns
        -------
        principal_disk_bounding_box_in_cropped_image_fractional_coords : `torch.Tensor` (`float`, shape=(core_attrs["disk_boundary_sample_size"], 2)) | `None`
            The attribute 
            :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_bounding_box_in_cropped_image_fractional_coords`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        attr_name = ("_principal_disk_bounding_box"
                     "_in_cropped_image_fractional_coords")
        attr = getattr(self, attr_name)

        if attr is None:
            method_name = ("_calc_principal_disk_analysis_results"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._principal_disk_analysis_results = method_alias()

            attr = getattr(self, attr_name)

        principal_disk_bounding_box_in_cropped_image_fractional_coords = \
            copy.deepcopy(attr) if (deep_copy == True) else attr

        return principal_disk_bounding_box_in_cropped_image_fractional_coords



    @property
    def principal_disk_bounding_box_in_cropped_image_fractional_coords(self):
        r"""`tuple` | `None`: The bounding box of the unclipped support of the 
        "principal" CBED disk, should it exist, in fractional coordinates of the
        cropped CBED pattern.

        See the documentation for the attribute
        :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_cropped_image_fractional_coords`
        for additional context.

        Let ``principal_disk_boundary_pts_in_cropped_image_fractional_coords``
        denote the attribute
        :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_cropped_image_fractional_coords`.

        ``principal_disk_bounding_box_in_cropped_image_fractional_coords`` is
        calculated effectively by:

        .. code-block:: python

           import torch
           
           pts = \
               principal_disk_boundary_pts_in_cropped_image_fractional_coords

           if pts is not None:
               bounding_box = (torch.amin(pts[:, 0]).item(),
                               torch.amax(pts[:, 0]).item(),
                               torch.amin(pts[:, 1]).item(),
                               torch.amax(pts[:, 1]).item())
           else:
               bounding_box = None

           principal_disk_bounding_box_in_cropped_image_fractional_coords = \
               bounding_box

        Note that
        ``principal_disk_bounding_box_in_cropped_image_fractional_coords``
        should be considered **read-only**.

        """
        method_name = ("get_principal_disk_bounding_box"
                       "_in_cropped_image_fractional_coords")
        method_alias = getattr(self, method_name)
        result = method_alias(deep_copy=True)

        return result



    def get_disk_supports(self, deep_copy=_default_deep_copy):
        r"""Return the image stack of the disk supports of the cropped fake CBED
        pattern.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``disk_supports`` denote the attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_supports`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``disk_supports`` is returned.  Otherwise, a reference to
            ``disk_supports`` is returned.

        Returns
        -------
        disk_supports : `torch.Tensor` (`bool`, ndim=3)
            The attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_supports`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._disk_supports is None:
            method_name = "_calc_disk_supports"
            method_alias = getattr(self, method_name)
            self._disk_supports = method_alias()

        disk_supports = (self._disk_supports.detach().clone()
                         if (deep_copy == True)
                         else self._disk_supports)

        return disk_supports



    def _calc_disk_supports(self):
        N_W_h, N_W_v = self._cropping_window_dims_in_pixels
        L, R, B, T = self._mask_frame  # Mask frame of cropped pattern.

        uncropped_cbed_pattern_disk_supports = \
            self._cbed_pattern.get_disk_supports(deep_copy=False)
        uncropped_cbed_pattern_disk_supports = \
            uncropped_cbed_pattern_disk_supports.cpu().detach().clone()

        signal_to_crop = \
            self._generate_uncropped_blank_signal()
        signal_to_crop.data[3:] = \
            uncropped_cbed_pattern_disk_supports.numpy(force=True)[:]

        optional_params = self._generate_optional_cropping_params()

        kwargs = {"input_signal": signal_to_crop,
                  "optional_params": optional_params}
        cropped_signal = empix.crop(**kwargs)

        disk_supports = cropped_signal.data[3:]
        disk_supports_dtype = uncropped_cbed_pattern_disk_supports.dtype
        disk_supports = torch.from_numpy(disk_supports)
        disk_supports = disk_supports.to(device=self._device,
                                         dtype=disk_supports_dtype)
        disk_supports[:, :T, :] = 0
        disk_supports[:, max(N_W_v-B, 0):, :] = 0
        disk_supports[:, :, :L] = 0
        disk_supports[:, :, max(N_W_h-R, 0):] = 0

        return disk_supports



    @property
    def disk_supports(self):
        r"""`torch.Tensor`: The image stack of the disk supports of the cropped
        fake CBED pattern.

        See the documentation for the attribute
        :attr:`fakecbed.discretized.CroppedCBEDPattern.signal` for additional
        context.

        Let ``signal``, ``device``, and ``core_attrs`` denote the attributes
        :attr:`fakecbed.discretized.CroppedCBEDPattern.signal`,
        :attr:`fakecbed.discretized.CroppedCBEDPattern.device`, and
        :attr:`~fancytypes.Checkable.core_attrs` respectively.

        ``disk_supports`` is calculated effectively by:

        .. code-block:: python

           N_W_h, N_W_v = core_attrs["cropping_window_dims_in_pixels"]
           L, R, B, T = core_attrs["mask_frame"]

           disk_supports = signal.data[3:]
           disk_supports = torch.from_numpy(disk_supports)
           disk_supports = disk_supports.to(device=device, dtype=bool)
           disk_supports[:, :T, :] = 0
           disk_supports[:, max(N_W_v-B, 0):, :] = 0
           disk_supports[:, :, :L] = 0
           disk_supports[:, :, max(N_W_h-R, 0):] = 0

        Note that ``disk_supports`` should be considered **read-only**.

        """
        result = self.get_disk_supports(deep_copy=True)

        return result



    def get_disk_clipping_registry(self, deep_copy=_default_deep_copy):
        r"""Return the disk clipping registry of the cropped fake CBED pattern.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``disk_clipping_registry`` denote the attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_clipping_registry`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``disk_clipping_registry`` is returned.  Otherwise, a reference to
            ``disk_clipping_registry`` is returned.

        Returns
        -------
        disk_clipping_registry : `torch.Tensor` (`bool`, ndim=1)
            The attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_clipping_registry`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._disk_clipping_registry is None:
            method_name = ("_calc_disk_clipping_registry"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._disk_clipping_registry = method_alias()

        disk_clipping_registry = (self._disk_clipping_registry.detach().clone()
                                  if (deep_copy == True)
                                  else self._disk_clipping_registry)

        return disk_clipping_registry


    
    def _calc_disk_clipping_registry_and_cache_select_intermediates(self):
        num_disks = self._num_disks

        if num_disks > 0:
            if self._disk_supports is None:
                method_name = "_calc_disk_supports"
                method_alias = getattr(self, method_name)
                self._disk_supports = method_alias()
            disk_supports = self._disk_supports

            clip_support = self._calc_clip_support()

            disk_clipping_map = disk_supports*clip_support[None, :, :]
    
            disk_clipping_registry = ((disk_clipping_map.sum(dim=(1, 2)) != 0)
                                      + (disk_supports.sum(dim=(1, 2)) == 0))
        else:
            disk_clipping_registry = torch.zeros((num_disks,),
                                                 device=self._device,
                                                 dtype=torch.bool)

        return disk_clipping_registry



    def _calc_clip_support(self):
        cbed_pattern = self._cbed_pattern
        
        cbed_pattern_core_attrs = cbed_pattern.get_core_attrs(deep_copy=False)
        
        illumination_support = \
            cbed_pattern.get_illumination_support(deep_copy=False)
        illumination_support = \
            illumination_support.cpu().detach().clone()

        L, R, B, T = cbed_pattern._mask_frame

        N_x = cbed_pattern_core_attrs["num_pixels_across_pattern"]
        N_y = N_x

        signal_to_crop = self._generate_uncropped_blank_signal()
        signal_to_crop.data[0] = illumination_support.numpy(force=True)
        signal_to_crop.data[0, :T, :] = 0
        signal_to_crop.data[0, max(N_y-B, 0):, :] = 0
        signal_to_crop.data[0, :, :L] = 0
        signal_to_crop.data[0, :, max(N_x-R, 0):] = 0

        optional_params = self._generate_optional_cropping_params()

        kwargs = {"input_signal": signal_to_crop,
                  "optional_params": optional_params}
        cropped_signal = empix.crop(**kwargs)
        
        clip_support = torch.from_numpy(1.0-cropped_signal.data[0])
        clip_support = clip_support.to(device=self._device, dtype=torch.float)
        for _ in range(2):
            clip_support = torch.unsqueeze(clip_support, dim=0)

        conv_weights = torch.ones((1, 1, 3, 3), device=self._device)

        N_W_h, N_W_v = self._cropping_window_dims_in_pixels
        L, R, B, T = self._mask_frame  # Mask frame of cropped pattern.

        kwargs = {"input": clip_support,
                  "weight": conv_weights,
                  "padding": "same"}
        clip_support = (torch.nn.functional.conv2d(**kwargs) != 0)[0, 0]
        clip_support[:T+1, :] = True
        clip_support[max(N_W_v-B-1, 0):, :] = True
        clip_support[:, :L+1] = True
        clip_support[:, max(N_W_h-R-1, 0):] = True

        return clip_support



    @property
    def disk_clipping_registry(self):
        r"""`torch.Tensor`: The disk clipping registry of the cropped fake CBED 
        pattern.

        See the summary documentation for the classes
        :class:`fakecbed.discretized.CBEDPattern`, and
        :class:`fakecbed.discretized.CroppedCBEDPattern` for additional context.

        Let ``num_disks``, ``disk_supports``, ``illumination_support``, and
        ``image`` denote the attributes
        :attr:`fakecbed.discretized.CroppedCBEDPattern.num_disks`,
        :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_supports`,
        :attr:`fakecbed.discretized.CroppedCBEDPattern.illumination_support`,
        and :attr:`fakecbed.discretized.CroppedCBEDPattern.image`
        respectively. Moreover, let ``k`` be a nonnegative integer less than
        ``num_disks``.

        ``disk_clipping_registry`` is a one-dimensional PyTorch tensor of length
        equal to ``num_disks``. If ``disk_clipping_registry[k]`` is equal to
        ``False``, then the image stored in ``disk_supports[k]`` has at least
        one nonzero pixel, and that the position of every nonzero pixel of said
        image is at least two pixels away from (i.e. at least pixel-wise
        next-nearest neighbours to) the position of every zero-valued pixel of
        the image stored in ``illumination_support``, at least two pixels away
        from the position of every pixel inside any of the mask frames that
        appears in the image stored in ``image``, and at least one pixel away
        from every pixel bordering the image stored in ``image``. Otherwise, if
        ``disk_clipping_registry[k]`` is equal to ``True``, then the opposite of
        the above scenario is true.

        Note that there are potentially two mask frames that may appear in the
        image stored in ``image``: the mask frame applied to the uncropped fake
        CBED pattern, and the mask frame that is applied after cropping the fake
        CBED pattern.

        Note that ``disk_clipping_registry`` should be considered **read-only**.

        """
        result = self.get_disk_clipping_registry(deep_copy=True)

        return result



    def get_disk_absence_registry(self, deep_copy=_default_deep_copy):
        r"""Return the disk absence registry of the cropped fake CBED pattern.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``disk_absence_registry`` denote the attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_absence_registry`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``disk_absence_registry`` is returned.  Otherwise, a reference to
            ``disk_absence_registry`` is returned.

        Returns
        -------
        disk_absence_registry : `torch.Tensor` (`bool`, ndim=1)
            The attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_absence_registry`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._disk_absence_registry is None:
            method_name = ("_calc_disk_absence_registry"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._disk_absence_registry = method_alias()

        disk_absence_registry = (self._disk_absence_registry.detach().clone()
                                 if (deep_copy == True)
                                 else self._disk_absence_registry)

        return disk_absence_registry


    
    def _calc_disk_absence_registry_and_cache_select_intermediates(self):
        num_disks = self._num_disks

        if num_disks > 0:
            if self._disk_supports is None:
                method_name = "_calc_disk_supports"
                method_alias = getattr(self, method_name)
                self._disk_supports = method_alias()
            disk_supports = self._disk_supports

            disk_absence_registry = (disk_supports.sum(dim=(1, 2)) == 0)
        else:
            disk_absence_registry = torch.zeros((num_disks,),
                                                 device=self._device,
                                                 dtype=torch.bool)
    
        return disk_absence_registry



    @property
    def disk_absence_registry(self):
        r"""`torch.Tensor`: The disk absence registry of the cropped fake CBED 
        pattern.

        See the summary documentation for the classes
        :class:`fakecbed.discretized.CBEDPattern`, and
        :class:`fakecbed.discretized.CroppedCBEDPattern` for additional context.

        Let ``num_disks``, and ``disk_supports`` denote the attributes
        :attr:`fakecbed.discretized.CroppedCBEDPattern.num_disks`, and
        :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_supports`
        respectively. Moreover, let ``k`` be a nonnegative integer less than
        ``num_disks``.

        ``disk_absence_registry`` is a one-dimensional PyTorch tensor of length
        equal to ``num_disks``. If ``disk_absence_registry[k]`` is equal to
        ``False``, then the image stored in ``disk_supports[k]`` has at least
        one nonzero pixel. Otherwise, if ``disk_absence_registry[k]`` is equal
        to ``True``, then the opposite of the above scenario is true.

        Note that ``disk_absence_registry`` should be considered **read-only**.

        """
        result = self.get_disk_absence_registry(deep_copy=True)

        return result



    @property
    def principal_disk_is_clipped(self):
        r"""`bool`: Equals ``True`` if the "principal" CBED disk either does not
        exists, or it exists and is clipped in image of the cropped fake CBED 
        pattern.

        See the summary documentation for the class
        :class:`fakecbed.discretized.CroppedCBEDPattern` for additional context,
        as well as the documentation for the attribute
        :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_clipping_registry`.

        Let ``core_attrs``, ``num_disks``, and ``disk_clipping_registry`` denote
        the attributes :attr:`~fancytypes.Checkable.core_attrs`,
        :attr:`fakecbed.discretized.CroppedCBEDPattern.num_disks`, and
        :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_clipping_registry`
        respectively.

        ``principal_disk_is_clipped`` is calculated effectively by:

        .. code-block:: python

           principal_disk_idx = \
               core_attrs["principal_disk_idx"]
           principal_disk_is_clipped = \
               (disk_clipping_registry[principal_disk_idx].item()
                if (principal_disk_idx < num_disks)
                else True)

        Note that ``principal_disk_is_clipped`` should be considered
        **read-only**.

        """
        if self._principal_disk_is_clipped is None:
            method_name = ("_calc_principal_disk_is_clipped"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._principal_disk_is_clipped = method_alias()

        result = self._principal_disk_is_clipped
        
        return result



    def _calc_principal_disk_is_clipped_and_cache_select_intermediates(self):
        principal_disk_idx = self._principal_disk_idx
        num_disks = self._num_disks

        if principal_disk_idx < num_disks:
            if self._disk_clipping_registry is None:
                method_name = ("_calc_disk_clipping_registry"
                               "_and_cache_select_intermediates")
                method_alias = getattr(self, method_name)
                self._disk_clipping_registry = method_alias()
            disk_clipping_registry = self._disk_clipping_registry

            disk_idx = principal_disk_idx
            principal_disk_is_clipped = disk_clipping_registry[disk_idx].item()
        else:
            principal_disk_is_clipped = True

        return principal_disk_is_clipped



    @property
    def principal_disk_is_absent(self):
        r"""`bool`: Equals ``True`` if the "principal" CBED disk either does not
        exists, or it exists and is absent from the image of the cropped fake 
        CBED pattern.

        See the summary documentation for the class
        :class:`fakecbed.discretized.CroppedCBEDPattern` for additional context,
        as well as the documentation for the attribute
        :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_absence_registry`.

        Let ``core_attrs``, ``num_disks``, and ``disk_absence_registry`` denote
        the attributes :attr:`~fancytypes.Checkable.core_attrs`,
        :attr:`fakecbed.discretized.CroppedCBEDPattern.num_disks`, and
        :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_absence_registry`
        respectively.

        ``principal_disk_is_absent`` is calculated effectively by:

        .. code-block:: python

           principal_disk_idx = \
               core_attrs["principal_disk_idx"]
           principal_disk_is_absent = \
               (disk_absence_registry[principal_disk_idx].item()
                if (principal_disk_idx < num_disks)
                else True)

        Note that ``principal_disk_is_absent`` should be considered
        **read-only**.

        """
        if self._principal_disk_is_absent is None:
            method_name = ("_calc_principal_disk_is_absent"
                           "_and_cache_select_intermediates")
            method_alias = getattr(self, method_name)
            self._principal_disk_is_absent = method_alias()

        result = self._principal_disk_is_absent
        
        return result


    
    def _calc_principal_disk_is_absent_and_cache_select_intermediates(self):
        principal_disk_idx = self._principal_disk_idx
        num_disks = self._num_disks

        if principal_disk_idx < num_disks:
            if self._disk_absence_registry is None:
                method_name = ("_calc_disk_absence_registry"
                               "_and_cache_select_intermediates")
                method_alias = getattr(self, method_name)
                self._disk_absence_registry = method_alias()
            disk_absence_registry = self._disk_absence_registry

            disk_idx = principal_disk_idx
            principal_disk_is_absent = disk_absence_registry[disk_idx].item()
        else:
            principal_disk_is_absent = True

        return principal_disk_is_absent



    def get_signal(self, deep_copy=_default_deep_copy):
        r"""Return the hyperspy signal representation of the cropped fake CBED 
        pattern.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``signal`` denote the attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.signal`.

            If ``deep_copy`` is set to ``True``, then a deep copy of ``signal``
            is returned.  Otherwise, a reference to ``signal`` is returned.

        Returns
        -------
        signal : :class:`hyperspy._signals.signal2d.Signal2D`
            The attribute 
            :attr:`fakecbed.discretized.CroppedCBEDPattern.signal`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._signal is None:
            method_name = "_calc_signal_and_cache_select_intermediates"
            method_alias = getattr(self, method_name)
            self._signal = method_alias()

        signal = (copy.deepcopy(self._signal)
                  if (deep_copy == True)
                  else self._signal)

        return signal



    def _calc_signal_and_cache_select_intermediates(self):
        method_name = "_calc_signal_metadata_and_cache_select_intermediates"
        method_alias = getattr(self, method_name)
        signal_metadata = method_alias()

        optional_params = self._generate_optional_cropping_params()

        kwargs = {"input_signal": \
                  self._cbed_pattern.get_signal(deep_copy=False),
                  "optional_params": \
                  optional_params}
        signal_data = empix.crop(**kwargs).data

        N_W_h, N_W_v = self._cropping_window_dims_in_pixels
        L, R, B, T = self._mask_frame  # Mask frame of cropped pattern.

        signal_data[:, :T, :] = 0
        signal_data[:, max(N_W_v-B, 0):, :] = 0
        signal_data[:, :, :L] = 0
        signal_data[:, :, max(N_W_h-R, 0):] = 0

        matrix_to_normalize = torch.from_numpy(signal_data[0])
        kwargs = {"input_matrix": matrix_to_normalize.to(device=self._device)}
        normalized_matrix = self._cbed_pattern._normalize_matrix(**kwargs)

        signal_data[0] = \
            normalized_matrix.cpu().detach().clone().numpy(force=True)

        signal = \
            hyperspy.signals.Signal2D(data=signal_data,
                                      metadata=signal_metadata)
        signal.axes_manager = \
            self._generate_cropped_blank_signal().axes_manager
        
        signal.axes_manager[0].name = \
            r"cropped fake CBED pattern attribute"
        signal.axes_manager[1].name = \
            r"fractional horizontal coordinate of uncropped fake CBED pattern"
        signal.axes_manager[2].name = \
            r"fractional vertical coordinate of uncropped fake CBED pattern"

        return signal



    def _calc_signal_metadata_and_cache_select_intermediates(self):
        cbed_pattern_signal = self._cbed_pattern.get_signal(deep_copy=False)

        title = "Cropped " + cbed_pattern_signal.metadata.General.title

        pre_serialized_core_attrs = self.pre_serialize()

        attr_name_subset = ("_principal_disk_analysis_results",
                            "_disk_clipping_registry",
                            "_disk_absence_registry",
                            "_principal_disk_is_clipped",
                            "_principal_disk_is_absent")
        for attr_name in attr_name_subset:
            attr = getattr(self, attr_name)
            if attr is None:
                method_name = ("_calc{}_and_cache_select"
                               "_intermediates").format(attr_name)
                method_alias = getattr(self, method_name)
                attr = method_alias()
                setattr(self, attr_name, attr)
        
        fakecbed_metadata = \
            cbed_pattern_signal.metadata.FakeCBED.as_dictionary()
        fakecbed_metadata["pre_serialized_core_attrs"] = \
            pre_serialized_core_attrs

        attr_name_subset += ("_principal_disk_bounding_box"
                             "_in_uncropped_image_fractional_coords",
                             "_principal_disk_bounding_box"
                             "_in_cropped_image_fractional_coords",
                             "_principal_disk_boundary_pts"
                             "_in_uncropped_image_fractional_coords",
                             "_principal_disk_boundary_pts"
                             "_in_cropped_image_fractional_coords")
        for attr_name in attr_name_subset[1:]:
            key = attr_name[1:]
            attr = getattr(self, attr_name)
            metadata_item = self._convert_attr_to_metadata_item(attr)
            fakecbed_metadata[key] = metadata_item

        signal_metadata = {"General": {"title": title},
                           "Signal": {"pixel value units": "dimensionless"},
                           "FakeCBED": fakecbed_metadata}

        return signal_metadata



    def _convert_attr_to_metadata_item(self, attr):
        if isinstance(attr, (tuple, bool)):
            metadata_item = attr
        else:
            if len(attr.shape) == 1:
                metadata_item = tuple(elem
                                      for elem
                                      in attr.cpu().detach().clone().tolist())
            else:
                metadata_item = tuple(tuple(pt)
                                      for pt
                                      in attr.cpu().detach().clone().tolist())

        return metadata_item



    @property
    def signal(self):
        r"""`hyperspy._signals.signal2d.Signal2D`: The hyperspy signal 
        representation of the cropped fake CBED pattern.

        See the summary documentation for the classes
        :class:`fakecbed.discretized.CBEDPattern` and
        :class:`fakecbed.discretized.CroppedCBEDPattern` for additional context.

        Let
        ``principal_disk_bounding_box_in_uncropped_image_fractional_coords``,
        ``principal_disk_bounding_box_in_cropped_image_fractional_coords``,
        ``principal_disk_boundary_pts_in_uncropped_image_fractional_coords``,
        ``principal_disk_boundary_pts_in_cropped_image_fractional_coords``, and
        ``core_attrs`` denote the attributes
        :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_bounding_box_in_uncropped_image_fractional_coords`,
        :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_bounding_box_in_cropped_image_fractional_coords`,
        :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_uncropped_image_fractional_coords`,
        :attr:`fakecbed.discretized.CroppedCBEDPattern.principal_disk_boundary_pts_in_cropped_image_fractional_coords`,
        and :attr:`~fancytypes.Checkable.core_attrs` respectively. Furthermore,
        let ``pre_serialize`` denote the method
        :meth:`~fancytypes.PreSerializable.pre_serialize`.

        ``signal`` is calculated effectively by:

        .. code-block:: python

           import empix
           import numpy as np

           cbed_pattern = core_attrs["cbed_pattern"]
           N_W_h, N_W_v = core_attrs["cropping_window_dims_in_pixels"]
           L, R, B, T = core_attrs["mask_frame"]

           cbed_pattern_signal = cbed_pattern.signal

           title = "Cropped " + cbed_pattern_signal.metadata.General.title

           cropping_window_center = \
               core_attrs["cropping_window_center"]
           cropping_window_dims_in_pixels = \
               core_attrs["cropping_window_dims_in_pixels"]

           kwargs = {"center": cropping_window_center,
                     "window_dims": cropping_window_dims_in_pixels,
                     "pad_mode": "zeros",
                     "apply_symmetric_mask": False}
           optional_params = empix.OptionalCroppingParams(**kwargs)

           kwargs = {"input_signal": cbed_pattern_signal,
                     "optional_params": optional_params}
           signal = empix.crop(**kwargs)

           signal.data[:, :T, :] = 0
           signal.data[:, max(N_W_v-B, 0):, :] = 0
           signal.data[:, :, :L] = 0
           signal.data[:, :, max(N_W_h-R, 0):] = 0

           # Normalize ``signal.data[0]``.
           matrix = signal.data[0]
           if matrix.max()-matrix.min() > 0:
               normalization_weight = 1 / (matrix.max()-matrix.min())
               normalization_bias = -normalization_weight*matrix.min()
               matrix = (matrix*normalization_weight
                         + normalization_bias).clip(min=0, max=1)
           else:
               matrix = np.zeros_like(matrix)
           signal.data[0] = matrix

           kwargs = {"item_path": "General.title", "value": title}
           signal.metadata.set_item(**kwargs)

           kwargs["item_path"] = "FakeCBED.pre_serialized_core_attrs"
           kwargs["value"] = pre_serialize()
           signal.metadata.set_item(**kwargs)

           tensor = \
               principal_disk_bounding_box_in_uncropped_image_fractional_coords
           kwargs["item_path"] = \
               ("FakeCBED.principal_disk_bounding_box"
                "_in_uncropped_image_fractional_coords")
           kwargs["value"] = \
               tuple(tensor.cpu().detach().clone().tolist())
           _ = \
               signal.metadata.set_item(**kwargs)

           tensor = \
               principal_disk_bounding_box_in_cropped_image_fractional_coords
           kwargs["item_path"] = \
               ("FakeCBED.principal_disk_bounding_box"
                "_in_cropped_image_fractional_coords")
           kwargs["value"] = \
               tuple(tensor.cpu().detach().clone().tolist())
           _ = \
               signal.metadata.set_item(**kwargs)

           tensor = \
               principal_disk_boundary_pts_in_uncropped_image_fractional_coords
           kwargs["item_path"] = \
               ("FakeCBED.principal_disk_boundary_pts"
                "_in_uncropped_image_fractional_coords")
           kwargs["value"] = \
               tuple(tensor.cpu().detach().clone().tolist())
           _ = \
               signal.metadata.set_item(**kwargs)

           tensor = \
               principal_disk_boundary_pts_in_cropped_image_fractional_coords
           kwargs["item_path"] = \
               ("FakeCBED.principal_disk_boundary_pts"
                "_in_cropped_image_fractional_coords")
           kwargs["value"] = \
               tuple(tensor.cpu().detach().clone().tolist())
           _ = \
               signal.metadata.set_item(**kwargs)

        Note that ``signal`` should be considered **read-only**.

        """
        result = self.get_signal(deep_copy=True)
        
        return result



    @property
    def image_has_been_overridden(self):
        r"""`bool`: Equals ``True`` if the image of the fake cropped CBED 
        pattern has been overridden.

        See the summary documentation for the classes
        :class:`fakecbed.discretized.CBEDPattern` and
        :class:`fakecbed.discretized.CroppedCBEDPattern` for additional context,
        as well as the documentation for the attribute
        :attr:`fakecbed.discretized.CBEDPattern.image_has_been_overridden`.

        Let ``core_attrs`` denote the attribute
        :attr:`~fancytypes.Checkable.core_attrs`.

        ``image_has_been_overridden`` is calculated effectively by:

        .. code-block:: python

           cbed_pattern = core_attrs["cbed_pattern"]
           image_has_been_overridden = cbed_pattern.image_has_been_overridden

        Note that ``image_has_been_overridden`` should be considered
        **read-only**.

        """
        result = self._image_has_been_overridden
        
        return result



    def get_image(self, deep_copy=_default_deep_copy):
        r"""Return the image of the cropped fake CBED pattern.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``image`` denote the attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.image`.

            If ``deep_copy`` is set to ``True``, then a deep copy of ``image``
            is returned.  Otherwise, a reference to ``image`` is returned.

        Returns
        -------
        image : `torch.Tensor` (`float`, ndim=2)
            The attribute :attr:`fakecbed.discretized.CroppedCBEDPattern.image`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._image is None:
            method_name = "_calc_image"
            method_alias = getattr(self, method_name)
            self._image = method_alias()

        image = (self._image.detach().clone()
                 if (deep_copy == True)
                 else self._image)

        return image



    def _calc_image(self):
        uncropped_cbed_pattern_image = \
            self._cbed_pattern.get_image(deep_copy=False)
        uncropped_cbed_pattern_image = \
            uncropped_cbed_pattern_image.cpu().detach().clone()

        signal_to_crop = \
            self._generate_uncropped_blank_signal()
        signal_to_crop.data[0] = \
            uncropped_cbed_pattern_image.numpy(force=True)

        optional_params = self._generate_optional_cropping_params()

        kwargs = {"input_signal": signal_to_crop,
                  "optional_params": optional_params}
        cropped_signal = empix.crop(**kwargs)

        image = cropped_signal.data[0]
        image_dtype = uncropped_cbed_pattern_image.dtype
        image = torch.from_numpy(image)
        image = image.to(device=self._device, dtype=image_dtype)

        N_W_h, N_W_v = self._cropping_window_dims_in_pixels
        L, R, B, T = self._mask_frame  # Mask frame of cropped pattern.

        image[:T, :] = 0
        image[max(N_W_v-B, 0):, :] = 0
        image[:, :L] = 0
        image[:, max(N_W_h-R, 0):] = 0

        kwargs = {"input_matrix": image}
        image = self._cbed_pattern._normalize_matrix(**kwargs)

        return image



    @property
    def image(self):
        r"""`torch.Tensor`: The image of the cropped fake CBED pattern.

        See the documentation for the attribute
        :attr:`fakecbed.discretized.CBEDPattern.signal` for additional
        context.

        Let ``signal``, ``device``, and ``core_attrs`` denote the attributes
        :attr:`fakecbed.discretized.CroppedCBEDPattern.signal`,
        :attr:`fakecbed.discretized.CroppedCBEDPattern.device`, and
        :attr:`~fancytypes.Checkable.core_attrs` respectively.

        ``image`` is calculated effectively by:

        .. code-block:: python

           N_W_h, N_W_v = core_attrs["cropping_window_dims_in_pixels"]
           L, R, B, T = core_attrs["mask_frame"]

           image = signal.data[0]
           image_dtype = image.dtype
           image = torch.from_numpy(image)
           image = image.to(device=device, dtype=image_dtype)
           image[:T, :] = 0
           image[max(N_W_v-B, 0):, :] = 0
           image[:, :L] = 0
           image[:, max(N_W_h-R, 0):] = 0

        Note that ``image`` should be considered **read-only**.

        """
        result = self.get_image(deep_copy=True)

        return result



    def get_illumination_support(self, deep_copy=_default_deep_copy):
        r"""Return the illumination support of the cropped fake CBED pattern.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``illumination_support`` denote the attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.illumination_support`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``illumination_support`` is returned.  Otherwise, a reference to
            ``illumination_support`` is returned.

        Returns
        -------
        illumination_support : `torch.Tensor` (`float`, ndim=2)
            The attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.illumination_support`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._illumination_support is None:
            method_name = "_calc_illumination_support"
            method_alias = getattr(self, method_name)
            self._illumination_support = method_alias()

        illumination_support = (self._illumination_support.detach().clone()
                 if (deep_copy == True)
                 else self._illumination_support)

        return illumination_support



    def _calc_illumination_support(self):
        uncropped_cbed_pattern_illumination_support = \
            self._cbed_pattern.get_illumination_support(deep_copy=False)
        uncropped_cbed_pattern_illumination_support = \
            uncropped_cbed_pattern_illumination_support.cpu().detach().clone()

        signal_to_crop = \
            self._generate_uncropped_blank_signal()
        signal_to_crop.data[1] = \
            uncropped_cbed_pattern_illumination_support.numpy(force=True)

        optional_params = self._generate_optional_cropping_params()

        kwargs = {"input_signal": signal_to_crop,
                  "optional_params": optional_params}
        cropped_signal = empix.crop(**kwargs)

        illumination_support = \
            cropped_signal.data[1]
        illumination_support_dtype = \
            uncropped_cbed_pattern_illumination_support.dtype
        illumination_support = \
            torch.from_numpy(illumination_support)
        illumination_support = \
            illumination_support.to(device=self._device,
                                    dtype=illumination_support_dtype)

        return illumination_support



    @property
    def illumination_support(self):
        r"""`torch.Tensor`: The illumination support of the cropped fake CBED 
        pattern.

        See the documentation for the attribute
        :attr:`fakecbed.discretized.CroppedCBEDPattern.signal` for additional
        context.

        Let ``signal``, and ``device`` denote the attributes
        :attr:`fakecbed.discretized.CroppedCBEDPattern.signal`, and
        :attr:`fakecbed.discretized.CroppedCBEDPattern.device` respectively.

        ``illumination_support`` is calculated effectively by:

        .. code-block:: python

           illumination_support = signal.data[1]
           illumination_support = torch.from_numpy(illumination_support)
           illumination_support = illumination_support.to(device=device, 
                                                          dtype=bool)

        Note that ``illumination_support`` should be considered **read-only**.

        """
        result = self.get_illumination_support(deep_copy=True)

        return result



    def get_disk_overlap_map(self, deep_copy=_default_deep_copy):
        r"""Return the image of the disk overlap map of the cropped fake CBED 
        pattern.

        Parameters
        ----------
        deep_copy : `bool`, optional
            Let ``disk_overlap_map`` denote the attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_overlap_map`.

            If ``deep_copy`` is set to ``True``, then a deep copy of
            ``disk_overlap_map`` is returned.  Otherwise, a reference to
            ``disk_overlap_map`` is returned.

        Returns
        -------
        disk_overlap_map : `torch.Tensor` (`int`, ndim=2)
            The attribute
            :attr:`fakecbed.discretized.CroppedCBEDPattern.disk_overlap_map`.

        """
        params = {"deep_copy": deep_copy}
        deep_copy = _check_and_convert_deep_copy(params)

        if self._disk_overlap_map is None:
            method_name = ("_calc_disk_overlap_map")
            method_alias = getattr(self, method_name)
            self._disk_overlap_map = method_alias()

        disk_overlap_map = (self._disk_overlap_map.detach().clone()
                            if (deep_copy == True)
                            else self._disk_overlap_map)

        return disk_overlap_map



    def _calc_disk_overlap_map(self):
        uncropped_cbed_pattern_disk_overlap_map = \
            self._cbed_pattern.get_disk_overlap_map(deep_copy=False)
        uncropped_cbed_pattern_disk_overlap_map = \
            uncropped_cbed_pattern_disk_overlap_map.cpu().detach().clone()

        signal_to_crop = \
            self._generate_uncropped_blank_signal()
        signal_to_crop.data[2] = \
            uncropped_cbed_pattern_disk_overlap_map.numpy(force=True)

        optional_params = self._generate_optional_cropping_params()

        kwargs = {"input_signal": signal_to_crop,
                  "optional_params": optional_params}
        cropped_signal = empix.crop(**kwargs)

        N_W_h, N_W_v = self._cropping_window_dims_in_pixels
        L, R, B, T = self._mask_frame  # Mask frame of cropped pattern.

        disk_overlap_map = cropped_signal.data[2]
        disk_overlap_map_dtype = uncropped_cbed_pattern_disk_overlap_map.dtype
        disk_overlap_map = torch.from_numpy(disk_overlap_map)
        disk_overlap_map = disk_overlap_map.to(device=self._device,
                                               dtype=disk_overlap_map_dtype)
        disk_overlap_map[:T, :] = 0
        disk_overlap_map[max(N_W_v-B, 0):, :] = 0
        disk_overlap_map[:, :L] = 0
        disk_overlap_map[:, max(N_W_h-R, 0):] = 0
    
        return disk_overlap_map



    @property
    def disk_overlap_map(self):
        r"""`torch.Tensor`: The image of the disk overlap map of the cropped 
        fake CBED pattern.

        See the documentation for the attribute
        :attr:`fakecbed.discretized.CroppedCBEDPattern.signal` for additional
        context.

        Let ``signal``, ``device``, and ``core_attrs`` denote the attributes
        :attr:`fakecbed.discretized.CroppedCBEDPattern.signal`,
        :attr:`fakecbed.discretized.CroppedCBEDPattern.device`, and
        :attr:`~fancytypes.Checkable.core_attrs` respectively.

        ``disk_overlap_map`` is calculated effectively by:

        .. code-block:: python

           N_W_h, N_W_v = core_attrs["cropping_window_dims_in_pixels"]
           L, R, B, T = core_attrs["mask_frame"]

           disk_overlap_map = signal.data[2]
           disk_overlap_map = torch.from_numpy(disk_overlap_map)
           disk_overlap_map = disk_overlap_map.to(device=device, dtype=int)
           disk_overlap_map[:T, :] = 0
           disk_overlap_map[max(N_W_v-B, 0):, :] = 0
           disk_overlap_map[:, :L] = 0
           disk_overlap_map[:, max(N_W_h-R, 0):] = 0

        Note that ``disk_overlap_map`` should be considered **read-only**.

        """
        result = self.get_disk_overlap_map(deep_copy=True)

        return result



###########################
## Define error messages ##
###########################

_check_and_convert_undistorted_disks_err_msg_1 = \
    ("The object ``undistorted_disks`` must be a sequence of "
     "`fakecbed.shapes.NonuniformBoundedShape` objects, where for each element "
     "``elem`` in ``undistorted_disks``, "
     "``isinstance(elem.core_attrs['support'], "
     "(fakecbed.shapes.Circle, fakecbed.shapes.Ellipse))`` evaluates to "
     "``True``.")

_check_and_convert_undistorted_misc_shapes_err_msg_1 = \
    ("The object ``undistorted_misc_shapes`` must be a sequence of objects of "
     "any of the following types: ("
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

_check_and_convert_distortion_model_err_msg_1 = \
    ("The dimensions, in units of pixels, of the distortion model sampling "
     "grid, specified by the object ``distortion_model``,  must be divisible "
     "by the object ``num_pixels_across_pattern``.")

_check_and_convert_rng_seed_err_msg_1 = \
    ("The object ``rng_seed`` must be either a nonnegative integer or of the "
     "type `NoneType`.")

_check_and_convert_cold_pixels_err_msg_1 = \
    ("The object ``cold_pixels`` must be a sequence of integer pairs, where "
     "each integer pair specifies valid pixel coordinates (i.e. row and column "
     "indices) of a pixel in the discretized fake CBED pattern.")

_check_and_convert_overriding_image_err_msg_1 = \
    ("The object ``overriding_image`` must have dimensions, in units of "
     "pixels, equal to those of the original CBED pattern intensity image "
     "being overridden, which in this case are ``({}, {})``.")

_cbed_pattern_err_msg_1 = \
    ("Failed to generate discretized fake CBED pattern. See traceback for "
     "details.")

_check_and_convert_cropping_window_center_err_msg_1 = \
    ("The object ``cropping_window_center`` must be `NoneType` or a pair of "
     "real numbers.")

_check_and_convert_cropping_window_dims_in_pixels_err_msg_1 = \
    ("The object ``cropping_window_dims_in_pixels`` must be `NoneType` or a "
     "pair of positive integers.")
