"""Insert a one-line description of module.

A more detailed description of the module. The more detailed description can
span multiple lines.

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

# For loading HDF5 datasubsets.
import h5pywrappers

# For validating, pre-serializing, and de-pre-serializing instances of the class
# :class:`distoptica.DistortionModel`. Also for extracting certain properties of
# distortion models.
import distoptica

# For downsampling images.
import skimage.measure



# For validating, pre-serializing, and de-pre-serializing instances of the
# classes :class:`fakecbed.shapes.Peak` and
# :class:`fakecbed.shapes.NonUniformDisk`.
import fakecbed.shapes

# For validating, pre-serializing, and de-pre-serializing instances of the
# class :class:`fakecbed.tds.Model`.
import fakecbed.tds



############################
## Authorship information ##
############################

__author__     = "Matthew Fitzpatrick"
__copyright__  = "Copyright 2023"
__credits__    = ["Matthew Fitzpatrick"]
__maintainer__ = "Matthew Fitzpatrick"
__email__      = "mrfitzpa@uvic.ca"
__status__     = "Development"



##################################
## Define classes and functions ##
##################################

# List of public objects in module.
__all__ = ["CBEDPatternParams",
           "CBEDPattern"]



def _check_and_convert_undistorted_tds_model(params):
    undistorted_tds_model = \
        fakecbed.tds._check_and_convert_undistorted_tds_model(params)

    return undistorted_tds_model



def _pre_serialize_undistorted_tds_model(undistorted_tds_model):
    serializable_rep = \
        fakecbed.tds._pre_serialize_undistorted_tds_model(undistorted_tds_model)
    
    return serializable_rep



def _de_pre_serialize_undistorted_tds_model(serializable_rep):
    undistorted_tds_model = \
        fakecbed.tds._de_pre_serialize_undistorted_tds_model(serializable_rep)

    return undistorted_tds_model



def _check_and_convert_undistorted_disks(params):
    try:
        obj_name = "undistorted_disks"
        undistorted_disks = params[obj_name]
        accepted_types = (fakecbed.shapes.NonUniformDisk,)

        for undistorted_disk in undistorted_disks:
            kwargs = {"obj": undistorted_disk,
                      "obj_name": "non_uniform_disk",
                      "accepted_types": accepted_types}
            czekitout.check.if_instance_of_any_accepted_types(**kwargs)
    except:
        raise TypeError(_check_and_convert_undistorted_disks_err_msg_1)

    return undistorted_disks



def _pre_serialize_undistorted_disks(undistorted_disks):
    serializable_rep = tuple()    
    for non_uniform_disk in undistorted_disks:
        serializable_rep += (non_uniform_disk.pre_serialize(),)
    
    return serializable_rep



def _de_pre_serialize_undistorted_disks(serializable_rep):
    undistorted_disks = tuple()    
    for serialized_non_uniform_disk in serializable_rep:
        cls = fakecbed.shapes.NonUniformDisk
        non_uniform_disk = cls.de_pre_serialize(serialized_non_uniform_disk)
        undistorted_disks += (non_uniform_disk,)

    return undistorted_disks



def _check_and_convert_undistorted_background_bands(params):
    try:
        obj_name = "undistorted_background_bands"
        undistorted_background_bands = params[obj_name]
        accepted_types = (fakecbed.shapes.Band,)

        for undistorted_background_band in undistorted_background_bands:
            kwargs = {"obj": undistorted_background_band,
                      "obj_name": "undistorted_background_band",
                      "accepted_types": accepted_types}
            czekitout.check.if_instance_of_any_accepted_types(**kwargs)
    except:
        err_msg = _check_and_convert_undistorted_background_bands_err_msg_1
        raise TypeError(err_msg)

    return undistorted_background_bands



def _pre_serialize_undistorted_background_bands(undistorted_background_bands):
    serializable_rep = tuple()
    for undistorted_background_band in undistorted_background_bands:
        serializable_rep += (undistorted_background_band.pre_serialize(),)
    
    return serializable_rep



def _de_pre_serialize_undistorted_background_bands(serializable_rep):
    undistorted_background_bands = \
        tuple()
    for serialized_undistorted_background_band in serializable_rep:
        cls = \
            fakecbed.shapes.Band
        undistorted_background_band = \
            cls.de_pre_serialize(serialized_undistorted_background_band)
        undistorted_background_bands += \
            (undistorted_background_band,)

    return undistorted_background_bands



def _check_and_convert_disk_support_gaussian_filter_std_dev(params):
    obj_name = \
        "disk_support_gaussian_filter_std_dev"
    obj = \
        params[obj_name]
    disk_support_gaussian_filter_std_dev = \
        czekitout.convert.to_nonnegative_float(obj, obj_name)

    return disk_support_gaussian_filter_std_dev



def _pre_serialize_disk_support_gaussian_filter_std_dev(
        disk_support_gaussian_filter_std_dev):
    serializable_rep = disk_support_gaussian_filter_std_dev
    
    return serializable_rep



def _de_pre_serialize_disk_support_gaussian_filter_std_dev(serializable_rep):
    disk_support_gaussian_filter_std_dev = serializable_rep

    return disk_support_gaussian_filter_std_dev



def _check_and_convert_intra_disk_gaussian_filter_std_dev(params):
    obj_name = \
        "intra_disk_gaussian_filter_std_dev"
    obj = \
        params[obj_name]
    intra_disk_gaussian_filter_std_dev = \
        czekitout.convert.to_nonnegative_float(obj, obj_name)

    return intra_disk_gaussian_filter_std_dev



def _pre_serialize_intra_disk_gaussian_filter_std_dev(
        intra_disk_gaussian_filter_std_dev):
    serializable_rep = intra_disk_gaussian_filter_std_dev
    
    return serializable_rep



def _de_pre_serialize_intra_disk_gaussian_filter_std_dev(serializable_rep):
    intra_disk_gaussian_filter_std_dev = serializable_rep

    return intra_disk_gaussian_filter_std_dev



def _check_and_convert_distortion_model(params):
    distortion_model = \
        distoptica._check_and_convert_distortion_model(params)

    return distortion_model



def _pre_serialize_distortion_model(distortion_model):
    serializable_rep = \
        distoptica._pre_serialize_distortion_model(distortion_model)
    
    return serializable_rep



def _de_pre_serialize_distortion_model(serializable_rep):
    distortion_model = \
        distoptica._de_pre_serialize_distortion_model(serializable_rep)

    return distortion_model



def _check_and_convert_apply_shot_noise(params):
    obj_name = "apply_shot_noise"
    obj = params[obj_name]
    apply_shot_noise = czekitout.convert.to_bool(obj, obj_name)
    
    return apply_shot_noise



def _pre_serialize_apply_shot_noise(apply_shot_noise):
    serializable_rep = apply_shot_noise

    return serializable_rep



def _de_pre_serialize_apply_shot_noise(serializable_rep):
    apply_shot_noise = serializable_rep

    return apply_shot_noise



def _check_and_convert_cold_pixels(params):
    obj_name = "cold_pixels"
    obj = params[obj_name]
    cold_pixels = czekitout.convert.to_pairs_of_ints(obj, obj_name)
    coords_of_cold_pixels = cold_pixels

    num_pixels_across_pattern = \
        _check_and_convert_num_pixels_across_pattern(params)

    for coords_of_cold_pixel in coords_of_cold_pixels:
        row, col = coords_of_cold_pixel
        if ((row < -num_pixels_across_pattern)
            or (num_pixels_across_pattern <= row)
            or (col < -num_pixels_across_pattern)
            or (num_pixels_across_pattern <= col)):
            raise ValueError(_check_and_convert_cold_pixels_err_msg_1)

    return cold_pixels



def _pre_serialize_cold_pixels(cold_pixels):
    serializable_rep = cold_pixels
    
    return serializable_rep



def _de_pre_serialize_cold_pixels(serializable_rep):
    cold_pixels = serializable_rep

    return cold_pixels



def _check_and_convert_num_pixels_across_pattern(params):
    obj_name = "num_pixels_across_pattern"
    obj = params[obj_name]
    num_pixels_across_pattern = czekitout.convert.to_positive_int(obj, obj_name)

    return num_pixels_across_pattern



def _pre_serialize_num_pixels_across_pattern(num_pixels_across_pattern):
    serializable_rep = num_pixels_across_pattern
    
    return serializable_rep



def _de_pre_serialize_num_pixels_across_pattern(serializable_rep):
    num_pixels_across_pattern = serializable_rep

    return num_pixels_across_pattern



def _check_and_convert_num_samples_across_each_pixel(params):
    obj_name = "num_samples_across_each_pixel"
    obj = params[obj_name]
    num_samples_across_each_pixel = czekitout.convert.to_positive_int(obj,
                                                                      obj_name)

    return num_samples_across_each_pixel



def _pre_serialize_num_samples_across_each_pixel(num_samples_across_each_pixel):
    serializable_rep = num_samples_across_each_pixel
    
    return serializable_rep



def _de_pre_serialize_num_samples_across_each_pixel(serializable_rep):
    num_samples_across_each_pixel = serializable_rep

    return num_samples_across_each_pixel



def _check_and_convert_mask_frame(params):
    obj_name = "mask_frame"
    obj = params[obj_name]
    mask_frame = czekitout.convert.to_quadruplet_of_nonnegative_ints(obj,
                                                                     obj_name)

    return mask_frame



def _pre_serialize_mask_frame(mask_frame):
    serializable_rep = mask_frame
    
    return serializable_rep



def _de_pre_serialize_mask_frame(serializable_rep):
    mask_frame = serializable_rep

    return mask_frame



_default_undistorted_tds_model = fakecbed.tds._default_undistorted_tds_model
_default_undistorted_disks = tuple()
_default_undistorted_background_bands = tuple()
_default_disk_support_gaussian_filter_std_dev = 0
_default_intra_disk_gaussian_filter_std_dev = 0
_default_distortion_model = None
_default_apply_shot_noise = False
_default_cold_pixels = tuple()
_default_num_pixels_across_pattern = 512
_default_num_samples_across_each_pixel = 1
_default_mask_frame = (0, 0, 0, 0)



class CBEDPatternParams(fancytypes.PreSerializableAndUpdatable):
    r"""Insert description here.

    Parameters
    ----------
    undistorted_tds_model : :class:`fakecbed.tds.Model` | `None`, optional
        Insert description here.
    undistorted_disks : `array_like` (:class:`fakecbed.shapes.NonUniformDisk`, ndim=1), optional
        Insert description here.
    undistorted_background_bands : `array_like` (:class:`fakecbed.shapes.Band`, ndim=1), optional
        Insert description here.
    disk_support_gaussian_filter_std_dev : `float`, optional
        Insert description here.
    intra_disk_gaussian_filter_std_dev : `float`, optional
        Insert description here.
    distortion_model : :class:`distoptica.DistortionModel` | `None`, optional
        Insert description here.
    apply_shot_noise : `bool`, optional
        Insert description here.
    cold_pixels : `int`, optional
        Insert description here.
    num_pixels_across_pattern : `int`, optional
        Insert description here.
    num_samples_across_each_pixel : `int`, optional
        Insert description here.
    mask_frame : `array_like` (`int`, shape=(``4``,)), optional
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
        {"undistorted_tds_model": _check_and_convert_undistorted_tds_model,
         "undistorted_disks": _check_and_convert_undistorted_disks,
         "undistorted_background_bands": \
         _check_and_convert_undistorted_background_bands,
         "disk_support_gaussian_filter_std_dev": \
         _check_and_convert_disk_support_gaussian_filter_std_dev,
         "intra_disk_gaussian_filter_std_dev": \
         _check_and_convert_intra_disk_gaussian_filter_std_dev,
         "distortion_model": _check_and_convert_distortion_model,
         "apply_shot_noise": _check_and_convert_apply_shot_noise,
         "cold_pixels": \
         _check_and_convert_cold_pixels,
         "num_pixels_across_pattern": \
         _check_and_convert_num_pixels_across_pattern,
         "num_samples_across_each_pixel": \
         _check_and_convert_num_samples_across_each_pixel,
         "mask_frame": _check_and_convert_mask_frame}

    _pre_serialization_funcs = \
        {"undistorted_tds_model": _pre_serialize_undistorted_tds_model,
         "undistorted_disks": _pre_serialize_undistorted_disks,
         "undistorted_background_bands": \
         _pre_serialize_undistorted_background_bands,
         "disk_support_gaussian_filter_std_dev": \
         _pre_serialize_disk_support_gaussian_filter_std_dev,
         "intra_disk_gaussian_filter_std_dev": \
         _pre_serialize_intra_disk_gaussian_filter_std_dev,
         "distortion_model": _pre_serialize_distortion_model,
         "apply_shot_noise": _pre_serialize_apply_shot_noise,
         "cold_pixels": _pre_serialize_cold_pixels,
         "num_pixels_across_pattern": _pre_serialize_num_pixels_across_pattern,
         "num_samples_across_each_pixel": \
         _pre_serialize_num_samples_across_each_pixel,
         "mask_frame": _pre_serialize_mask_frame}

    _de_pre_serialization_funcs = \
        {"undistorted_tds_model": _de_pre_serialize_undistorted_tds_model,
         "undistorted_disks": _de_pre_serialize_undistorted_disks,
         "undistorted_background_bands": \
         _de_pre_serialize_undistorted_background_bands,
         "disk_support_gaussian_filter_std_dev": \
         _de_pre_serialize_disk_support_gaussian_filter_std_dev,
         "intra_disk_gaussian_filter_std_dev": \
         _de_pre_serialize_intra_disk_gaussian_filter_std_dev,
         "distortion_model": _de_pre_serialize_distortion_model,
         "apply_shot_noise": _de_pre_serialize_apply_shot_noise,
         "cold_pixels": _de_pre_serialize_cold_pixels,
         "num_pixels_across_pattern": \
         _de_pre_serialize_num_pixels_across_pattern,
         "num_samples_across_each_pixel": \
         _de_pre_serialize_num_samples_across_each_pixel,
         "mask_frame": _de_pre_serialize_mask_frame}

    def __init__(self,
                 undistorted_tds_model=\
                 _default_undistorted_tds_model,
                 undistorted_disks=\
                 _default_undistorted_disks,
                 undistorted_background_bands=\
                 _default_undistorted_background_bands,
                 disk_support_gaussian_filter_std_dev=\
                 _default_disk_support_gaussian_filter_std_dev,
                 intra_disk_gaussian_filter_std_dev=\
                 _default_intra_disk_gaussian_filter_std_dev,
                 distortion_model=\
                 _default_distortion_model,
                 apply_shot_noise=\
                 _default_apply_shot_noise,
                 cold_pixels=\
                 _default_cold_pixels,
                 num_pixels_across_pattern=\
                 _default_num_pixels_across_pattern,
                 num_samples_across_each_pixel=\
                 _default_num_samples_across_each_pixel,
                 mask_frame=\
                 _default_mask_frame):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        return None



def _check_and_convert_cbed_pattern_params(params):
    obj_name = "cbed_pattern_params"
    obj = copy.deepcopy(params[obj_name])
    
    if obj is None:
        cbed_pattern_params = CBEDPatternParams()
    else:
        accepted_types = (CBEDPatternParams, type(None))
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        cbed_pattern_params = obj

    return cbed_pattern_params



def _pre_serialize_cbed_pattern_params(cbed_pattern_params):
    serializable_rep = cbed_pattern_params.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_cbed_pattern_params(serializable_rep):
    cbed_pattern_params = CBEDPatternParams.de_pre_serialize(serializable_rep)

    return cbed_pattern_params



def _x_and_y_data(cbed_pattern_params, i_range, j_range, resolution):
    i_subrange, j_subrange = _i_and_j_subranges(cbed_pattern_params,
                                                i_range,
                                                j_range,
                                                resolution)

    func_alias = _tilde_x_and_tilde_y_data_subviews
    tilde_x_data_subview, tilde_y_data_subview = func_alias(cbed_pattern_params,
                                                            i_subrange,
                                                            j_subrange,
                                                            resolution)

    distortion_model = \
        cbed_pattern_params._core_attrs["distortion_model"]
    obj_alias = \
        distortion_model
    method_alias = \
        obj_alias._map_to_fractional_cartesian_coords_of_undistorted_image
    x_data_subview, y_data_subview, _, _ = \
        method_alias(q_x=tilde_x_data_subview, q_y=tilde_y_data_subview)

    multi_dim_slice = _instance_of_first_kind_of_multi_dim_slice(i_range,
                                                                 i_subrange,
                                                                 j_range,
                                                                 j_subrange)

    x_data_shape = (len(i_range), len(j_range))
    x_data = torch.zeros(x_data_shape, device=x_data_subview.device)
    x_data[multi_dim_slice] = x_data_subview
    
    y_data = torch.zeros_like(x_data)
    y_data[multi_dim_slice] = y_data_subview

    return x_data, y_data



def _i_and_j_subranges(cbed_pattern_params, i_range, j_range, resolution):
    mask_frame = cbed_pattern_params._core_attrs["mask_frame"]
    L, R, B, T = mask_frame

    key = "num_pixels_across_pattern"
    N_1 = cbed_pattern_params._core_attrs[key]
    M_1 = N_1

    if resolution == "hd":
        key = "num_samples_across_each_pixel"
        num_samples_across_each_pixel = cbed_pattern_params._core_attrs[key]
        M_2 = num_samples_across_each_pixel
    else:
        M_2 = 1

    i_subrange_beg_candidate_1 = T*M_2
    i_subrange_beg_candidate_2 = i_range[0]
    i_subrange_beg = max(i_subrange_beg_candidate_1, i_subrange_beg_candidate_2)

    i_subrange_end_candidate_1 = M_1*M_2 - B*M_2
    i_subrange_end_candidate_2 = i_range[-1] + 1
    i_subrange_end = min(i_subrange_end_candidate_1, i_subrange_end_candidate_2)

    i_subrange = torch.arange(i_subrange_beg,
                              i_subrange_end,
                              device=i_range.device)

    j_subrange_beg_candidate_1 = L*M_2
    j_subrange_beg_candidate_2 = j_range[0]
    j_subrange_beg = max(j_subrange_beg_candidate_1, j_subrange_beg_candidate_2)

    j_subrange_end_candidate_1 = M_1*M_2 - R*M_2
    j_subrange_end_candidate_2 = j_range[-1] + 1
    j_subrange_end = min(j_subrange_end_candidate_1, j_subrange_end_candidate_2)
    
    j_subrange = torch.arange(j_subrange_beg,
                              j_subrange_end,
                              device=j_range.device)

    return i_subrange, j_subrange



def _tilde_x_and_tilde_y_data_subviews(cbed_pattern_params,
                                       i_subrange,
                                       j_subrange,
                                       resolution):
    key = "num_pixels_across_pattern"
    N_1 = cbed_pattern_params._core_attrs[key]
    M_1 = N_1

    if resolution == "hd":
        key = "num_samples_across_each_pixel"
        num_samples_across_each_pixel = cbed_pattern_params._core_attrs[key]
        M_2 = num_samples_across_each_pixel
    else:
        M_2 = 1
        
    tilde_x_coords_of_grid = \
        (j_subrange + 0.5)/(M_1*M_2)
    tilde_y_coords_of_grid = \
        1 - (i_subrange + 0.5)/(M_1*M_2)
    pair_of_1d_coord_arrays = \
        (tilde_x_coords_of_grid, tilde_y_coords_of_grid)

    tilde_x_data_subview, tilde_y_data_subview = \
        torch.meshgrid(*pair_of_1d_coord_arrays, indexing="xy")

    return tilde_x_data_subview, tilde_y_data_subview



def _instance_of_first_kind_of_multi_dim_slice(i_range,
                                               i_subrange,
                                               j_range,
                                               j_subrange):
    if len(i_subrange) > 0:
        single_dim_slice_1_start = i_subrange[0] - i_range[0]
        single_dim_slice_1_stop = i_subrange[0] - i_range[0] + len(i_subrange)
        single_dim_slice_1 = slice(single_dim_slice_1_start,
                                   single_dim_slice_1_stop)
    else:
        single_dim_slice_1 = slice(0, 0)

    if len(j_subrange) > 0:
        single_dim_slice_2_start = j_subrange[0] - j_range[0]
        single_dim_slice_2_stop = j_subrange[0] - j_range[0] + len(j_subrange)
        single_dim_slice_2 = slice(single_dim_slice_2_start,
                                   single_dim_slice_2_stop)
    else:
        single_dim_slice_2 = slice(0, 0)
        
    multi_dim_slice = (single_dim_slice_1, single_dim_slice_2)

    return multi_dim_slice



def _ld_misc_data_and_multi_dim_slice_set(cbed_pattern_params,
                                          hd_i_range,
                                          hd_j_range,
                                          hd_x_data,
                                          hd_y_data):
    ld_i_range, ld_j_range = _ld_i_and_j_range(cbed_pattern_params,
                                               hd_i_range,
                                               hd_j_range)

    kwargs = {"cbed_pattern_params": cbed_pattern_params,
              "i_range": ld_i_range,
              "j_range": ld_j_range,
              "resolution": "ld"}
    ld_x_data, ld_y_data = _x_and_y_data(**kwargs)
    ld_mask_frame_data = _mask_frame_data(**kwargs)

    kwargs = {"cbed_pattern_params": cbed_pattern_params,
              "x_data": ld_x_data,
              "y_data": ld_y_data}
    ld_bg_data = _bg_data(**kwargs)
    
    kwargs = {"cbed_pattern_params": cbed_pattern_params,
              "hd_x_data": hd_x_data,
              "hd_y_data": hd_y_data,
              "ld_x_data": ld_x_data,
              "ld_y_data": ld_y_data,
              "ld_mask_frame_data": ld_mask_frame_data}
    func_alias = _ld_disk_support_data_and_multi_dim_slice_set
    ld_disk_support_data, ld_multi_dim_slice_set = func_alias(**kwargs)
    
    kwargs = {"cbed_pattern_params": cbed_pattern_params,
              "x_data": ld_x_data,
              "y_data": ld_y_data,
              "multi_dim_slice_set": ld_multi_dim_slice_set}
    kwargs["multi_dim_slice_set"] = ld_multi_dim_slice_set
    ld_intra_disk_data = _intra_disk_data(**kwargs)

    ld_misc_data = (ld_bg_data,
                    ld_disk_support_data,
                    ld_intra_disk_data,
                    ld_mask_frame_data)

    return ld_misc_data, ld_multi_dim_slice_set



def _ld_i_and_j_range(cbed_pattern_params, hd_i_range, hd_j_range):
    num_samples_across_each_pixel = \
        cbed_pattern_params._core_attrs["num_samples_across_each_pixel"]

    hd_i_min = hd_i_range[0]
    hd_i_max = hd_i_range[-1]

    ld_i_min = hd_i_min//num_samples_across_each_pixel
    ld_i_max = ((hd_i_max+1)//num_samples_across_each_pixel)-1
    ld_i_range = torch.arange(ld_i_min, ld_i_max+1, device=ld_i_min.device)

    hd_j_min = hd_j_range[0]
    hd_j_max = hd_j_range[-1]

    ld_j_min = hd_j_min//num_samples_across_each_pixel
    ld_j_max = ((hd_j_max+1)//num_samples_across_each_pixel)-1
    ld_j_range = torch.arange(ld_j_min, ld_j_max+1, device=ld_j_min.device)

    return ld_i_range, ld_j_range



def _mask_frame_data(cbed_pattern_params, i_range, j_range, resolution):
    i_subrange, j_subrange = _i_and_j_subranges(cbed_pattern_params,
                                                i_range,
                                                j_range,
                                                resolution)

    multi_dim_slice = _instance_of_first_kind_of_multi_dim_slice(i_range,
                                                                 i_subrange,
                                                                 j_range,
                                                                 j_subrange)

    mask_frame_data_shape = (len(i_range), len(j_range))
    mask_frame_data = torch.zeros(mask_frame_data_shape,
                                  dtype=bool,
                                  device=i_range.device)
    mask_frame_data[multi_dim_slice] = True

    return mask_frame_data



def _bg_data(cbed_pattern_params, x_data, y_data):
    undistorted_tds_model = \
        cbed_pattern_params._core_attrs["undistorted_tds_model"]
    undistorted_background_bands = \
        cbed_pattern_params._core_attrs["undistorted_background_bands"]
    bg_data = \
        undistorted_tds_model._eval(x=x_data, y=y_data)
        
    for undistorted_background_band in undistorted_background_bands:
        bg_data += undistorted_background_band._eval(x=x_data, y=y_data)

    return bg_data



def _ld_disk_support_data_and_multi_dim_slice_set(cbed_pattern_params,
                                                  hd_x_data,
                                                  hd_y_data,
                                                  ld_x_data,
                                                  ld_y_data,
                                                  ld_mask_frame_data):
    undistorted_disks = cbed_pattern_params._core_attrs["undistorted_disks"]
    num_disks = len(undistorted_disks)
    ld_disk_support_data_shape = (max(num_disks, 1),)+ld_x_data.shape
    ld_disk_support_data = torch.zeros(ld_disk_support_data_shape,
                                       device=ld_x_data.device)
        
    ld_multi_dim_slice_set = (tuple()
                              if num_disks > 0
                              else ((0, slice(None), slice(None)),))
        
    for disk_idx, _ in enumerate(undistorted_disks):
        func_alias = _ld_support_data_of_disk_of_interest
        kwargs = {"cbed_pattern_params": cbed_pattern_params,
                  "hd_x_data": hd_x_data,
                  "hd_y_data": hd_y_data,
                  "ld_x_data": ld_x_data,
                  "ld_y_data": ld_y_data,
                  "disk_idx": disk_idx,
                  "ld_mask_frame_data": ld_mask_frame_data}
        ld_support_data_of_disk_of_interest = func_alias(**kwargs)
        ld_disk_support_data[disk_idx] = ld_support_data_of_disk_of_interest

        func_alias = _instance_of_third_kind_of_multi_dim_slice
        kwargs = {"cbed_pattern_params": \
                  cbed_pattern_params,
                  "disk_idx": \
                  disk_idx,
                  "ld_support_data_of_disk_of_interest": \
                  ld_disk_support_data[disk_idx]}
        multi_dim_slice = func_alias(**kwargs)
        ld_multi_dim_slice_set += (multi_dim_slice,)

    return ld_disk_support_data, ld_multi_dim_slice_set



def _ld_support_data_of_disk_of_interest(
        cbed_pattern_params,
        hd_x_data,
        hd_y_data,
        ld_x_data,
        ld_y_data,
        disk_idx,
        ld_mask_frame_data):
    undistorted_disks = cbed_pattern_params._core_attrs["undistorted_disks"]
    undistorted_disk = undistorted_disks[disk_idx]

    key = "num_samples_across_each_pixel"
    num_samples_across_each_pixel = cbed_pattern_params._core_attrs[key]

    kwargs = \
        {"x": ld_x_data, "y": ld_y_data}
    approx_ld_support_data_of_disk_of_interest = \
        1.0*undistorted_disk._eval_without_intra_disk_shapes(**kwargs)
    approx_ld_support_data_of_disk_of_interest *= \
        ld_mask_frame_data

    kwargs = {"input_tensor": approx_ld_support_data_of_disk_of_interest}
    ld_edge_matrix = _ld_edge_matrix(**kwargs)

    ld_rows, ld_cols = torch.where(ld_edge_matrix)

    ld_support_data_of_disk_of_interest = \
        approx_ld_support_data_of_disk_of_interest
    
    for ld_row, ld_col in zip(ld_rows, ld_cols):
        kwargs = \
            {"ld_row": ld_row.item(),
             "ld_col": ld_col.item(),
             "hd_x_data": hd_x_data,
             "hd_y_data": hd_y_data,
             "undistorted_disk": undistorted_disk,
             "num_samples_across_each_pixel": num_samples_across_each_pixel}
        val_of_ld_pixel_to_update = \
            _val_of_ld_pixel_to_update(**kwargs)
        
        ld_support_data_of_disk_of_interest[(ld_row, ld_col)] = \
            val_of_ld_pixel_to_update

    ld_support_data_of_disk_of_interest *= ld_mask_frame_data

    return ld_support_data_of_disk_of_interest



def _ld_edge_matrix(input_tensor):
    kwargs = {"input": input_tensor,
              "pad": (1, 1),
              "mode": "constant",
              "value": 0}
    input_tensor_after_padding = torch.nn.functional.pad(**kwargs)

    input_tensor_after_padding_and_unsqueezing = \
        torch.unsqueeze(input_tensor_after_padding, dim=0)
    input_tensor_after_padding_and_unsqueezing = \
        torch.unsqueeze(input_tensor_after_padding_and_unsqueezing, dim=0)

    weights = torch.ones((1, 1, 3, 3), device=input_tensor.device)
    num_weights = weights.numel()

    kwargs = \
        {"input": input_tensor_after_padding_and_unsqueezing,
         "weight": weights,
         "padding": 0}
    ld_edge_matrix = \
        ((torch.nn.functional.conv2d(**kwargs) % num_weights) > 0)[0, 0]

    return ld_edge_matrix



def _val_of_ld_pixel_to_update(ld_row,
                               ld_col,
                               hd_x_data,
                               hd_y_data,
                               undistorted_disk,
                               num_samples_across_each_pixel):
    func_alias = _instance_of_second_kind_of_multi_dim_slice
    kwargs = {"ld_row": ld_row,
              "ld_col": ld_col,
              "num_samples_across_each_pixel": num_samples_across_each_pixel}
    hd_multi_dim_slice = func_alias(**kwargs)

    method_alias = undistorted_disk._eval_without_intra_disk_shapes
    kwargs = {"x": hd_x_data[hd_multi_dim_slice],
              "y": hd_y_data[hd_multi_dim_slice]}
    val_of_ld_pixel_to_update = (1.0*method_alias(**kwargs)).mean()

    return val_of_ld_pixel_to_update



def _instance_of_second_kind_of_multi_dim_slice(ld_row,
                                                ld_col,
                                                num_samples_across_each_pixel):
    single_dim_slice_1 = slice(ld_row*num_samples_across_each_pixel,
                               (ld_row+1)*num_samples_across_each_pixel)
    single_dim_slice_2 = slice(ld_col*num_samples_across_each_pixel,
                               (ld_col+1)*num_samples_across_each_pixel)
    multi_dim_slice = (single_dim_slice_1, single_dim_slice_2)

    return multi_dim_slice



def _instance_of_third_kind_of_multi_dim_slice(
        cbed_pattern_params,
        disk_idx,
        ld_support_data_of_disk_of_interest):
    undistorted_disks = cbed_pattern_params._core_attrs["undistorted_disks"]
    undistorted_disk = undistorted_disks[disk_idx]
        
    key = "num_pixels_across_pattern"
    N_1 = cbed_pattern_params._core_attrs[key]
    M_1 = N_1

    key = "disk_support_gaussian_filter_std_dev"
    sigma = cbed_pattern_params._core_attrs[key]
    M_3 = round(_truncate*sigma)

    rows_are_nonzero = torch.any(ld_support_data_of_disk_of_interest, dim=1)+0
    cols_are_nonzero = torch.any(ld_support_data_of_disk_of_interest, dim=0)+0

    single_dim_slice_1_start = \
        max(rows_are_nonzero.argmax() - M_3, 0)
    single_dim_slice_1_stop = \
        min(M_1-torch.flip(rows_are_nonzero, dims=(0,)).argmax() + M_3, M_1)
    single_dim_slice_1 = \
        slice(single_dim_slice_1_start, single_dim_slice_1_stop)

    single_dim_slice_2_start = \
        max(cols_are_nonzero.argmax() - M_3, 0)
    single_dim_slice_2_stop = \
        min(M_1-torch.flip(cols_are_nonzero, dims=(0,)).argmax() + M_3, M_1)
    single_dim_slice_2 = \
        slice(single_dim_slice_2_start, single_dim_slice_2_stop)

    multi_dim_slice = (disk_idx, single_dim_slice_1, single_dim_slice_2)

    return multi_dim_slice



# Fixed value used for parameter ``truncate`` our custom ``torch``
# implementation the function ``scipy.ndimage.gaussian_filter``.
_truncate = 4



def _intra_disk_data(cbed_pattern_params,
                     x_data,
                     y_data,
                     multi_dim_slice_set):
    undistorted_disks = cbed_pattern_params._core_attrs["undistorted_disks"]
    num_disks = len(undistorted_disks)
    intra_disk_data_shape = (max(num_disks, 1),)+x_data.shape
    intra_disk_data = torch.zeros(intra_disk_data_shape,
                                  device=x_data.device)

    for disk_idx, undistorted_disk in enumerate(undistorted_disks):
        multi_dim_slice = multi_dim_slice_set[disk_idx]

        kwargs = \
            {"x": x_data[multi_dim_slice[1:]],
             "y": y_data[multi_dim_slice[1:]]}
        intra_disk_data[multi_dim_slice] += \
            undistorted_disk._eval_without_support(**kwargs)

    return intra_disk_data



def _signal_data(cbed_pattern_params,
                 ld_misc_data,
                 ld_multi_dim_slice_set,
                 cold_pixels):
    ld_disk_support_data = ld_misc_data[1] + 0.0  # Lazy way to copy.
    num_disks = len(ld_disk_support_data)

    ld_disk_overlap_map_data = _ld_disk_overlap_map_data(ld_misc_data)
    
    ld_fg_support_data = 1.0*(ld_disk_overlap_map_data > 0)

    ld_cbed_data = _ld_cbed_data(cbed_pattern_params,
                                 ld_misc_data,
                                 ld_multi_dim_slice_set,
                                 cold_pixels)

    signal_data_shape = (num_disks+3,) + ld_cbed_data.shape
    signal_data = np.zeros(signal_data_shape)
    signal_data[0] = ld_cbed_data.cpu().detach().numpy()
    signal_data[1] = ld_disk_overlap_map_data.cpu().detach().numpy()
    signal_data[2] = ld_fg_support_data.cpu().detach().numpy()
    signal_data[3:] = ld_disk_support_data.cpu().detach().numpy()
    
    return signal_data



def _ld_disk_overlap_map_data(ld_misc_data):
    _, ld_disk_support_data, _, ld_mask_frame_data = ld_misc_data

    ld_disk_overlap_map_data = (1.0
                                * ld_mask_frame_data
                                * torch.sum(ld_disk_support_data>0, dim=0))

    return ld_disk_overlap_map_data



def _ld_cbed_data(cbed_pattern_params,
                  ld_misc_data,
                  ld_multi_dim_slice_set,
                  cold_pixels):
    ld_bg_data, ld_disk_support_data, ld_intra_disk_data, ld_mask_frame_data = \
        ld_misc_data

    func_alias = _blurred_ld_disk_support_data
    kwargs = {"cbed_pattern_params": cbed_pattern_params,
              "ld_misc_data": ld_misc_data,
              "ld_multi_dim_slice_set": ld_multi_dim_slice_set}
    blurred_ld_disk_support_data = func_alias(**kwargs)

    func_alias = _blurred_ld_intra_disk_data
    blurred_ld_intra_disk_data = func_alias(**kwargs)

    func_alias = _maskless_and_noiseless_ld_cbed_data
    kwargs = {"ld_misc_data": ld_misc_data,
              "ld_multi_dim_slice_set": ld_multi_dim_slice_set,
              "blurred_ld_disk_support_data": blurred_ld_disk_support_data,
              "blurred_ld_intra_disk_data": blurred_ld_intra_disk_data}
    maskless_and_noiseless_ld_cbed_data = func_alias(**kwargs)

    ld_mask_data = _ld_mask_data(ld_mask_frame_data, cold_pixels)

    noiseless_ld_cbed_data = \
        1.0*torch.clip(ld_mask_data*maskless_and_noiseless_ld_cbed_data, min=0)

    ld_cbed_data = \
        (torch.poisson(noiseless_ld_cbed_data)
         if cbed_pattern_params._core_attrs["apply_shot_noise"]
         else noiseless_ld_cbed_data)

    ld_cbed_data = _normalize_image_data(image_data=ld_cbed_data)

    return ld_cbed_data



def _blurred_ld_disk_support_data(cbed_pattern_params,
                                  ld_misc_data,
                                  ld_multi_dim_slice_set):
    ld_disk_support_data = ld_misc_data[1]

    blurred_ld_disk_support_data = ld_disk_support_data
    key = "disk_support_gaussian_filter_std_dev"
    sigma = cbed_pattern_params._core_attrs[key]
    if sigma > 0:
        for disk_idx, mat in enumerate(blurred_ld_disk_support_data):
            ld_multi_dim_slice = ld_multi_dim_slice_set[disk_idx]
            ld_3d_slice = ld_multi_dim_slice
            ld_2d_slice = ld_multi_dim_slice[1:]

            func_alias = _apply_2d_guassian_filter
            kwargs = {"input_matrix": mat[ld_2d_slice],
                      "sigma": sigma,
                      "truncate": _truncate}
            blurred_ld_disk_support_data[ld_3d_slice] = func_alias(**kwargs)

    return blurred_ld_disk_support_data



def _apply_2d_guassian_filter(input_matrix, sigma, truncate):
    intermediate_tensor = input_matrix
    for axis_idx in range(2):
        kwargs = {"input_matrix": intermediate_tensor,
                  "sigma": sigma,
                  "truncate": truncate,
                  "axis_idx": axis_idx}
        intermediate_tensor = _apply_1d_guassian_filter(**kwargs)
    output_matrix = intermediate_tensor

    return output_matrix



def _apply_1d_guassian_filter(input_matrix, sigma, truncate, axis_idx):
    intermediate_tensor = torch.unsqueeze(input_matrix, dim=0)
    intermediate_tensor = torch.unsqueeze(intermediate_tensor, dim=0)

    radius = int(truncate * sigma + 0.5)
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

    return output_matrix



def _blurred_ld_intra_disk_data(cbed_pattern_params,
                                ld_misc_data,
                                ld_multi_dim_slice_set):
    _, _, ld_intra_disk_data, _ = ld_misc_data

    blurred_ld_intra_disk_data = ld_intra_disk_data
    key = "intra_disk_gaussian_filter_std_dev"
    sigma = cbed_pattern_params._core_attrs[key]
    if sigma > 0:
        for disk_idx, mat in enumerate(blurred_ld_intra_disk_data):
            ld_multi_dim_slice = ld_multi_dim_slice_set[disk_idx]
            ld_3d_slice = ld_multi_dim_slice
            ld_2d_slice = ld_multi_dim_slice[1:]

            func_alias = _apply_2d_guassian_filter
            kwargs = {"input_matrix": mat[ld_2d_slice],
                      "sigma": sigma,
                      "truncate": _truncate}
            blurred_ld_intra_disk_data[ld_3d_slice] = func_alias(**kwargs)

    return blurred_ld_intra_disk_data



def _maskless_and_noiseless_ld_cbed_data(ld_misc_data,
                                         ld_multi_dim_slice_set,
                                         blurred_ld_disk_support_data,
                                         blurred_ld_intra_disk_data):
    ld_bg_data, _, _, _ = ld_misc_data

    maskless_and_noiseless_ld_cbed_data = ld_bg_data
    for ld_multi_dim_slice in ld_multi_dim_slice_set:
        ld_3d_slice = ld_multi_dim_slice
        ld_2d_slice = ld_multi_dim_slice[1:]
            
        mat_1 = blurred_ld_disk_support_data[ld_3d_slice]
        mat_2 = blurred_ld_intra_disk_data[ld_3d_slice]
        maskless_and_noiseless_ld_cbed_data[ld_2d_slice] += mat_1*mat_2
    
    return maskless_and_noiseless_ld_cbed_data



def _ld_mask_data(ld_mask_frame_data, cold_pixels):
    coords_of_cold_pixels = cold_pixels
    ld_mask_data = ld_mask_frame_data + 0.0
    for coords_of_cold_pixel in coords_of_cold_pixels:
        ld_mask_data[coords_of_cold_pixel] = False

    return ld_mask_data



def _normalize_image_data(image_data):
    if image_data.max()-image_data.min() > 0:
        normalization_weight = 1 / (image_data.max()-image_data.min())
        normalization_bias = -normalization_weight*image_data.min()
        image_data = (image_data*normalization_weight
                      + normalization_bias).clip(min=0, max=1)

    return image_data



def _ld_overriding_masked_distorted_intensity_data(
        overriding_undistorted_intensity_data,
        hd_x_data,
        hd_y_data,
        cbed_pattern_params,
        cold_pixels,
        ld_misc_data):
    kwargs = {"datasubset_id": overriding_undistorted_intensity_data,
              "device": hd_x_data.device}
    input_tensor_to_sample = _load_nonnegative_matrix(**kwargs)
    input_tensor_to_sample = torch.unsqueeze(input_tensor_to_sample, dim=0)
    input_tensor_to_sample = torch.unsqueeze(input_tensor_to_sample, dim=0)

    grid_shape = (1,) + hd_x_data.shape + (2,)
    grid = torch.zeros(grid_shape,
                       dtype=hd_x_data.dtype,
                       device=hd_x_data.device)
    grid[0, :, :, 0] = 2*(hd_x_data-0.5)
    grid[0, :, :, 1] = -2*(hd_y_data-0.5)

    kwargs = \
        {"input": input_tensor_to_sample,
         "grid": grid,
         "mode": "bilinear",
         "padding_mode": "zeros",
         "align_corners": False}
    hd_overriding_distorted_intensity_data = \
        torch.nn.functional.grid_sample(**kwargs)[0, 0]

    key = "num_samples_across_each_pixel"
    num_samples_across_each_pixel = cbed_pattern_params._core_attrs[key]

    ld_mask_frame_data = ld_misc_data[3]
    ld_mask_data = _ld_mask_data(ld_mask_frame_data, cold_pixels)

    kwargs = \
        {"block_size": (num_samples_across_each_pixel,)*2, "func": np.min}
    ld_overriding_distorted_intensity_data = \
        skimage.measure.block_reduce(hd_overriding_distorted_intensity_data,
                                     **kwargs)
    ld_overriding_masked_distorted_intensity_data = \
        ld_mask_data*ld_overriding_distorted_intensity_data

    kwargs = \
        {"image_data": ld_overriding_masked_distorted_intensity_data}
    ld_overriding_masked_distorted_intensity_data = \
        _normalize_image_data(**kwargs)

    return ld_overriding_masked_distorted_intensity_data



def _disk_clipping_registry(ld_misc_data):
    _, ld_disk_support_data, _, ld_mask_frame_data = ld_misc_data

    M_4 = ld_disk_support_data.shape[1]

    rows_are_nonzero = torch.any(ld_mask_frame_data, dim=1)+0
    cols_are_nonzero = torch.any(ld_mask_frame_data, dim=0)+0

    if cols_are_nonzero.sum() > 0:
        L = cols_are_nonzero.argmax().item()
        R = torch.flip(cols_are_nonzero, dims=(0,)).argmax().item()
    else:
        L = M_4
        R = M_4

    if rows_are_nonzero.sum() > 0:
        B = torch.flip(rows_are_nonzero, dims=(0,)).argmax().item()
        T = rows_are_nonzero.argmax().item()
    else:
        B = M_4
        T = M_4

    image_mask_frame = np.array((L, R, B, T))
    
    num_disks = len(ld_disk_support_data)

    disk_clipping_registry = tuple()
    for disk_idx in range(num_disks):
        ld_support_data_of_disk_of_interest = ld_disk_support_data[disk_idx]
        disk_is_clipped = _disk_is_clipped(ld_support_data_of_disk_of_interest,
                                           image_mask_frame)
        disk_clipping_registry += (disk_is_clipped,)

    return disk_clipping_registry



def _disk_is_clipped(ld_support_data_of_disk_of_interest, image_mask_frame):
    ld_support_data_of_disk = ld_support_data_of_disk_of_interest
    M_4 = ld_support_data_of_disk.shape[0]
    
    disk_is_absent = _disk_is_absent(ld_support_data_of_disk_of_interest)

    if disk_is_absent or np.any(M_4-2 <= image_mask_frame):
        disk_is_clipped = True
    else:
        L, R, B, T = image_mask_frame

        disk_is_clipped = ((ld_support_data_of_disk[M_4-B-2:M_4-B, :].sum()
                            + ld_support_data_of_disk[T:T+2, :].sum()
                            + ld_support_data_of_disk[:, M_4-R-2:M_4-R].sum()
                            + ld_support_data_of_disk[:, L:L+2].sum()) > 0)
        disk_is_clipped = disk_is_clipped.item()

    return disk_is_clipped



def _disk_is_absent(ld_support_data_of_disk_of_interest):
    disk_is_absent = (ld_support_data_of_disk_of_interest.sum() == 0).item()

    return disk_is_absent



def _disk_absence_registry(ld_misc_data):
    _, ld_disk_support_data, _, _ = ld_misc_data
    
    disk_absence_registry = tuple()
    num_disks = len(ld_disk_support_data)
    for disk_idx in range(num_disks):
        ld_support_data_of_disk_of_interest = ld_disk_support_data[disk_idx]
        disk_is_absent = _disk_is_absent(ld_support_data_of_disk_of_interest)
        disk_absence_registry += (disk_is_absent,)

    return disk_absence_registry



def _update_signal_axes(signal):
    num_pixels_across_image = signal.axes_manager.signal_shape[0]

    sizes = (2,
             num_pixels_across_image,
             num_pixels_across_image)
    scales = (1,
              1/num_pixels_across_image,
              -1/num_pixels_across_image)
    offsets = (0,
               0.5/num_pixels_across_image,
               1-(1-0.5)/num_pixels_across_image)
    axes_labels = (r"CBED pattern (0) / support data (>0)",
                   r"fractional horizontal coordinate",
                   r"fractional vertical coordinate")
    units = ("dimensionless",)*3

    num_axes = len(units)

    for axis_idx in range(num_axes):
        axis = hyperspy.axes.UniformDataAxis(size=sizes[axis_idx],
                                             scale=scales[axis_idx],
                                             offset=offsets[axis_idx],
                                             units=units[axis_idx])
        signal.axes_manager[axis_idx].update_from(axis)
        signal.axes_manager[axis_idx].name = axis.name

    return None



def _num_samples_across_pattern(cbed_pattern_params):
    num_pixels_across_pattern = \
        cbed_pattern_params._core_attrs["num_pixels_across_pattern"]
    num_samples_across_each_pixel = \
        cbed_pattern_params._core_attrs["num_samples_across_each_pixel"]
    num_samples_across_pattern = \
        num_pixels_across_pattern * num_samples_across_each_pixel

    return num_samples_across_pattern



_default_cbed_pattern_params = None



class CBEDPattern(fancytypes.PreSerializableAndUpdatable):
    r"""Insert description here.

    Parameters
    ----------
    cbed_pattern_params : :class:`fakecbed.discretized.CBEDPatternParams` | `None`, optional
        Insert description here.

    Attributes
    ----------
    core_attrs : `dict`, read-only
        A `dict` representation of the core attributes: each `dict` key is a
        `str` representing the name of a core attribute, and the corresponding
        `dict` value is the object to which said core attribute is set. The core
        attributes are the same as the construction parameters, except that 
        their values might have been updated since construction.
    signal : :class:`hyperspy._signals.signal2d.Signal2D` | `None`, read-only
        Insert description here.
    num_disks : `int`, read-only
        Insert description here.
    disk_clipping_registry : `array_like` (`bool`, shape=(``num_disks``,)), optional
        Insert description here.
    disk_absence_registry : `array_like` (`bool`, shape=(``num_disks``,)), optional
        Insert description here.

    """
    _validation_and_conversion_funcs = \
        {"cbed_pattern_params": _check_and_convert_cbed_pattern_params}

    _pre_serialization_funcs = \
        {"cbed_pattern_params": _pre_serialize_cbed_pattern_params}

    _de_pre_serialization_funcs = \
        {"cbed_pattern_params": _de_pre_serialize_cbed_pattern_params}

    def __init__(self, cbed_pattern_params=_default_cbed_pattern_params):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        self._post_base_update()

        return None



    def _post_base_update(self):
        self._device = self._get_device()
        
        cbed_pattern_params = \
            self._core_attrs["cbed_pattern_params"]
        cold_pixels = \
            cbed_pattern_params._core_attrs["cold_pixels"]
        num_pixels_across_pattern = \
            cbed_pattern_params._core_attrs["num_pixels_across_pattern"]
        num_samples_across_pattern = \
            _num_samples_across_pattern(cbed_pattern_params)

        hd_i_range = torch.arange(num_samples_across_pattern,
                                  device=self._device)
        hd_j_range = torch.arange(num_samples_across_pattern,
                                  device=self._device)

        try:
            kwargs = {"cbed_pattern_params": cbed_pattern_params,
                      "i_range": hd_i_range,
                      "j_range": hd_j_range,
                      "resolution": "hd"}
            hd_x_data, hd_y_data = _x_and_y_data(**kwargs)

            func_alias = _ld_misc_data_and_multi_dim_slice_set
            kwargs = {"cbed_pattern_params": cbed_pattern_params,
                      "hd_i_range": hd_i_range,
                      "hd_j_range": hd_j_range,
                      "hd_x_data": hd_x_data,
                      "hd_y_data": hd_y_data}
            ld_misc_data, ld_multi_dim_slice_set = func_alias(**kwargs)
        except:
            raise RuntimeError(_cbed_pattern_err_msg_1)

        undistorted_disks = cbed_pattern_params._core_attrs["undistorted_disks"]
        self._num_disks = len(undistorted_disks)
        
        signal_data = _signal_data(cbed_pattern_params,
                                   ld_misc_data,
                                   ld_multi_dim_slice_set,
                                   cold_pixels)
        metadata = self._metadata(ld_misc_data)

        self._signal = hyperspy.signals.Signal2D(data=signal_data,
                                                 metadata=metadata)
        _update_signal_axes(signal=self._signal)

        return None



    def _get_device(self):
        cbed_pattern_params = self._core_attrs["cbed_pattern_params"]
        distortion_model = cbed_pattern_params._core_attrs["distortion_model"]
        device = distortion_model._device

        return device



    def _metadata(self, ld_misc_data):
        cbed_pattern_params = self._core_attrs["cbed_pattern_params"]
        distortion_model = cbed_pattern_params._core_attrs["distortion_model"]

        disk_clipping_registry = _disk_clipping_registry(ld_misc_data)
        self._disk_clipping_registry = disk_clipping_registry

        disk_absence_registry = _disk_absence_registry(ld_misc_data)
        self._disk_absence_registry = disk_absence_registry
        
        if distortion_model.is_trivial:
            title = "Fake Undistorted CBED Intensity Pattern"
        else:
            title = "Fake Distorted CBED Intensity Pattern"

        pre_serialize_cbed_pattern_params = cbed_pattern_params.pre_serialize()

        fakecbed_metadata = {"num_disks": \
                             self._num_disks,
                             "disk_clipping_registry": \
                             disk_clipping_registry,
                             "disk_absence_registry": \
                             disk_absence_registry,
                             "cbed_pattern_params": \
                             pre_serialize_cbed_pattern_params,
                             "intensity_image_has_been_overridden": \
                             False}
        
        metadata = {"General": {"title": title},
                    "Signal": {"pixel value units": "dimensionless"},
                    "FakeCBED": fakecbed_metadata}

        return metadata



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self._post_base_update()

        return None



    def override_maskless_intensity_image_data(self,
                                               maskless_intensity_image_data):
        func_alias = czekitout.convert.to_nonnegative_numpy_matrix
        kwargs = {"obj": maskless_intensity_image_data,
                  "obj_name": "maskless_intensity_image_data"}
        maskless_intensity_image_data = func_alias(**kwargs)

        cbed_pattern_params = \
            self._core_attrs["cbed_pattern_params"]
        num_pixels_across_pattern = \
            cbed_pattern_params._core_attrs["num_pixels_across_pattern"]

        expected_image_dims_in_pixels = 2*(num_pixels_across_pattern,)

        if maskless_intensity_image_data.shape != expected_image_dims_in_pixels:
            args = expected_image_dims_in_pixels
            unformatted_err_msg = _cbed_pattern_err_msg_2
            err_msg = unformatted_err_msg.format(*args)
            raise ValueError(err_msg)

        kwargs = {"maskless_intensity_image_data": \
                  maskless_intensity_image_data}
        self._override_maskless_intensity_image_data(**kwargs)

        return None



    def _override_maskless_intensity_image_data(self,
                                                maskless_intensity_image_data):
        self._device = self._get_device()

        cbed_pattern_params = \
            self._core_attrs["cbed_pattern_params"]
        num_pixels_across_pattern = \
            cbed_pattern_params._core_attrs["num_pixels_across_pattern"]
        cold_pixels = \
            cbed_pattern_params._core_attrs["cold_pixels"]

        ld_i_range = torch.arange(num_pixels_across_pattern,
                                  device=self._device)
        ld_j_range = torch.arange(num_pixels_across_pattern,
                                  device=self._device)

        kwargs = {"cbed_pattern_params": cbed_pattern_params,
                  "i_range": ld_i_range,
                  "j_range": ld_j_range,
                  "resolution": "ld"}
        ld_mask_frame_data = _mask_frame_data(**kwargs)

        kwargs = {"ld_mask_frame_data": ld_mask_frame_data,
                  "cold_pixels": cold_pixels}
        ld_mask_data = _ld_mask_data(**kwargs).cpu().detach().numpy()

        ld_cbed_data = \
            (1.0*ld_mask_data*maskless_intensity_image_data).clip(min=0)

        kwargs = {"image_data": ld_cbed_data}
        ld_cbed_data = _normalize_image_data(**kwargs)

        self._signal.data[0] = ld_cbed_data

        path_to_item = "FakeCBED.intensity_image_has_been_overridden"
        self._signal.metadata.set_item(path_to_item, True)

        return None



    @property
    def signal(self):
        return copy.deepcopy(self._signal)



    @property
    def num_disks(self):
        return self._num_disks


    
    @property
    def disk_clipping_registry(self):
        return copy.deepcopy(self._disk_clipping_registry)



    @property
    def disk_absence_registry(self):
        return copy.deepcopy(self._disk_absence_registry)



###########################
## Define error messages ##
###########################

_check_and_convert_undistorted_disks_err_msg_1 = \
    ("The object ``undistorted_disks`` must be a sequence of "
     "`fakecbed.shapes.NonUniformDisk` objects.")

_check_and_convert_undistorted_background_bands_err_msg_1 = \
    ("The object ``undistorted_background_bands`` must be a sequence of "
     "`fakecbed.shapes.Band` objects.")

_check_and_convert_cold_pixels_err_msg_1 = \
    ("The object ``cold_pixels`` must be a sequence of integer pairs, where "
     "each integer pair specifies valid pixel coordinates (i.e. row and column "
     "indices) of a pixel in the discretized CBED pattern.")

_cbed_pattern_err_msg_1 = \
    ("Failed to generate discretized CBED pattern. See traceback for details.")
_cbed_pattern_err_msg_2 = \
    ("The object ``maskless_intensity_image_data`` must have dimensions, in "
     "units of pixels, equal to those of the original CBED pattern intensity "
     "image being overridden, which in this case are ``({}, {})``.")
