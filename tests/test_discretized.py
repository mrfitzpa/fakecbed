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
r"""Contains tests for the module :mod:`fakecbed.discretized`.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For operations related to unit tests.
import pytest

# For general array handling.
import numpy as np

# For creating distortion models.
import distoptica



# For creating discretized fake CBED patterns.
import fakecbed



##################################
## Define classes and functions ##
##################################



def generate_cbed_pattern():
    kwargs = generate_cbed_pattern_ctor_params()
    cbed_pattern = fakecbed.discretized.CBEDPattern(**kwargs)

    return cbed_pattern



def generate_cbed_pattern_ctor_params():
    num_pixels_across_pattern = 128

    undistorted_tds_model = generate_undistorted_tds_model()
    
    undistorted_disks = generate_undistorted_disks()

    kwargs = {"undistorted_disks": undistorted_disks}
    undistorted_misc_shapes = generate_undistorted_misc_shapes(**kwargs)

    undistorted_outer_illumination_shape = \
        generate_undistorted_outer_illumination_shape(**kwargs)

    kwargs = {"num_pixels_across_pattern": num_pixels_across_pattern}
    distortion_model = generate_distortion_model(**kwargs)

    cold_pixels = generate_cold_pixels(num_pixels_across_pattern)

    cbed_pattern_ctor_params = {"undistorted_tds_model": \
                                undistorted_tds_model,
                                "undistorted_disks": \
                                undistorted_disks,
                                "undistorted_misc_shapes": \
                                undistorted_misc_shapes,
                                "undistorted_outer_illumination_shape": \
                                undistorted_outer_illumination_shape,
                                "gaussian_filter_std_dev": \
                                2,
                                "num_pixels_across_pattern": \
                                num_pixels_across_pattern,
                                "distortion_model": \
                                distortion_model,
                                "apply_shot_noise": \
                                True,
                                "rng_seed": \
                                5,
                                "detector_partition_width_in_pixels": \
                                4,
                                "cold_pixels": \
                                cold_pixels,
                                "mask_frame": \
                                (2, 5, 0, 7),
                                "skip_validation_and_conversion": \
                                False}

    return cbed_pattern_ctor_params



def generate_undistorted_tds_model():
    kwargs = {"center": (0.480, 0.490),
              "widths": (0.060, 0.050, 0.070, 0.055),
              "rotation_angle": np.pi/3,
              "val_at_center": 50,
              "functional_form": "asymmetric_gaussian"}
    undistorted_tds_peak_0 = fakecbed.shapes.Peak(**kwargs)

    kwargs = {"center": (0.50, 0.51),
              "widths": (0.075, 0.060, 0.045, 0.055),
              "rotation_angle": np.pi,
              "val_at_center": 55,
              "functional_form": "asymmetric_lorentzian"}
    undistorted_tds_peak_1 = fakecbed.shapes.Peak(**kwargs)

    kwargs = {"peaks": (undistorted_tds_peak_0, undistorted_tds_peak_1),
                        "constant_bg": 3}
    undistorted_tds_model = fakecbed.tds.Model(**kwargs)

    return undistorted_tds_model



def generate_undistorted_disks():
    undistorted_disks = tuple()

    num_undistorted_disks_to_be_defined = 5
    for disk_idx in range(num_undistorted_disks_to_be_defined):
        func_name = "generate_undistorted_disk_{}".format(disk_idx)
        func_alias = globals()[func_name]
        undistorted_disk = func_alias()
        undistorted_disks += (undistorted_disk,)

    return undistorted_disks



def generate_undistorted_disk_0():
    undistorted_disk_radius = generate_undistorted_disk_radius()
    
    kwargs = {"center": (0.500, 0.500),
              "radius": undistorted_disk_radius,
              "intra_shape_val": 1}
    undistorted_disk_0_support = fakecbed.shapes.Circle(**kwargs)

    kwargs = {"center": undistorted_disk_0_support.core_attrs["center"],
              "radius": undistorted_disk_radius,
              "intra_shape_val": 50}
    circle = fakecbed.shapes.Circle(**kwargs)

    kwargs = {"amplitude": 10,
              "wavelength": 1/40,
              "propagation_direction": 7*np.pi/8,
              "phase": 0}
    plane_wave = fakecbed.shapes.PlaneWave(**kwargs)

    kwargs = {"center": undistorted_disk_0_support.core_attrs["center"],
              "semi_major_axis": 1.0*undistorted_disk_radius,
              "eccentricity": 0.9,
              "rotation_angle": np.pi/4,
              "intra_shape_val": -50}
    ellipse = fakecbed.shapes.Ellipse(**kwargs)

    intra_support_shapes = (ellipse, circle, plane_wave)

    kwargs = {"support": undistorted_disk_0_support,
              "intra_support_shapes": intra_support_shapes}
    undistorted_disk_0 = fakecbed.shapes.NonuniformBoundedShape(**kwargs)

    return undistorted_disk_0



def generate_undistorted_disk_radius():
    undistorted_disk_radius = 1/20

    return undistorted_disk_radius



def generate_undistorted_disk_1():
    undistorted_disk_radius = generate_undistorted_disk_radius()
    
    kwargs = {"center": (0.300, 0.300),
              "radius": undistorted_disk_radius,
              "intra_shape_val": 1}
    undistorted_disk_1_support = fakecbed.shapes.Circle(**kwargs)

    kwargs = {"center": undistorted_disk_1_support.core_attrs["center"],
              "principal_quantum_number": 3,
              "azimuthal_quantum_number": 1,
              "magnetic_quantum_number": 0,
              "effective_size": undistorted_disk_radius/10,
              "renormalization_factor": 1e-2,
              "rotation_angle": 2*np.pi/3}
    orbital = fakecbed.shapes.Orbital(**kwargs)

    intra_support_shapes = (orbital,)

    kwargs = {"support": undistorted_disk_1_support,
              "intra_support_shapes": intra_support_shapes}
    undistorted_disk_1 = fakecbed.shapes.NonuniformBoundedShape(**kwargs)

    return undistorted_disk_1



def generate_undistorted_disk_2():
    undistorted_disk_radius = generate_undistorted_disk_radius()
    
    kwargs = {"center": (0.400, 0.980),
              "radius": undistorted_disk_radius,
              "intra_shape_val": 1}
    undistorted_disk_2_support = fakecbed.shapes.Circle(**kwargs)

    kwargs = {"center": undistorted_disk_2_support.core_attrs["center"],
              "radius": undistorted_disk_radius,
              "intra_shape_val": 5}
    bg_ellipse = fakecbed.shapes.Circle(**kwargs)

    ellipse_center = (undistorted_disk_2_support.core_attrs["center"][0]-0.01,
                      undistorted_disk_2_support.core_attrs["center"][1])
    kwargs = {"center": ellipse_center,
              "radius": undistorted_disk_radius,
              "intra_shape_val": 1}
    fg_ellipse = fakecbed.shapes.Circle(**kwargs)

    kwargs = {"fg_ellipse": fg_ellipse,
                        "bg_ellipse": bg_ellipse}
    lune = fakecbed.shapes.Lune(**kwargs)

    kwargs = {"center": undistorted_disk_2_support.core_attrs["center"],
              "radius": undistorted_disk_radius,
              "intra_shape_val": 2}
    circle = fakecbed.shapes.Circle(**kwargs)

    intra_support_shapes = (lune, circle)

    kwargs = {"support": undistorted_disk_2_support,
              "intra_support_shapes": intra_support_shapes}
    undistorted_disk_2 = fakecbed.shapes.NonuniformBoundedShape(**kwargs)

    return undistorted_disk_2



def generate_undistorted_disk_3():
    undistorted_disk_radius = generate_undistorted_disk_radius()
    
    kwargs = {"center": (0.400, 0.910),
              "radius": undistorted_disk_radius,
              "intra_shape_val": 1}
    undistorted_disk_3_support = fakecbed.shapes.Circle(**kwargs)

    kwargs = {"center": undistorted_disk_3_support.core_attrs["center"],
              "radius": undistorted_disk_radius,
              "intra_shape_val": 5}
    circle = fakecbed.shapes.Circle(**kwargs)

    intra_support_shapes = (circle,)

    kwargs = {"support": undistorted_disk_3_support,
              "intra_support_shapes": intra_support_shapes}
    undistorted_disk_3 = fakecbed.shapes.NonuniformBoundedShape(**kwargs)

    return undistorted_disk_3



def generate_undistorted_disk_4():
    undistorted_disk_radius = generate_undistorted_disk_radius()
    
    kwargs = {"center": (2, 2),
              "radius": undistorted_disk_radius,
              "intra_shape_val": 1}
    undistorted_disk_4_support = fakecbed.shapes.Circle(**kwargs)

    kwargs = {"center": undistorted_disk_4_support.core_attrs["center"],
              "radius": undistorted_disk_radius,
              "intra_shape_val": 5}
    circle = fakecbed.shapes.Circle(**kwargs)

    intra_support_shapes = (circle,)

    kwargs = {"support": undistorted_disk_4_support,
              "intra_support_shapes": intra_support_shapes}
    undistorted_disk_4 = fakecbed.shapes.NonuniformBoundedShape(**kwargs)

    return undistorted_disk_4



def generate_undistorted_misc_shapes(undistorted_disks):
    undistorted_disk_0 = undistorted_disks[0]
    undistorted_disk_0_support = undistorted_disk_0.core_attrs["support"]
    undistorted_disk_radius = undistorted_disk_0_support.core_attrs["radius"]

    kwargs = {"end_pt_1": (0.2, -0.05),
              "end_pt_2": (1.05, 0.60),
              "width": 0.03,
              "intra_shape_val": 2}
    undistorted_misc_shape_0 = fakecbed.shapes.Band(**kwargs)

    radial_range = (0.6*undistorted_disk_radius, 0.7*undistorted_disk_radius)

    kwargs = {"center": (0.2, 0.8),
              "midpoint_angle": 5*np.pi/4,
              "subtending_angle": np.pi/3,
              "radial_range": radial_range,
              "intra_shape_val": 8}
    undistorted_misc_shape_1 = fakecbed.shapes.Arc(**kwargs)

    undistorted_misc_shapes = (undistorted_misc_shape_0,
                               undistorted_misc_shape_1)

    return undistorted_misc_shapes



def generate_undistorted_outer_illumination_shape(undistorted_disks):
    undistorted_disk_0 = undistorted_disks[0]
    undistorted_disk_0_support = undistorted_disk_0.core_attrs["support"]
    radial_reference_pt = undistorted_disk_0_support.core_attrs["center"]

    kwargs = {"radial_reference_pt": radial_reference_pt,
              "radial_amplitudes": (0.55, 0.08, 0.07),
              "radial_phases": (0.00, 3*np.pi/5),
              "intra_shape_val": 1}
    undistorted_outer_illumination_shape = fakecbed.shapes.GenericBlob(**kwargs)

    return undistorted_outer_illumination_shape



def generate_distortion_model(num_pixels_across_pattern):
    standard_coord_transform_params = generate_standard_coord_transform_params()

    kwargs = {"max_num_iterations": 20,
              "initial_damping": 1e-3,
              "factor_for_decreasing_damping": 9,
              "factor_for_increasing_damping": 11,
              "improvement_tol": 0.1,
              "rel_err_tol": 1e-2,
              "plateau_tol": 1e-3,
              "plateau_patience": 2,
              "skip_validation_and_conversion": False}
    least_squares_alg_params = distoptica.LeastSquaresAlgParams(**kwargs)

    kwargs = \
        {"standard_coord_transform_params": standard_coord_transform_params,
         "sampling_grid_dims_in_pixels": 2*(num_pixels_across_pattern,),
         "device_name": "cpu",
         "least_squares_alg_params": least_squares_alg_params}
    distortion_model = \
        distoptica.generate_standard_distortion_model(**kwargs)

    return distortion_model



def generate_standard_coord_transform_params():
    center = (0.52, 0.49)

    quadratic_radial_distortion_amplitude = -0.1

    spiral_distortion_amplitude = 0.1

    amplitude = 0.07
    phase = 7*np.pi/8
    elliptical_distortion_vector = (amplitude*np.cos(2*phase).item(),
                                    amplitude*np.sin(2*phase).item())

    amplitude = 0.1
    phase = 4*np.pi/3
    parabolic_distortion_vector = (amplitude*np.cos(phase),
                                   amplitude*np.sin(phase))

    kwargs = \
        {"center": \
         center,
         "quadratic_radial_distortion_amplitude": \
         quadratic_radial_distortion_amplitude,
         "elliptical_distortion_vector": \
         elliptical_distortion_vector,
         "spiral_distortion_amplitude": \
         spiral_distortion_amplitude,
         "parabolic_distortion_vector": \
         parabolic_distortion_vector}
    standard_coord_transform_params = \
        distoptica.StandardCoordTransformParams(**kwargs)

    return standard_coord_transform_params



def generate_cold_pixels(num_pixels_across_pattern):
    cold_pixels = ((num_pixels_across_pattern//2, num_pixels_across_pattern//2),
                   (0, num_pixels_across_pattern//2),
                   (num_pixels_across_pattern//2, 0))

    return cold_pixels



def test_1_of_CBEDPattern():
    cbed_pattern = fakecbed.discretized.CBEDPattern()

    cbed_pattern.validation_and_conversion_funcs
    cbed_pattern.pre_serialization_funcs
    cbed_pattern.de_pre_serialization_funcs

    cbed_pattern.num_disks
    cbed_pattern.device

    kwargs = {"serializable_rep": cbed_pattern.pre_serialize()}
    fakecbed.discretized.CBEDPattern.de_pre_serialize(**kwargs)

    cbed_pattern_ctor_params = generate_cbed_pattern_ctor_params()

    new_core_attr_subset_candidate = cbed_pattern_ctor_params
    cbed_pattern.update(new_core_attr_subset_candidate)

    kwargs = {"serializable_rep": cbed_pattern.pre_serialize()}
    fakecbed.discretized.CBEDPattern.de_pre_serialize(**kwargs)

    kwargs = cbed_pattern_ctor_params.copy()
    kwargs_keys = tuple(kwargs.keys())

    key_1 = kwargs_keys[-1]
    for key_2 in kwargs_keys:
        if key_1 != "skip_validation_and_conversion":
            kwargs[key_1] = cbed_pattern_ctor_params[key_1]
        if key_2 != "skip_validation_and_conversion":
            kwargs[key_2] = slice(None)
            with pytest.raises(TypeError) as err_info:
                cbed_pattern = fakecbed.discretized.CBEDPattern(**kwargs)
        key_1 = key_2

    return None



def test_2_of_CBEDPattern():
    num_pixels_across_pattern = 128

    kwargs = {"sampling_grid_dims_in_pixels": (128, 130)}
    distortion_model = distoptica.DistortionModel(**kwargs)

    kwargs = {"num_pixels_across_pattern": num_pixels_across_pattern,
              "distortion_model": distortion_model}
    with pytest.raises(ValueError) as err_info:
        cbed_pattern = fakecbed.discretized.CBEDPattern(**kwargs)

    kwargs = {"sampling_grid_dims_in_pixels": (130, 128)}
    distortion_model = distoptica.DistortionModel(**kwargs)

    kwargs = {"num_pixels_across_pattern": num_pixels_across_pattern,
              "distortion_model": distortion_model}
    with pytest.raises(ValueError) as err_info:
        cbed_pattern = fakecbed.discretized.CBEDPattern(**kwargs)

    cold_pixel_sets = (((-num_pixels_across_pattern-1, 0),),
                       ((0, -num_pixels_across_pattern-1),),
                       ((num_pixels_across_pattern, 0),),
                       ((0, num_pixels_across_pattern),))
        
    for cold_pixel_set in cold_pixel_sets:
        kwargs = {"num_pixels_across_pattern": num_pixels_across_pattern,
                  "cold_pixels": cold_pixel_set}
        with pytest.raises(TypeError) as err_info:
            cbed_pattern = fakecbed.discretized.CBEDPattern(**kwargs)

    return None



def test_3_of_CBEDPattern():
    undistorted_misc_shapes = (fakecbed.shapes.Circle(),
                               fakecbed.shapes.Ellipse(),
                               fakecbed.shapes.Peak(),
                               fakecbed.shapes.PlaneWave(),
                               fakecbed.shapes.GenericBlob(),
                               fakecbed.shapes.Orbital(),
                               fakecbed.shapes.Lune(),
                               fakecbed.shapes.NonuniformBoundedShape())

    kwargs = {"undistorted_misc_shapes": undistorted_misc_shapes}
    cbed_pattern = fakecbed.discretized.CBEDPattern(**kwargs)

    kwargs = {"serializable_rep": cbed_pattern.pre_serialize()}
    fakecbed.discretized.CBEDPattern.de_pre_serialize(**kwargs)

    kwargs = {"undistorted_outer_illumination_shape": \
              fakecbed.shapes.Ellipse()}
    cbed_pattern = fakecbed.discretized.CBEDPattern(**kwargs)

    kwargs = {"serializable_rep": cbed_pattern.pre_serialize()}
    fakecbed.discretized.CBEDPattern.de_pre_serialize(**kwargs)

    undistorted_tds_model = generate_undistorted_tds_model()

    new_core_attr_subset_candidate = {"peaks": (fakecbed.shapes.Circle(),)}
    with pytest.raises(TypeError) as err_info:
        undistorted_tds_model.update(new_core_attr_subset_candidate)

    new_core_attr_subset_candidate = {"peaks": (fakecbed.shapes.Peak(),)}
    undistorted_tds_model.update(new_core_attr_subset_candidate)

    return None



def test_4_of_CBEDPattern():
    cbed_pattern = generate_cbed_pattern()

    assert (cbed_pattern.image_has_been_overridden == False)

    num_pixels_across_pattern = \
        cbed_pattern.core_attrs["num_pixels_across_pattern"]

    overriding_image = np.ones(2*(num_pixels_across_pattern,))
    
    kwargs = {"overriding_image": overriding_image,
              "skip_validation_and_conversion": False}
    cbed_pattern.override_image_then_reapply_mask(**kwargs)

    assert (cbed_pattern.image_has_been_overridden == True)

    overriding_image = 0*cbed_pattern.image
    cbed_pattern.signal

    kwargs = {"overriding_image": overriding_image,
              "skip_validation_and_conversion": True}
    cbed_pattern.override_image_then_reapply_mask(**kwargs)

    assert (cbed_pattern.image_has_been_overridden == True)

    overriding_image = np.ones(2*(num_pixels_across_pattern-1,))

    kwargs = {"overriding_image": overriding_image,
              "skip_validation_and_conversion": False}
    with pytest.raises(ValueError) as err_info:
        cbed_pattern.override_image_then_reapply_mask(**kwargs)

    new_core_attr_subset_candidate = {"cold_pixels": tuple()}
    cbed_pattern.update(new_core_attr_subset_candidate)

    assert (cbed_pattern.image_has_been_overridden == False)

    return None



def test_5_of_CBEDPattern():
    cbed_pattern = generate_cbed_pattern()

    cbed_pattern.image

    distortion_model = \
        cbed_pattern.core_attrs["distortion_model"]
    least_squares_alg_params = \
        distortion_model.core_attrs["least_squares_alg_params"]

    new_core_attr_subset_candidate = {"max_num_iterations": 1}
    least_squares_alg_params.update(new_core_attr_subset_candidate)

    new_core_attr_subset_candidate = {"least_squares_alg_params": \
                                      least_squares_alg_params}
    distortion_model.update(new_core_attr_subset_candidate)

    new_core_attr_subset_candidate = {"distortion_model": distortion_model}
    cbed_pattern.update(new_core_attr_subset_candidate)

    with pytest.raises(RuntimeError) as err_info:
        cbed_pattern.image
    
    return None



def test_6_of_CBEDPattern():
    attr_names = ("signal",
                  "image",
                  "illumination_support",
                  "disk_supports",
                  "disk_overlap_map",
                  "disk_clipping_registry",
                  "disk_absence_registry")

    for attr_name in attr_names:
        cbed_pattern = fakecbed.discretized.CBEDPattern()
        getattr(cbed_pattern, attr_name)

        method_name = "get_{}".format(attr_name)
        method_alias = getattr(cbed_pattern, method_name)
        method_alias(deep_copy=False)
    
    return None



def test_7_of_CBEDPattern():
    cbed_pattern = fakecbed.discretized.CBEDPattern()
    cbed_pattern.disk_overlap_map
    cbed_pattern.signal

    cbed_pattern = fakecbed.discretized.CBEDPattern()
    cbed_pattern.disk_clipping_registry
    cbed_pattern.disk_absence_registry
    cbed_pattern.signal

    cbed_pattern = generate_cbed_pattern()
    cbed_pattern.disk_clipping_registry

    cbed_pattern = generate_cbed_pattern()
    cbed_pattern.disk_supports
    cbed_pattern.disk_clipping_registry

    cbed_pattern = generate_cbed_pattern()
    cbed_pattern.disk_absence_registry

    cbed_pattern = fakecbed.discretized.CBEDPattern()
    cbed_pattern.disk_supports
    cbed_pattern.image

    cbed_pattern = generate_cbed_pattern()
    cbed_pattern.disk_overlap_map
    
    return None



def test_8_of_CBEDPattern():
    cbed_pattern_1 = generate_cbed_pattern()
    cbed_pattern_2 = generate_cbed_pattern()

    signal_1 = cbed_pattern_1.signal
    signal_2 = cbed_pattern_2.signal

    assert np.all(signal_1.data == signal_2.data)

    new_core_attr_subset_candidate = {"rng_seed": None}
    cbed_pattern_1.update(new_core_attr_subset_candidate)

    cbed_pattern_1.signal

    with pytest.raises(TypeError) as err_info:
        new_core_attr_subset_candidate = {"rng_seed": -1}
        cbed_pattern_1.update(new_core_attr_subset_candidate)

    new_core_attr_subset_candidate = {"apply_shot_noise": False}
    cbed_pattern_1.update(new_core_attr_subset_candidate)

    cbed_pattern_1.signal
    
    return None



def test_9_of_CBEDPattern():
    cbed_pattern = fakecbed.discretized.CBEDPattern()

    num_pixels_across_pattern = \
        cbed_pattern.core_attrs["num_pixels_across_pattern"]

    overriding_image = np.zeros(2*(num_pixels_across_pattern,))

    kwargs = {"overriding_image": overriding_image,
              "skip_validation_and_conversion": False}
    cbed_pattern.override_image_then_reapply_mask(**kwargs)

    cbed_pattern_signal = cbed_pattern.get_signal(deep_copy=False)
    
    return None



###########################
## Define error messages ##
###########################
