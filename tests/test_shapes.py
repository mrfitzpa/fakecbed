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
r"""Contains tests for the module :mod:`fakecbed.shapes`.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For operations related to unit tests.
import pytest

# For general array handling.
import numpy as np
import torch



# For creating discretized fake CBED patterns.
import fakecbed.shapes



##################################
## Define classes and functions ##
##################################



def test_1_of_BaseShape():
    with pytest.raises(NotImplementedError) as err_info:
        base_shape = fakecbed.shapes.BaseShape(ctor_params=dict())

    return None



def test_1_of_Circle():
    circle = fakecbed.shapes.Circle()

    circle.validation_and_conversion_funcs
    circle.pre_serialization_funcs
    circle.de_pre_serialization_funcs

    with pytest.raises(ValueError) as err_info:
        kwargs = {"u_x": torch.rand(3, 4), "u_y": torch.rand(4, 3)}
        circle.eval(**kwargs)

    with pytest.raises(TypeError) as err_info:
        kwargs["u_x"] = torch.rand(3, 4, 5)
        circle.eval(**kwargs)

    kwargs = {"u_x": torch.rand(3, 4),
              "u_y": torch.rand(3, 4),
              "device": torch.device("cpu"),
              "skip_validation_and_conversion": False}
    circle.eval(**kwargs)

    kwargs["skip_validation_and_conversion"] = True
    circle.eval(**kwargs)

    new_core_attr_subset_candidate = {"intra_shape_val": 0}
    circle.update(new_core_attr_subset_candidate)

    return None



def test_1_of_Ellipse():
    ellipse = fakecbed.shapes.Ellipse()

    new_core_attr_subset_candidate = {"eccentricity": 0.5}
    ellipse.update(new_core_attr_subset_candidate)

    with pytest.raises(ValueError) as err_info:
        new_core_attr_subset_candidate = {"eccentricity": 2}
        ellipse.update(new_core_attr_subset_candidate)

    return None



def test_1_of_Peak():
    kwargs = {"functional_form": "asymmetric_exponential"}
    peak = fakecbed.shapes.Peak(**kwargs)

    kwargs = {"u_x": torch.rand(3, 4), "u_y": torch.rand(3, 4)}
    peak.eval(**kwargs)

    new_core_attr_subset_candidate = {"functional_form": "asymmetric_gaussian"}
    peak.update(new_core_attr_subset_candidate)

    return None



def test_1_of_Band():
    band = fakecbed.shapes.Band()

    new_core_attr_subset_candidate = {"intra_shape_val": 0}
    band.update(new_core_attr_subset_candidate)

    return None



def test_1_of_PlaneWave():
    plane_wave = fakecbed.shapes.PlaneWave()

    new_core_attr_subset_candidate = {"amplitude": 0}
    plane_wave.update(new_core_attr_subset_candidate)

    return None



def test_1_of_Arc():
    arc = fakecbed.shapes.Arc()

    new_core_attr_subset_candidate = {"radial_range": (0.1, 0.2)}
    arc.update(new_core_attr_subset_candidate)

    with pytest.raises(ValueError) as err_info:
        new_core_attr_subset_candidate = {"radial_range": (0.2, 0.2)}
        arc.update(new_core_attr_subset_candidate)

    return None



def test_1_of_GenericBlob():
    generic_blob = fakecbed.shapes.GenericBlob()

    new_core_attr_subset_candidate = {"radial_amplitudes": (2, 0.5, 1.0),
                                      "radial_phases": (0, np.pi)}
    generic_blob.update(new_core_attr_subset_candidate)

    with pytest.raises(ValueError) as err_info:
        new_core_attr_subset_candidate = {"radial_amplitudes": tuple()}
        generic_blob.update(new_core_attr_subset_candidate)

    with pytest.raises(ValueError) as err_info:
        new_core_attr_subset_candidate = {"radial_amplitudes": (2, 1, 1.5)}
        generic_blob.update(new_core_attr_subset_candidate)

    with pytest.raises(ValueError) as err_info:
        new_core_attr_subset_candidate = {"radial_amplitudes": (2, 0.5),
                                          "radial_phases": (0, np.pi)}
        generic_blob.update(new_core_attr_subset_candidate)

    return None



def test_1_of_Orbital():
    orbital = fakecbed.shapes.Orbital()

    new_core_attr_subset_candidate = {"principal_quantum_number": 5,
                                      "azimuthal_quantum_number": 3,
                                      "magnetic_quantum_number": 1}
    orbital.update(new_core_attr_subset_candidate)

    kwargs = {"u_x": torch.rand(3, 4), "u_y": torch.rand(3, 4)}
    orbital.eval(**kwargs)

    with pytest.raises(ValueError) as err_info:
        new_core_attr_subset_candidate = {"azimuthal_quantum_number": 5}
        orbital.update(new_core_attr_subset_candidate)

    with pytest.raises(ValueError) as err_info:
        new_core_attr_subset_candidate = {"magnetic_quantum_number": 4}
        orbital.update(new_core_attr_subset_candidate)

    new_core_attr_subset_candidate = {"azimuthal_quantum_number": 1}
    orbital.update(new_core_attr_subset_candidate)

    kwargs = {"u_x": torch.rand(3, 4), "u_y": torch.rand(3, 4)}
    orbital.eval(**kwargs)

    return None



def test_1_of_Lune():
    lune = fakecbed.shapes.Lune()

    new_core_attr_subset_candidate = {"bg_ellipse": fakecbed.shapes.Circle()}
    lune.update(new_core_attr_subset_candidate)

    return None



def test_1_of_NonuniformBoundedShape():
    nonuniform_bounded_shape = fakecbed.shapes.NonuniformBoundedShape()

    supports = (fakecbed.shapes.Ellipse(),
                fakecbed.shapes.Band(),
                fakecbed.shapes.Arc(),
                fakecbed.shapes.GenericBlob(),
                fakecbed.shapes.Lune())

    for support in supports:
        intra_support_shapes = (fakecbed.shapes.Peak(),
                                fakecbed.shapes.Band(),
                                fakecbed.shapes.Arc(),
                                fakecbed.shapes.GenericBlob(),
                                fakecbed.shapes.NonuniformBoundedShape())

        new_core_attr_subset_candidate = {"support": \
                                          support,
                                          "intra_support_shapes": \
                                          intra_support_shapes}
        nonuniform_bounded_shape.update(new_core_attr_subset_candidate)

        serializable_rep = nonuniform_bounded_shape.pre_serialize()
    
        method_alias = fakecbed.shapes.NonuniformBoundedShape.de_pre_serialize
        method_alias(serializable_rep)

    with pytest.raises(TypeError) as err_info:
        new_core_attr_subset_candidate = {"intra_support_shapes": 0}
        nonuniform_bounded_shape.update(new_core_attr_subset_candidate)

    kwargs = {"u_x": torch.rand(3, 4), "u_y": torch.rand(3, 4)}
    nonuniform_bounded_shape.eval(**kwargs)

    return None



###########################
## Define error messages ##
###########################
