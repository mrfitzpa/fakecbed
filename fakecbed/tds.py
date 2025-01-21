"""For modelling thermal diffuse scattering.

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



# For validating, pre-serializing, and de-pre-serializing instances of the class
# :class:`fakecbed.shapes.Peak`. Also for defining subclasses of the
# :class:`fakecbed.shapes.BaseShape` class.
import fakecbed.shapes



##################################
## Define classes and functions ##
##################################

# List of public objects in module.
__all__ = ["Model"]



def _check_and_convert_peaks(params):
    obj_name = "peaks"
    obj = params[obj_name]

    accepted_types = (fakecbed.shapes.Peak,)

    current_func_name = "_check_and_convert_peaks"

    try:
        for peak in obj:
            kwargs = {"obj": peak,
                      "obj_name": "peak",
                      "accepted_types": accepted_types}
            czekitout.check.if_instance_of_any_accepted_types(**kwargs)
    except:
        err_msg = globals()[current_func_name+"_err_msg_1"]
        raise TypeError(err_msg)

    peaks = copy.deepcopy(obj)

    return peaks



def _pre_serialize_peaks(peaks):
    obj_to_pre_serialize = peaks
    serializable_rep = tuple()
    for elem in obj_to_pre_serialize:
        serializable_rep += (elem.pre_serialize(),)
    
    return serializable_rep



def _de_pre_serialize_peaks(serializable_rep):
    peaks = tuple()
    for pre_serialized_peak in serializable_rep:
        peak = fakecbed.shapes.Peak.de_pre_serialize(pre_serialized_peak)
        peaks += (peak,)

    return peaks



def _check_and_convert_constant_bg(params):
    obj_name = "constant_bg"
    kwargs = {"obj": params[obj_name], "obj_name": obj_name}
    constant_bg = czekitout.convert.to_nonnegative_float(**kwargs)

    return constant_bg



def _pre_serialize_constant_bg(constant_bg):
    obj_to_pre_serialize = constant_bg
    serializable_rep = obj_to_pre_serialize
    
    return serializable_rep



def _de_pre_serialize_constant_bg(serializable_rep):
    constant_bg = serializable_rep

    return constant_bg



_default_peaks = \
    tuple()
_default_constant_bg = \
    0
_default_skip_validation_and_conversion = \
    fakecbed.shapes._default_skip_validation_and_conversion



class Model(fakecbed.shapes.BaseShape):
    r"""The intensity pattern of a thermal diffuse scattering (TDS) model.

    Let :math:`N_{\text{TDS}}`, and :math:`\left\{
    \mathcal{I}_{k;\text{TDS}}\left(u_{x},u_{y}\right)\right\}
    _{k=0}^{N_{\text{TDS}}-1}` be the number of peaks in the intensity pattern
    of the TDS model, and the intensity patterns of said peaks
    respectively. Furthermore, let :math:`C_{\text{TDS}}` be the value of the
    intensity pattern of the TDS model at any point that is an arbitrarily large
    distance away from the origin of the image coordinate axes. The undistorted
    intensity pattern of the TDS model is given by:

    .. math ::
        \mathcal{I}_{\text{TDS}}\left(u_{x},u_{y}\right)=
        C_{\text{TDS}}+\left|\sum_{k=0}^{N_{\text{TDS}}-1}
        \mathcal{I}_{k;\text{TDS}}\left(u_{x},u_{y}\right)\right|,
        :label: intensity_pattern_of_tds_model__1

    where :math:`u_{x}` and :math:`u_{y}` are fractional horizontal and vertical
    coordinates of the undistorted intensity pattern of the TDS model
    respectively.

    Parameters
    ----------
    peaks : `array_like` (`fakecbed.shapes.Peak`, ndim=1), optional
        The intensity patterns of the peaks, :math:`\left\{
        \mathcal{I}_{k;\text{TDS}}\left(u_{x},u_{y}\right)\right\}
        _{k=0}^{N_{\text{TDS}}-1}`, inside the intensity pattern of the TDS 
        model.
    constant_bg : `float`, optional
        The value of the intensity pattern of the TDS model, 
        :math:`C_{\text{TDS}}`, at any point that is an arbitrarily large 
        distance away from the origin of the image coordinate axes. Must be 
        nonnegative.
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
    ctor_param_names = ("peaks",
                        "constant_bg")
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
                 peaks=\
                 _default_peaks,
                 constant_bg=\
                 _default_constant_bg,
                 skip_validation_and_conversion=\
                 _default_skip_validation_and_conversion):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        fakecbed.shapes.BaseShape.__init__(self, ctor_params)

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
        peaks = self._peaks
        C = self._constant_bg
        
        result = torch.zeros_like(u_x)
        for peak in peaks:
            result += peak._eval(u_x, u_y)
        result = C+torch.abs(result)

        return result



###########################
## Define error messages ##
###########################

_check_and_convert_peaks_err_msg_1 = \
    ("The object ``peaks`` must be a sequence of `fakecbed.shapes.Peak` "
     "objects.")
