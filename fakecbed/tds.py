"""Insert a one-line description of module.

A more detailed description of the module. The more detailed description can
span multiple lines.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For performing deep copies.
import copy



# For validating and converting objects.
import czekitout.check
import czekitout.convert

# For defining classes that support enforced validation, updatability,
# pre-serialization, and de-serialization.
import fancytypes



# For validating, pre-serializing, and de-pre-serializing instances of the class
# :class:`fakecbed.shapes.Peak`.
import fakecbed.shapes



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
__all__ = ["Model"]



def _check_and_convert_cartesian_coords(params):
    cartesian_coords = \
        fakecbed.shapes._check_and_convert_cartesian_coords(params)

    return cartesian_coords



_default_x = fakecbed.shapes._default_x
_default_y = fakecbed.shapes._default_y



def _check_and_convert_device(params):
    device = fakecbed.shapes._check_and_convert_device(params)

    return device



_default_device = fakecbed.shapes._default_device



def _check_and_convert_peak(params):
    obj_name = "peak"
    obj = copy.deepcopy(params[obj_name])
    
    if obj is None:
        kwargs = {"center": (0.5, 0.5), "width": 1, "val_at_center": 0}
        peak = fakecbed.shapes.Peak(**kwargs)
    else:   
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": (fakecbed.shapes.Peak, type(None))}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        peak = obj

        if peak._core_attrs["val_at_center"] < 0:
            raise ValueError(_check_and_convert_peak_err_msg_1)

    return peak



def _pre_serialize_peak(peak):
    serializable_rep = peak.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_peak(serializable_rep):
    peak = fakecbed.shapes.Peak.de_pre_serialize(serializable_rep)

    return peak



def _check_and_convert_constant_background(params):
    obj_name = "constant_background"
    obj = params[obj_name]
    constant_background = czekitout.convert.to_nonnegative_float(obj, obj_name)

    return constant_background



def _pre_serialize_constant_background(constant_background):
    serializable_rep = constant_background
    
    return serializable_rep



def _de_pre_serialize_constant_background(serializable_rep):
    constant_background = serializable_rep

    return constant_background



_default_peak = None
_default_constant_background = 0



class Model(fancytypes.PreSerializableAndUpdatable):
    r"""Insert description here.

    Parameters
    ----------
    peak : :class:`fakecbed.shapes.Peak` | `None`, optional
        Insert description here.
    constant_background : `float`, optional
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
        {"peak": _check_and_convert_peak,
         "constant_background": _check_and_convert_constant_background}

    _pre_serialization_funcs = \
        {"peak": _pre_serialize_peak,
         "constant_background": _pre_serialize_constant_background}

    _de_pre_serialization_funcs = \
        {"peak": _de_pre_serialize_peak,
         "constant_background": _de_pre_serialize_constant_background}

    def __init__(self,
                 peak=_default_peak,
                 constant_background=_default_constant_background):
        ctor_params = {key: val
                       for key, val in locals().items()
                       if (key not in ("self", "__class__"))}
        fancytypes.PreSerializableAndUpdatable.__init__(self, ctor_params)

        self._post_base_update()

        return None



    def _post_base_update(self):
        self._peak = self._core_attrs["peak"]
        self._constant_background = self._core_attrs["constant_background"]

        return None



    def update(self, core_attr_subset):
        super().update(core_attr_subset)
        self._post_base_update()

        return None



    def eval(self, x=_default_x, y=_default_y, device=_default_device):
        params = {"cartesian_coords": (x, y), "device": device}
        device = _check_and_convert_device(params)
        x, y = _check_and_convert_cartesian_coords(params)

        result = self._eval(x, y)

        return result



    def _eval(self, x, y):
        result = self._peak.eval(x, y) + self._constant_background

        return result



def _check_and_convert_undistorted_tds_model(params):
    obj_name = "undistorted_tds_model"
    obj = copy.deepcopy(params[obj_name])
    
    if obj is None:
        undistorted_tds_model = Model()
    else:
        accepted_types = (Model, type(None))
        kwargs = {"obj": obj,
                  "obj_name": obj_name,
                  "accepted_types": accepted_types}
        czekitout.check.if_instance_of_any_accepted_types(**kwargs)
        undistorted_tds_model = obj

    return undistorted_tds_model



def _pre_serialize_undistorted_tds_model(undistorted_tds_model):
    serializable_rep = undistorted_tds_model.pre_serialize()
    
    return serializable_rep



def _de_pre_serialize_undistorted_tds_model(serializable_rep):
    undistorted_tds_model = Model.de_pre_serialize(serializable_rep)

    return undistorted_tds_model



_default_undistorted_tds_model = None



###########################
## Define error messages ##
###########################

_check_and_convert_peak_err_msg_1 = \
    ("The object ``peak`` must specify either a positive-valued peak, or "
     "trivially a zero-valued peak.")
