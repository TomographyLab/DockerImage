# -*- coding: utf-8 -*-
# tomolab
# Michele Scipioni
# Harvard University, Martinos Center for Biomedical Imaging
# University of Pisa



# If you are looking for PET reconstruction, this is where to start.
# The objects defined here provide abstractions for Static and Dynamic PET reconstruction,
# abstracting the scanner geometries and vendor models and providing an interface to the
# software tools for projection, backprojection and reconstruction.
# Occiput enables reconstruction both from list-mode and from sinogram data using GPU
# acceleration.


import copy
import time

# Import interfile data handling module

# Import occiput:
from tomolab.Core import Image3D
from tomolab.Transformation.Transformations import RigidTransform, Transform_Scale
from tomolab.Core.Errors import  UnexpectedParameter
from tomolab.Reconstruction.PET.PET_projection import PET_Projection, PET_Projection_Sparsity
from tomolab.Reconstruction.PET.PET_raytracer import PET_project_compressed, PET_backproject_compressed
from tomolab.Reconstruction.PET.PET_subsets import SubsetGenerator
from tomolab.Visualization.Visualization import ProgressBar
from tomolab.Visualization import Colors as C

# Set verbose level
# This is a global setting for occiput. There are 3 levels of verbose:
# high, low, no_printing
from tomolab.global_settings import *
set_verbose_no_printing()

import numpy as np
from scipy import ndimage

try:
    import pylab
except BaseException:
    has_pylab = False
else:
    has_pylab = True

set_verbose_no_printing()

# Default parameters
DEFAULT_SUBSET_SIZE = 24
DEFAULT_RECON_ITERATIONS = 10
DEFAULT_N_TIME_BINS = 15
EPS = 1e-6


def f_continuous(var):
    """Makes an nd_array Fortran-contiguous. """
    if isinstance(var, np.ndarray):
        if not var.flags.f_contiguous:
            var = np.asarray(var, order="F")
    else:
        if hasattr(var, "data"):
            if isinstance(var.data, np.ndarray):
                if not var.data.flags.f_contiguous:
                    var.data = np.asarray(var.data, order="F")
    return var

'''
def make_Image3D_activity(self, data=None):
        shape = np.float32(self.activity_shape)
        size = np.float32(self.activity_size)
        T_scanner_to_world = self.transform_scanner_to_world
        T_pix_to_scanner = Transform_Scale(
            size / shape, map_from="pixels_PET_Static", map_to="scanner"
        )
        T_pix_to_world = T_scanner_to_world.left_multiply(T_pix_to_scanner)
        image = Image3D(data=data, affine=T_pix_to_world, space="world")
        return image

def make_Image3D_attenuation(self, data=None):
        shape = np.float32(self.attenuation_shape)
        size = np.float32(self.attenuation_size)
        T_scanner_to_world = self.transform_scanner_to_world
        T_pix_to_scanner = Transform_Scale(
            size / shape, map_from="pixels_PET_Static", map_to="scanner"
        )
        T_pix_to_world = T_scanner_to_world.left_multiply(T_pix_to_scanner)
        image = Image3D(data=data, affine=T_pix_to_world, space="world")
        return image


def get_sparsity(self):
        # if self.prompts == None:
        # in this case returns sparsity pattern for uncompressed projection
        sparsity = PET_Projection_Sparsity(
            self.binning.N_axial,
            self.binning.N_azimuthal,
            self.binning.N_u,
            self.binning.N_v,
        )
        # else:
        #    sparsity = self.prompts.sparsity
        return sparsity
'''

def get_sparsity(binning, prompts):
    if prompts == None:
        # in this case returns sparsity pattern for uncompressed projection
        sparsity = PET_Projection_Sparsity(
            binning.N_axial,
            binning.N_azimuthal,
            binning.N_u,
            binning.N_v,
        )
    else:
            sparsity = prompts.sparsity
    return sparsity

def make_Image3D_activity( shape, size, T_scanner_to_world, data=None):
        #shape = np.float32(self.activity_shape)
        #size = np.float32(self.activity_size)
        #T_scanner_to_world = self.transform_scanner_to_world
        T_pix_to_scanner = Transform_Scale(
            size / shape, map_from="pixels_PET_Static", map_to="scanner"
        )
        T_pix_to_world = T_scanner_to_world.left_multiply(T_pix_to_scanner)
        image = Image3D(data=data, affine=T_pix_to_world, space="world")
        return image

def make_Image3D_attenuation(shape, size, T_scanner_to_world, data=None):
        #shape = np.float32(self.attenuation_shape)
        #size = np.float32(self.attenuation_size)
        #T_scanner_to_world = self.transform_scanner_to_world
        T_pix_to_scanner = Transform_Scale(
            size / shape, map_from="pixels_PET_Static", map_to="scanner"
        )
        T_pix_to_world = T_scanner_to_world.left_multiply(T_pix_to_scanner)
        image = Image3D(data=data, affine=T_pix_to_world, space="world")
        return image

####################################################################################################################

def project_attenuation(
        self,
        attenuation=None,
        unit="inv_cm",
        transformation=None,
        sparsity=None,
        subsets_matrix=None,
        exponentiate=True,
    ):

    self.profiler.tic()

    if attenuation is None:
        attenuation = self.attenuation
    if isinstance(attenuation, np.ndarray):
        attenuation_data = f_continuous(np.float32(attenuation))
    else:
        attenuation_data = f_continuous(np.float32(attenuation.data))
    self.profiler.rec_project_make_continuous()

    if not list(attenuation_data.shape) == list(self.attenuation_shape):
        raise UnexpectedParameter(
            "Attenuation must have the same shape as self.attenuation_shape"
        )

    # By default, the center of the imaging volume is at the center of the scanner
    tx = 0.5 * (
        self.attenuation_size[0]
        - self.attenuation_size[0] / self.attenuation_shape[0]
    )
    ty = 0.5 * (
        self.attenuation_size[1]
        - self.attenuation_size[1] / self.attenuation_shape[1]
    )
    tz = 0.5 * (
        self.attenuation_size[2]
        - self.attenuation_size[2] / self.attenuation_shape[2]
    )
    if transformation is None:
        transformation = RigidTransform((tx, ty, tz, 0, 0, 0))
    else:
        transformation = copy.copy(transformation)
        transformation.x = transformation.x + tx
        transformation.y = transformation.y + ty
        transformation.z = transformation.z + tz

    # Scale according to the unit measure of the specified attenuation. It is assumed that the attenuation map
    # is constant in a voxel, with the value specified in 'attenuation', of unit measure 'unit'.
    if unit == "inv_mm":
        invert = False
        scale = 1.0
    elif unit == "inv_cm":
        invert = False
        scale = 10.0
    elif unit == "mm":
        invert = True
        scale = 1.0
    elif unit == "cm":
        invert = True
        scale = 10.0
    else:
        print(
            "Unit measure unknown. Assuming inv_cm. Keep track of the unit measures! "
        )
        invert = False
        scale = 10.0

    if invert:
        attenuation_data = 1.0 / (attenuation_data + EPS)
    step_size_mm = self.attenuation_projection_parameters.sample_step
    step_size = step_size_mm / scale

    # Optionally project with a sparsity pattern not equal to sparsity associated to the loaded prompts data
    # Note: if prompts have not been loaded, self.get_sparsity() assumes no compression.

    if sparsity is None:
        sparsity = get_sparsity(self.binning, self.prompts)

    # Optionally project only to a subset of projection planes
    if subsets_matrix is None:
        sparsity_subset = sparsity
        self.profiler.tic()
        angles = self.binning.get_angles()
        self.profiler.rec_project_get_angles()
    else:
        self.profiler.tic()
        sparsity_subset = sparsity.get_subset(subsets_matrix)
        self.profiler.rec_project_get_subset_sparsity()
        self.profiler.tic()
        angles = self.binning.get_angles(subsets_matrix)
        self.profiler.rec_project_get_angles()

    offsets = sparsity_subset.offsets
    locations = sparsity_subset.locations
    activations = np.ones([angles.shape[1], angles.shape[2]], dtype="uint32")

    # Call the raytracer
    self.profiler.tic()
    projection_data, timing = PET_project_compressed(
        attenuation_data,
        None,
        offsets,
        locations,
        activations,
        angles.shape[2],
        angles.shape[1],
        angles,
        self.binning.N_u,
        self.binning.N_v,
        self.binning.size_u,
        self.binning.size_v,
        self.attenuation_size[0],
        self.attenuation_size[1],
        self.attenuation_size[2],
        0.0,
        0.0,
        0.0,
        transformation.x,
        transformation.y,
        transformation.z,
        transformation.theta_x,
        transformation.theta_y,
        transformation.theta_z,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        self.attenuation_projection_parameters.gpu_acceleration,
        self.attenuation_projection_parameters.N_samples,
        self.attenuation_projection_parameters.sample_step,
        self.attenuation_projection_parameters.background_attenuation,
        0.0,
        self.attenuation_projection_parameters.truncate_negative_values,
        self.attenuation_projection_parameters.direction,
        self.attenuation_projection_parameters.block_size,
    )

    self.profiler.rec_project_projection()
    self.profiler.rec_projection(timing)

    # Fix scale and exponentiate
    if exponentiate:
        self.profiler.tic()
        projection_data = np.exp(-projection_data * step_size)
        self.profiler.rec_project_exponentiate()
    else:
        self.profiler.tic()
        projection_data = projection_data * step_size
        self.profiler.rec_project_scale()

    # Create object PET_Projection: it contains the raw projection data and the description of the projection geometry
    # and sparsity pattern.
    time_bins = np.int32([0, 0])

    # Projection of the attenuation does not have timing information
    self.profiler.tic()
    projection = PET_Projection(
            self.binning,
            projection_data,
            sparsity.offsets,
            sparsity.locations,
            time_bins,
            subsets_matrix,
    )
    self.profiler.rec_project_wrap()
    self.set_attenuation_projection(projection)
    return projection

def backproject_attenuation(
        self,
        projection,
        unit="inv_cm",
        transformation=None,
        sparsity=None,
        subsets_matrix=None,
    ):
        if isinstance(projection, np.ndarray):
            projection_data = np.float32(projection)
        else:
            projection_data = np.float32(projection.data)

        # By default, the center of the imaging volume is at the center of the scanner
        tx = 0.5 * (
            self.attenuation_size[0]
            - self.attenuation_size[0] / self.attenuation_shape[0]
        )
        ty = 0.5 * (
            self.attenuation_size[1]
            - self.attenuation_size[1] / self.attenuation_shape[1]
        )
        tz = 0.5 * (
            self.attenuation_size[2]
            - self.attenuation_size[2] / self.attenuation_shape[2]
        )
        if transformation is None:
            transformation = RigidTransform((tx, ty, tz, 0, 0, 0))
        else:
            transformation = copy.copy(transformation)
            transformation.x = transformation.x + tx
            transformation.y = transformation.y + ty
            transformation.z = transformation.z + tz

        # Scale according to the unit measure of the specified attenuation. It is assumed that the attenuation map
        # is constant in a voxel, with the value specified in 'attenuation', of unit measure 'unit'.
        if unit == "inv_mm":
            invert = False
            scale = 1.0
        elif unit == "inv_cm":
            invert = False
            scale = 10.0
        elif unit == "mm":
            invert = True
            scale = 1.0
        elif unit == "cm":
            invert = True
            scale = 10.0
        else:
            print(
                "Unit measure unknown. Assuming inv_cm. Keep track of the unit measures! "
            )
            invert = False
            scale = 10.0

        if invert:
            attenuation_data = 1.0 / (np.attenuation_data + np.eps)
        step_size_mm = self.attenuation_projection_parameters.sample_step
        step_size = step_size_mm / scale

        if sparsity is None:
            sparsity = get_sparsity(self.binning, self.prompts)

        if isinstance(projection, np.ndarray):
            projection_data = np.float32(projection)
            offsets = sparsity.offsets
            locations = sparsity.locations
            angles = self.binning.get_angles(subsets_matrix)
            activations = np.ones(
                [self.binning.N_azimuthal, self.binning.N_axial], dtype="uint32"
            )
        else:
            projection_data = np.float32(projection.data)
            offsets = projection.sparsity.offsets
            locations = projection.sparsity.locations
            angles = projection.get_angles()
            activations = np.ones(
                [projection.sparsity.N_azimuthal, projection.sparsity.N_axial],
                dtype="uint32",
            )

        # print "backproject attenuation"
        # print "projection_data",projection_data.shape
        # print "offsets",offsets.shape
        # print "locations",locations.shape
        # print "activations",activations.shape
        # print "angles",angles.shape

        # Call ray-tracer
        backprojection_data, timing = PET_backproject_compressed(
            projection_data,
            None,
            offsets,
            locations,
            activations,
            angles.shape[2],
            angles.shape[1],
            angles,
            self.binning.N_u,
            self.binning.N_v,
            self.binning.size_u,
            self.binning.size_v,
            self.attenuation_shape[0],
            self.attenuation_shape[1],
            self.attenuation_shape[2],
            self.attenuation_size[0],
            self.attenuation_size[1],
            self.attenuation_size[2],
            0.0,
            0.0,
            0.0,
            transformation.x,
            transformation.y,
            transformation.z,
            transformation.theta_x,
            transformation.theta_y,
            transformation.theta_z,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            self.attenuation_backprojection_parameters.gpu_acceleration,
            self.attenuation_backprojection_parameters.N_samples,
            self.attenuation_backprojection_parameters.sample_step,
            self.attenuation_backprojection_parameters.background_attenuation,
            0.0,
            self.attenuation_backprojection_parameters.direction,
            self.attenuation_backprojection_parameters.block_size,
        )

        self.profiler.rec_backprojection(timing)
        backprojection_data = backprojection_data * step_size

        # Set the correct scale - unit measure and return Image3D - FIXME: set scale for requested unit measure
        return make_Image3D_attenuation(shape=np.float32(self.attenuation_shape),
                                         size=np.float32(self.attenuation_size),
                                         T_scanner_to_world=self.transform_scanner_to_world,
                                         data=backprojection_data)


def project_activity(
        self,
        activity,
        unit="Bq/mm3",
        transformation=None,
        sparsity=None,
        subsets_matrix=None,
    ):

        self.profiler.tic()
        if isinstance(activity, np.ndarray):
            activity_data = f_continuous(np.float32(activity))
        else:
            activity_data = f_continuous(np.float32(activity.data))
        self.profiler.rec_project_make_continuous()

        # By default, the center of the imaging volume is at the center of the scanner; no rotation
        tx = 0.5 * (
            self.activity_size[0] - self.activity_size[0] / self.activity_shape[0]
        )
        ty = 0.5 * (
            self.activity_size[1] - self.activity_size[1] / self.activity_shape[1]
        )
        tz = 0.5 * (
            self.activity_size[2] - self.activity_size[2] / self.activity_shape[2]
        )
        if transformation is None:
            transformation = RigidTransform((tx, ty, tz, 0, 0, 0))
        else:
            transformation = copy.copy(transformation)
            transformation.x = transformation.x + tx
            transformation.y = transformation.y + ty
            transformation.z = transformation.z + tz

        # Optionally project with a sparsity pattern not equal to sparsity associated to the loaded prompts data
        # Note: if prompts have not been loaded, self.get_sparsity() assumes no compression.
        if sparsity is None:
            sparsity = get_sparsity(self.binning, self.prompts)

        # Optionally project only to a subset of projection planes
        if subsets_matrix is None:
            self.profiler.tic()
            sparsity_subset = sparsity
            angles = self.binning.get_angles()
            self.profiler.rec_project_get_angles()
        else:
            self.profiler.tic()
            sparsity_subset = sparsity.get_subset(subsets_matrix)
            self.profiler.rec_project_get_subset_sparsity()
            self.profiler.tic()
            angles = self.binning.get_angles(subsets_matrix)
            self.profiler.rec_project_get_angles()

        scale = (
            1.0
        )  # FIXME: change this according to the input unit measure - check how this is done in project_attenuation
        step_size_mm = self.activity_projection_parameters.sample_step
        step_size = step_size_mm / scale

        offsets = sparsity_subset.offsets
        locations = sparsity_subset.locations
        activations = np.ones([angles.shape[1], angles.shape[2]], dtype="uint32")

        # print locations[:,0:20]
        # print locations.flags
        # print sparsity.locations[:,0:20]
        # print sparsity.locations.flags

        # print "project activity"
        # print "activity",activity_data.shape
        # print "offsets",offsets.shape
        # print "locations",locations.shape
        # print "activations",activations.shape
        # print "angles.shape",angles.shape

        # Call the raytracer
        self.profiler.tic()
        projection_data, timing = PET_project_compressed(
            activity_data,
            None,
            offsets,
            locations,
            activations,
            angles.shape[2],
            angles.shape[1],
            angles,
            self.binning.N_u,
            self.binning.N_v,
            self.binning.size_u,
            self.binning.size_v,
            self.activity_size[0],
            self.activity_size[1],
            self.activity_size[2],
            0.0,
            0.0,
            0.0,
            transformation.x,
            transformation.y,
            transformation.z,
            transformation.theta_x,
            transformation.theta_y,
            transformation.theta_z,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            self.activity_projection_parameters.gpu_acceleration,
            self.activity_projection_parameters.N_samples,
            self.activity_projection_parameters.sample_step,
            self.activity_projection_parameters.background_activity,
            0.0,
            self.activity_projection_parameters.truncate_negative_values,
            self.activity_projection_parameters.direction,
            self.activity_projection_parameters.block_size,
        )

        self.profiler.rec_project_projection()
        self.profiler.rec_projection(timing)

        # Create object PET_Projection: it contains the raw projection data and the description of the projection geometry
        # and sparsity pattern.
        time_bins = np.int32( [0, 1000.0])  # 1 second - projection returns a rate - by design

        self.profiler.tic()
        projection_data = projection_data * step_size
        self.profiler.rec_project_scale()

        self.profiler.tic()
        projection = PET_Projection(
            self.binning,
            projection_data,
            sparsity.offsets,
            sparsity.locations,
            time_bins,
            subsets_matrix,
        )
        self.profiler.rec_project_wrap()

        # Optionally scale by sensitivity, attenuation, time, global sensitivity
        # if attenuation is not None:
        #    projection.data = projection.data * attenuation.data #attenuation.compress_as(projection).data
        # if self.sensitivity is not None:
        #    projection.data = projection.data * self.sensitivity.data.reshape(projection_data.shape)
        # self.sensitivity.compress_as(projection).data

        return projection

def backproject_activity(self, projection, transformation=None, subsets_matrix=None):
        # By default, the center of the imaging volume is at the center of the scanner
        t0 = time.time()
        tx = 0.5 * (
            self.activity_size[0] - self.activity_size[0] / self.activity_shape[0]
        )
        ty = 0.5 * (
            self.activity_size[1] - self.activity_size[1] / self.activity_shape[1]
        )
        tz = 0.5 * (
            self.activity_size[2] - self.activity_size[2] / self.activity_shape[2]
        )
        if transformation is None:
            transformation = RigidTransform((tx, ty, tz, 0, 0, 0))
        else:
            transformation = copy.copy(transformation)
            transformation.x = transformation.x + tx
            transformation.y = transformation.y + ty
            transformation.z = transformation.z + tz

        if not isinstance(projection, np.ndarray):
            if not subsets_matrix is None:
                self.profiler.tic()
                projection_subset = projection.get_subset(subsets_matrix)
                self.profiler.rec_backpro_get_subset()
                self.profiler.tic()
                sparsity_subset = projection_subset.sparsity
                angles = projection_subset.get_angles()
                self.profiler.rec_backpro_get_angles()
                projection_data = np.float32(projection_subset.data)
            else:
                self.profiler.tic()
                projection_subset = projection
                sparsity_subset = projection.sparsity
                angles = projection_subset.get_angles()
                self.profiler.rec_backpro_get_angles()
                projection_data = np.float32(projection_subset.data)
        else:
            sparsity = get_sparsity(self.binning, self.prompts)
            if not subsets_matrix is None:
                # doesn't look right
                self.profiler.tic()
                indexes = subsets_matrix.flatten() == 1
                projection_data = np.float32(
                    projection.swapaxes(0, 1).reshape(
                        (
                            sparsity.N_axial * sparsity.N_azimuthal,
                            self.binning.N_u,
                            self.binning.N_v,
                        )
                    )[indexes, :, :]
                )
                self.profiler.rec_backpro_get_subset_data()
                self.profiler.tic()
                sparsity_subset = sparsity.get_subset(subsets_matrix)
                self.profiler.rec_backpro_get_subset_sparsity()
                self.profiler.tic()
                angles = self.binning.get_angles(subsets_matrix)
                self.profiler.rec_backpro_get_angles()
            else:
                self.profiler.tic()
                sparsity_subset = sparsity
                angles = self.binning.get_angles()
                self.profiler.rec_backpro_get_angles()
                projection_data = np.float32(projection)

        offsets = sparsity_subset.offsets
        locations = sparsity_subset.locations
        activations = np.ones(
            [sparsity_subset.N_azimuthal, sparsity_subset.N_axial], dtype=np.uint32
        )

        scale = ( 1.0 )  # FIXME: change this according to the input unit measure - check how this is done in project_attenuation
        step_size_mm = self.activity_projection_parameters.sample_step
        step_size = step_size_mm / scale

        # print "backproject activity"
        # print "projection_data",projection_data.shape
        # print "offsets",offsets.shape
        # print "locations",locations.shape
        # print "activations",activations.shape
        # print "angles",angles.shape
        # print angles[:,0,0:5]
        # print offsets[0:3,0:5]
        # print locations[:,0:5]
        # time.sleep(0.2)

        # Call ray-tracer
        self.profiler.tic()
        backprojection_data, timing = PET_backproject_compressed(
            projection_data,
            None,
            offsets,
            locations,
            activations,
            angles.shape[2],
            angles.shape[1],
            angles,
            self.binning.N_u,
            self.binning.N_v,
            self.binning.size_u,
            self.binning.size_v,
            self.activity_shape[0],
            self.activity_shape[1],
            self.activity_shape[2],
            self.activity_size[0],
            self.activity_size[1],
            self.activity_size[2],
            0.0,
            0.0,
            0.0,
            transformation.x,
            transformation.y,
            transformation.z,
            transformation.theta_x,
            transformation.theta_y,
            transformation.theta_z,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            self.activity_backprojection_parameters.gpu_acceleration,
            self.activity_backprojection_parameters.N_samples,
            self.activity_backprojection_parameters.sample_step,
            self.activity_backprojection_parameters.background_activity,
            0.0,
            self.activity_backprojection_parameters.direction,
            self.activity_backprojection_parameters.block_size,
        )

        self.profiler.rec_backpro_backprojection()
        self.profiler.rec_backprojection(timing)

        self.profiler.tic()
        backprojection_data = backprojection_data * step_size

        self.profiler.rec_backpro_scale()

        self.profiler.tic()
        backprojection = make_Image3D_activity( shape = np.float32(self.activity_shape),
                                      size = np.float32(self.activity_size),
                                      T_scanner_to_world = self.transform_scanner_to_world,
                                      data=backprojection_data)
        self.profiler.rec_backpro_wrap()

        return backprojection


def get_gradient_activity(
        self,
        activity,
        attenuation=None,
        unit_activity="Bq/mm3",
        transformation_activity=None,
        sparsity=None,
        duration_ms=None,
        subset_size=None,
        subset_mode="random",
        subsets_matrix=None,
        azimuthal_range=None,
        separate_additive_terms=False,
        epsilon=None,
    ):
        # Optionally use only a subset of the projections - use, in order: subsets_matrix; subset_size, subset_mode and az_range
        if subsets_matrix is None:
            if subset_size is not None:
                if subset_size >= 0:
                    subsets_matrix = self._subsets_generator.new_subset(
                        subset_mode, subset_size, azimuthal_range
                    )

        # Optionally use the specified value of epsilon (small number added to the denominator is divisions)
        if epsilon is None:
            epsilon = EPS

        if attenuation is None:
            attenuation = 1.0

        if self.prompts is None:
            print("self.prompts is None, please set prompts. ")
            return
            # FIXME : throw an error

        # By default use the timing information stored in the prompts, however optionally enable overriding
        if duration_ms is None:
            duration_ms = self.prompts.get_duration()
        duration_sec = duration_ms / 1000.0

        # Precompute attenuation*sensitivity - FIXME: do it only is the subset, same for other calculation in proj space
        alpha = self.scale_activity
        prompts = self.prompts
        randoms = self.randoms
        scatter = self.scatter
        sensitivity = self.sensitivity
        if randoms is None:
            randoms = 0.0
        if scatter is None:
            scatter = 0.0
        if sensitivity is None or sensitivity is 1.0:
            att_sens = attenuation
        else:
            att_sens = sensitivity * attenuation

        att_sens = att_sens.get_subset(subsets_matrix)

        # Compute the firt term of the gradient: backprojection of the sensitivity of the scanner
        # If it is requested that the gradient is computed using all the projection measurements, use the
        # memoized normalization. self.get_normalization() takes care of memoization.
        #        gradient_term1 = self.get_normalization(att_sens, transformation_activity, sparsity, duration_ms, subsets_matrix, epsilon=epsilon)
        norm = PET_Projection(self.binning, data=1.0, subsets_matrix=subsets_matrix)
        gradient_term1 = self.backproject_activity(norm)
        print("the two lines above (1033-1034 static.py) are temporary")
        self._time_profiling_record_norm()

        # Compute the second term of the gradient: backprojection of the ratio between the measurement and the projection of
        # current activity estimate... Ordinary Poisson to include scatter and randoms.
        projection = self.project_activity(
            activity,
            unit=unit_activity,
            transformation=transformation_activity,
            sparsity=sparsity,
            subsets_matrix=subsets_matrix,
        )
        self._time_profiling_record_projection()
        prompts_subset = prompts.get_subset(subsets_matrix)
        gradient_term2 = self.backproject_activity(
            prompts_subset
            / (
                projection
                + randoms / (att_sens * alpha * duration_sec + epsilon)
                + scatter / (attenuation * alpha * duration_sec + epsilon)
                + epsilon
            ),
            transformation=transformation_activity,
        )
        self._time_profiling_record_backprojection()
        if separate_additive_terms:
            return (gradient_term1, gradient_term2, subsets_matrix)
        else:
            gradient = gradient_term1 + gradient_term2
            return (gradient, subsets_matrix)

def get_gradient_attenuation(
        self,
        attenuation,
        activity,
        sparsity=None,
        duration_ms=None,
        subset_size=None,
        subset_mode="random",
        subsets_matrix=None,
        azimuthal_range=None,
        epsilon=None,
    ):
        # Optionally use only a subset of the projections - use, in order: subsets_matrix; subset_size, subset_mode and az_range
        if subsets_matrix is None:
            if subset_size is not None:
                if subset_size >= 0:
                    subsets_matrix = self._subsets_generator.new_subset(
                        subset_mode, subset_size, azimuthal_range
                    )

        # Optionally use the specified value of epsilon (small number added to the denominator is divisions)
        if epsilon is None:
            epsilon = EPS

        if attenuation is None:
            attenuation = 1.0

        if self.prompts is None:
            print("self.prompts is None, please set prompts. ")
            return
            # FIXME : throw an error

        # By default use the timing information stored in the prompts, however optionally enable overriding
        if duration_ms is None:
            duration_ms = self.prompts.get_duration()
        duration_sec = duration_ms / 1000.0

        # Precompute attenuation*sensitivity - FIXME: do it only is the subset, same for other calculation in proj space
        alpha = self.scale_activity
        prompts = self.prompts
        randoms = self.randoms
        scatter = self.scatter
        sensitivity = self.sensitivity
        if randoms is None:
            randoms = 0.0
        if scatter is None:
            scatter = 0.0
        if sensitivity is None:
            sensitivity = 1.0

        attenuation_projection = self.project_attenuation(
            attenuation,
            unit="inv_cm",
            transformation=None,
            sparsity=sparsity,
            subsets_matrix=subsets_matrix,
            exponentiate=True,
        )

        # FIXME: transformation = None
        pr_activity = (
            self.project_activity(
                activity,
                transformation=None,
                sparsity=sparsity,
                subsets_matrix=subsets_matrix,
            )
            * sensitivity
            * attenuation_projection
            * duration_sec
            * alpha
        )
        gradient = self.backproject_attenuation(
            pr_activity
            - prompts
            / (
                randoms / (pr_activity + epsilon)
                + scatter / (pr_activity / (sensitivity + epsilon) + epsilon)
                + 1
            ),
            unit="inv_cm",
            transformation=None,
            sparsity=sparsity,
            subsets_matrix=subsets_matrix,
        )
        return gradient

def estimate_activity_and_attenuation(
        self,
        activity=None,
        attenuation=None,
        iterations=DEFAULT_RECON_ITERATIONS,
        sparsity=None,
        subset_size=DEFAULT_SUBSET_SIZE,
        subset_mode="random",
        epsilon=None,
        subsets_matrix=None,
        azimuthal_range=None,
        show_progressbar=True,
    ):
        # FIXME: save time: don't compute twice the proj of the attenuation
        activity = make_Image3D_activity( shape = np.float32(self.activity_shape),
                                      size = np.float32(self.activity_size),
                                      T_scanner_to_world = self.transform_scanner_to_world,
                                      data=np.ones(self.activity_shape, dtype=np.float32, order="F"))

        attenuation = make_Image3D_attenuation(shape=np.float32(self.attenuation_shape),
                                 size=np.float32(self.attenuation_size),
                                 T_scanner_to_world=self.transform_scanner_to_world,
                                 data=np.zeros(self.attenuation_shape, dtype=np.float32, order="F"))
        if show_progressbar:
            progress_bar = ProgressBar()
            progress_bar.set_percentage(0.1)
        for iteration in range(iterations):
            activity = self.estimate_activity(
                activity,
                attenuation,
                1,
                sparsity,
                subset_size,
                subset_mode,
                epsilon,
                subsets_matrix,
                azimuthal_range,
                show_progressbar=False,
            )
            attenuation = self.estimate_attenuation(
                activity,
                attenuation,
                1,
                sparsity,
                subset_size,
                subset_mode,
                epsilon,
                subsets_matrix,
                azimuthal_range,
                show_progressbar=False,
            )
            if show_progressbar:
                progress_bar.set_percentage((iteration + 1) * 100.0 / iterations)
        if show_progressbar:
            progress_bar.set_percentage(100.0)
        return (activity, attenuation)

def estimate_attenuation(
        self,
        activity=None,
        attenuation=None,
        iterations=DEFAULT_RECON_ITERATIONS,
        sparsity=None,
        subset_size=DEFAULT_SUBSET_SIZE,
        subset_mode="random",
        epsilon=None,
        subsets_matrix=None,
        azimuthal_range=None,
        show_progressbar=True,
    ):
        if show_progressbar:
            progress_bar = ProgressBar()
            progress_bar.set_percentage(0.1)

        if attenuation is None:
            attenuation = make_Image3D_attenuation(shape=np.float32(self.attenuation_shape),
                                                   size=np.float32(self.attenuation_size),
                                                   T_scanner_to_world=self.transform_scanner_to_world,
                                                   data=np.zeros(self.attenuation_shape, dtype=np.float32, order="F"))
        for iteration in range(iterations):
            attenuation = attenuation + self.get_gradient_attenuation(
                attenuation,
                activity,
                sparsity,
                duration_ms=None,
                subset_size=subset_size,
                subset_mode=subset_mode,
                subsets_matrix=subsets_matrix,
                azimuthal_range=azimuthal_range,
                epsilon=epsilon,
            )
            if show_progressbar:
                progress_bar.set_percentage((iteration + 1) * 100.0 / iterations)
        if show_progressbar:
            progress_bar.set_percentage(100.0)
        return attenuation

def estimate_activity(
        self,
        activity=None,
        attenuation=None,
        iterations=DEFAULT_RECON_ITERATIONS,
        sparsity=None,
        subset_size=DEFAULT_SUBSET_SIZE,
        subset_mode="random",
        epsilon=None,
        subsets_matrix=None,
        azimuthal_range=None,
        show_progressbar=True,
    ):
        # Optionally use the specified value of epsilon (small number added to the denominator is divisions)
        if epsilon is None:
            epsilon = EPS

        if self.prompts is None:
            print("self.prompts is None, please set prompts. ")
            return
            # FIXME : throw an error

        duration_ms = self.prompts.get_duration()

        if show_progressbar:
            progress_bar = ProgressBar()
            progress_bar.set_percentage(0.1)

        # print "Projection of the attenuation. "
        if attenuation is None:
            attenuation = self.attenuation
        if attenuation is not None:
            self.attenuation_projection = self.project_attenuation(
                attenuation
            )  # FIXME: now it's only here that this is defined
        else:
            self.attenuation_projection = 1.0

        if activity is None:
            activity = make_Image3D_activity(shape=np.float32(self.activity_shape),
                                             size=np.float32(self.activity_size),
                                             T_scanner_to_world=self.transform_scanner_to_world,
                                             data=np.ones(self.activity_shape, dtype=np.float32, order="F"))

        # FIXME: use transformation - also notice that transformation_activity is always set to None here

        self._time_profiling_reset()
        for iteration in range(iterations):
            [gradient1, gradient2, subsets_matrix] = self.get_gradient_activity(
                activity,
                self.attenuation_projection,
                transformation_activity=None,
                sparsity=sparsity,
                duration_ms=duration_ms,
                subset_size=subset_size,
                subset_mode=subset_mode,
                subsets_matrix=subsets_matrix,
                azimuthal_range=azimuthal_range,
                separate_additive_terms=True,
                epsilon=epsilon,
            )
            activity = activity * gradient2 / (gradient1 + epsilon)
            if show_progressbar:
                progress_bar.set_percentage((iteration + 1) * 100.0 / iterations)
        if show_progressbar:
            progress_bar.set_percentage(100.0)
        return activity


def mlem_reconstruction(
        self,
        iterations=10,
        activity=None,
        attenuation_projection=None,
        transformation=None,
        azimuthal_range=None,
        show_progressbar=True,
        title_progressbar=None,
        SaveAll=False,
        SaveDisk=False,
        savepath=""
):
    if show_progressbar:
        if title_progressbar is None:
                title_progressbar = "MLEM Reconstruction"
        progress_bar = ProgressBar(
                color=C.LIGHT_BLUE, title=title_progressbar)
        progress_bar.set_percentage(0.0)

    if activity is None:
        activity = make_Image3D_activity(shape=np.float32(self.activity_shape),
                                         size=np.float32(self.activity_size),
                                         T_scanner_to_world=self.transform_scanner_to_world,
                                         data=np.ones(self.activity_shape, dtype=np.float32, order="F"))

    if self.sensitivity is None:
        sensitivity = self.prompts.copy()
        sensitivity.data = 0.0 * sensitivity.data + 1
        self.set_sensitivity(sensitivity)

    if SaveAll:
        activity_all = np.ones(
            (self.activity_shape[0],
             self.activity_shape[1],
             self.activity_shape[2],
             iterations),
            dtype=np.float32)

    self.profiler.reset()
    for i in range(iterations):
        if not show_progressbar:
            if iterations >= 15:
                if i == iterations - 1:
                    print("iteration ", (i + 1), "/", iterations)
                elif i + 1 == 1:
                    print("iteration ", (i + 1), "/", iterations)
                elif (np.int32(i + 1) / 5) * 5 == i + 1:
                    print("iteration ", (i + 1), "/", iterations)
            else:
                print("iteration ", (i + 1), "/", iterations)
        subsets_matrix = None
        activity = osem_step(self, activity,
                             subsets_matrix,
                             attenuation_projection,
                             transformation)

        if SaveAll:
            activity_all[:, :, :, i] = np.flip(activity.data, (0, 1))

        if SaveDisk:
            activity.save_to_file(savepath + 'activity_recon_%d.nii' % i)

        if show_progressbar:
            progress_bar.set_percentage((i + 1) * 100.0 / iterations)

    return activity


def osem_reconstruction(
        self,
        iterations=10,
        activity=None,
        attenuation_projection=None,
        subset_mode="oreder_axial",
        subset_size=64,
        transformation=None,
        azimuthal_range=None,
        show_progressbar=True,
        title_progressbar=None,
        SaveAll=False,
        SaveDisk=False,
        savepath=""
    ):
        if show_progressbar:
            if title_progressbar is None:
                title_progressbar = "OSEM Reconstruction"
            progress_bar = ProgressBar(
                color=C.LIGHT_BLUE, title=title_progressbar)
            progress_bar.set_percentage(0.0)

        if activity is None:
            activity = make_Image3D_activity(shape=np.float32(self.activity_shape),
                                             size=np.float32(self.activity_size),
                                             T_scanner_to_world=self.transform_scanner_to_world,
                                             data=np.ones(self.activity_shape, dtype=np.float32, order="F"))

        if self.sensitivity is None:
            sensitivity = self.prompts.copy()
            sensitivity.data = 0.0 * sensitivity.data + 1
            self.set_sensitivity(sensitivity)

        if SaveAll:
            activity_all = np.ones(
                (self.activity_shape[0],
                 self.activity_shape[1],
                 self.activity_shape[2],
                 iterations),
                dtype=np.float32)

        subsets_generator = SubsetGenerator(self.binning.N_azimuthal,
                                            self.binning.N_axial)

        self.profiler.reset()
        for i in range(iterations):
            if not show_progressbar:
                if iterations >= 15:
                    if i == iterations - 1:
                        print("iteration ", (i + 1), "/", iterations)
                    elif i + 1 == 1:
                        print("iteration ", (i + 1), "/", iterations)
                    elif (np.int32(i + 1) / 5) * 5 == i + 1:
                        print("iteration ", (i + 1), "/", iterations)
                else:
                    print("iteration ", (i + 1), "/", iterations)

            subsets_matrix = subsets_generator.new_subset(subset_mode, subset_size, azimuthal_range)
            activity = osem_step(self, activity,
                                 subsets_matrix,
                                 attenuation_projection,
                                 transformation)

            if SaveAll:
                activity_all[:, :, :, i] = np.flip(activity.data, (0,1))

            if SaveDisk:
                activity.save_to_file(savepath + 'activity_recon_%d.nii' % i)

            if show_progressbar:
                progress_bar.set_percentage((i + 1) * 100.0 / iterations)
        if SaveAll:
            return activity, activity_all
        else:
            return activity


def osem_step(
        self,
        activity,
        subsets_matrix=None,
        attenuation_projection=None,
        transformation=None,
        gradient_prior_type=None,
        gradient_prior_args=(),
    ):
        epsilon = 1e-08

        self.profiler.rec_iteration()
        self.profiler.tic()
        prompts = self.prompts
        if self._use_compression:
            prompts = prompts.uncompress_self()
        self.profiler.rec_uncompress()

        duration_ms = prompts.get_duration()
        if duration_ms is None:
            print(
                "Acquisition duration unknown (self.prompts.time_bins undefined); assuming 60 minutes. "
            )
            duration_ms = 1000 * 60 * 60
        duration = duration_ms / 1000.0
        alpha = self.scale_activity

        if attenuation_projection is not None:
            self.profiler.tic()
            attenuation_projection = attenuation_projection.get_subset(subsets_matrix)
            self.profiler.rec_get_subset_attenuation()
        elif self.attenuation_projection is not None:
            self.profiler.tic()
            attenuation_projection = self.attenuation_projection.get_subset(
                subsets_matrix
            )
            self.profiler.rec_get_subset_attenuation()
        elif self.attenuation is not None:
            print("Projecting attenuation")
            self.attenuation_projection = self.project_attenuation(self.attenuation)
            self.profiler.tic()
            attenuation_projection = self.attenuation_projection.get_subset(
                subsets_matrix
            )
            self.profiler.rec_get_subset_attenuation()
            print("Done")
        else:
            attenuation_projection = 1.0

        if self.sensitivity is not None:
            self.profiler.tic()
            sens_x_att = self.sensitivity.get_subset(subsets_matrix)
            self.profiler.rec_get_subset_sensitivity()
            self.profiler.tic()
            sens_x_att = sens_x_att * attenuation_projection
            self.profiler.rec_compose_various()
        else:
            sens_x_att = attenuation_projection
        if np.isscalar(sens_x_att):
            sens_x_att = sens_x_att * np.ones(prompts.data.shape, dtype=np.float32)

        if self.randoms is not None:
            randoms = self.randoms
            if self._use_compression:
                self.profiler.tic()
                randoms = randoms.uncompress_self()
                self.profiler.rec_uncompress()
            self.profiler.tic()
            randoms = randoms.get_subset(subsets_matrix)
            self.profiler.rec_get_subset_randoms()
            self.profiler.tic()
            randoms = (randoms + epsilon) / \
                      (sens_x_att * alpha * duration + epsilon)
            self.profiler.rec_compose_randoms()

        if self.scatter is not None:
            self.profiler.tic()
            mscatter = self.scatter.get_subset(subsets_matrix)
            self.profiler.rec_get_subset_scatter()
            self.profiler.tic()
            mscatter = (mscatter + epsilon) / \
                       (attenuation_projection * alpha * duration + epsilon)
            self.profiler.rec_compose_scatter()

            # Scale scatter: this is used in dynamic and kinetic imaging,
            # when scatter is calculated using the ativity for a time period
            # longer than the current frame:
            if self.scatter.get_duration() is not None:
                if self.scatter.get_duration() > 1e-6:
                    self.profiler.tic()
                    mscatter = mscatter * duration / self.scatter.get_duration()
                    self.profiler.rec_compose_scatter()

        if gradient_prior_type is not None:
            if gradient_prior_type == "smooth":
                gradient_prior_func = smoothness_prior
            elif gradient_prior_type == "kinetic":
                gradient_prior_func = kinetic_model_prior
            elif gradient_prior_type == "both":
                gradient_prior_func = kinetic_plus_smoothing_prior
        else:
            gradient_prior_func = None

        # print duration, alpha
        self.profiler.tic()
        norm = self.backproject_activity(
            sens_x_att * alpha * duration,
            transformation=transformation)
        if gradient_prior_func is not None:
            update2 = norm + epsilon - \
                      gradient_prior_func(self, activity, *gradient_prior_args)
        else:
            update2 = norm + epsilon
        self.profiler.rec_backprojection_norm_total()

        self.profiler.tic()
        projection = self.project_activity(
            activity,
            subsets_matrix=subsets_matrix,
            transformation=transformation)
        self.profiler.rec_projection_activity_total()

        self.profiler.tic()
        p = prompts.get_subset(subsets_matrix)
        self.profiler.rec_get_subset_prompts()

        if self.randoms is not None:
            if self.scatter is not None:
                self.profiler.tic()
                s = (projection + randoms + mscatter + epsilon)
                self.profiler.rec_compose_various()
            else:
                self.profiler.tic()
                s = (projection + randoms + epsilon)
                self.profiler.rec_compose_various()
                self.profiler.tic()
        else:
            if self.scatter is not None:
                self.profiler.tic()
                s = (projection + mscatter + epsilon)
                self.profiler.rec_compose_various()
            else:
                self.profiler.tic()
                s = (projection + epsilon)
                self.profiler.rec_compose_various()

        self.profiler.tic()
        update1 = self.backproject_activity(
            p / s, transformation=transformation)
        self.profiler.rec_backprojection_activity_total()

        self.profiler.tic()
        # activity = activity - gradient1
        activity = (activity / update2) * update1
        self.profiler.rec_update()

        return activity

################ Gradient prior terms for OSL version of osem_step #######
def kinetic_model_prior(self, activity_, model, sigma, sf):
    # derivative of a gaussian prior that enforces similarity between recon
    # and fitting
    gradient = (model - sf * activity_.data) / sigma ** 2
    return gradient

def smoothness_prior(self, activity_, importance):
    # kernel = ones((3,3,1))
    # kernel[1,1] = -8.0
    kernel = np.asarray([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])  # 3D laplacian operator
    gradient = ndimage.convolve(
            activity_.data, kernel, mode='constant', cval=0.0)
    return importance * gradient

def kinetic_plus_smoothing_prior(
        self,
        activity_,
        sources,
        sigma,
        sf,
        importance
):
    return self.kinetic_model_prior(activity_, sources, sigma, sf) + self.smoothness_prior(activity_, importance)
