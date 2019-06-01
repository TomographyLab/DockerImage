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


# Import occiput:
import numpy as np
from tomolab.Core.Print import array_to_string
from tomolab.Core.Print import millisec_to_min_sec, pretty_print_large_number
from tomolab.DataSources.FileSources.Files import guess_file_type_by_name
from tomolab.DataSources.PET.PET_sinogram import import_interfile_projection, import_h5f_projection
from tomolab.DataSources.PET.PET_volume import import_interfile_volume
from tomolab.Transformation.Transformations import Transform_Identity
from tomolab.Reconstruction.PET.PET_projection import PET_Projection, Binning
from tomolab.Reconstruction.PET.PET_projection import display_PET_Projection_geometry
from tomolab.Reconstruction.PET.PET_raytracer import ProjectionParameters, BackprojectionParameters
from tomolab.ScannerGeometries.PET_scanners import *  # Generic, get_scanner_by_name
from tomolab.Reconstruction.PET.PET_subsets import SubsetGenerator
from tomolab.Reconstruction.PET.PET_profiler import ReconstructionProfiler

from tomolab._removed import StaticReconstruction as static_rec
from tomolab.DataSources.PET import PET_listmode

# Set verbose level
# This is a global setting for occiput. There are 3 levels of verbose:
# high, low, no_printing
from tomolab.global_settings import *
set_verbose_no_printing()

try:
    import pylab
except BaseException:
    has_pylab = False
else:
    has_pylab = True

# Default parameters
DEFAULT_SUBSET_SIZE = 24
DEFAULT_RECON_ITERATIONS = 10
DEFAULT_N_TIME_BINS = 15
EPS = 1e-6


class PET_Static_Scan:
    """PET Static Scan. """

    def __init__(self):
        # by default, use GPU.
        self.use_gpu(True)
        # set scanner geometry and load_interfile interface
        self.set_scanner('Generic')
        # memoization of activity.
        self.activity = None
        # memoization of attenuation.
        self.attenuation = None
        # memoization of attenuation projection.
        self.attenuation_projection = None
        # sensitivity is a permanent parameter.
        self.sensitivity = None
        # measurement: prompts. Initialized as empty data structure.
        self.prompts = None
        self.randoms = None
        self.scatter = None
        # normalization volume - for all projections - memoize
        self._normalization = (None)
        # If True, the normalization volume needs to be recomputed
        self._need_normalization_update = (True)
        self.use_compression(False)

        self.set_transform_scanner_to_world(
                    Transform_Identity(
                        map_from="scanner",
                        map_to="world"))
        self.profiler = ReconstructionProfiler()
        self.resolution = self.activity_size / self.activity_shape
        self.pixel_size = self.resolution[:2]
        self.slice_thickness = self.resolution[2]

    import_listmode = PET_listmode.import_listmode
    osem_reconstruction = static_rec.osem_reconstruction
    mlem_reconstruction = static_rec.mlem_reconstruction
    estimate_activity = static_rec.estimate_activity
    estimate_attenuation = static_rec.estimate_attenuation
    estimate_activity_and_attenuation = \
                    static_rec.estimate_activity_and_attenuation
    backproject_activity = static_rec.backproject_activity
    project_activity = static_rec.project_activity
    project_attenuation = static_rec.project_attenuation

    def set_transform_scanner_to_world(self, transform):
        # FIXME: verify that the transform maps from 'scanner' to 'world'
        self.transform_scanner_to_world = transform

    def set_activity_shape(self, activity_shape):
        activity_shape = np.asarray(activity_shape)
        if not len(activity_shape) == 3:
            print("Invalid activity shape")  # FIXME: raise invalid input error
        else:
            self.activity_shape = activity_shape
        try:
            self.set_activity_resolution()
        except:
            pass

    def set_activity_size(self, activity_size):
        activity_size = np.asarray(activity_size)
        if not len(activity_size) == 3:
            print("Invalid activity size")  # FIXME: raise invalid input error
        else:
            self.activity_size = activity_size
        self._adapt_line_step_size_activity()
        try:
            self.set_activity_resolution()
        except:
            pass

    def set_activity_resolution(self):
        self.resolution = self.activity_size / self.activity_shape
        self.pixel_size = self.resolution[:2]
        self.slice_thickness = self.resolution[2]

    def set_activity_scale(self, scale):
        self.scale_activity = scale

    def set_attenuation_shape(self, attenuation_shape):
        attenuation_shape = np.asarray(attenuation_shape)
        if not len(attenuation_shape) == 3:
            print("Invalid attenuation shape")  # FIXME: raise invalid input error
        else:
            self.attenuation_shape = attenuation_shape

    def set_attenuation_size(self, attenuation_size):
        attenuation_size = np.asarray(attenuation_size)
        if not len(attenuation_size) == 3:
            print("Invalid attenuation size")  # FIXME: raise invalid input error
        else:
            self.attenuation_size = attenuation_size
        self._adapt_line_step_size_attenuation()

    def _adapt_line_step_size_activity(
        self
    ):  # FIXME: move this calculation in the raytracer
        if not hasattr(self, "activity_size"):
            activity_size = np.float32([0, 0, 0])
        elif self.activity_size is None:
            activity_size = np.float32([0, 0, 0])
        else:
            activity_size = np.float32(self.activity_size)
        diagonal = np.sqrt((activity_size ** 2).sum())
        self.activity_projection_parameters.sample_step = (
            diagonal / self.activity_projection_parameters.N_samples
        )
        self.activity_backprojection_parameters.sample_step = (
            diagonal / self.activity_backprojection_parameters.N_samples
        )

    def _adapt_line_step_size_attenuation(
        self
    ):  # FIXME: move this calculation in the raytracer
        if not hasattr(self, "attenuation_size"):
            attenuation_size = np.float32([0, 0, 0])
        elif self.attenuation_size is None:
            attenuation_size = np.float32([0, 0, 0])
        else:
            attenuation_size = np.float32(self.attenuation_size)
        diagonal = np.sqrt((attenuation_size ** 2).sum())
        self.attenuation_projection_parameters.sample_step = (
            diagonal / self.attenuation_projection_parameters.N_samples
        )
        self.attenuation_backprojection_parameters.sample_step = (
            diagonal / self.attenuation_backprojection_parameters.N_samples
        )

    def set_binning(self, binning):
        if isinstance(binning, Binning):
            self.binning = binning
        else:
            self.binning = Binning(binning)
        self._subsets_generator = SubsetGenerator(
            self.binning.N_azimuthal, self.binning.N_axial
        )
        return self.binning

    def set_scanner(self, scanner):
        try:
            scanner = get_scanner_by_name(scanner)
            self.scanner = scanner()
        except BaseException:
            try:
                self.scanner = scanner()
            except BaseException:
                raise NotImplementedError

        self.activity_projection_parameters = ProjectionParameters()
        self.activity_backprojection_parameters = BackprojectionParameters()
        self.activity_projection_parameters.N_samples = (
            self.scanner.activity_N_samples_projection_DEFAULT
        )
        self.activity_projection_parameters.sample_step = (
            self.scanner.activity_sample_step_projection_DEFAULT
        )
        self.activity_backprojection_parameters.N_samples = (
            self.scanner.activity_N_samples_backprojection_DEFAULT
        )
        self.activity_backprojection_parameters.sample_step = (
            self.scanner.activity_sample_step_backprojection_DEFAULT
        )

        self.set_activity_shape(self.scanner.activity_shape_DEFAULT)
        self.set_activity_size(self.scanner.activity_size_DEFAULT)

        self.activity_projection_parameters.gpu_acceleration = self._use_gpu
        self.activity_backprojection_parameters.gpu_acceleration = self._use_gpu

        self.attenuation_projection_parameters = ProjectionParameters()
        self.attenuation_backprojection_parameters = BackprojectionParameters()
        self.attenuation_projection_parameters.N_samples = (
            self.scanner.attenuation_N_samples_projection_DEFAULT
        )
        self.attenuation_projection_parameters.sample_step = (
            self.scanner.attenuation_sample_step_projection_DEFAULT
        )
        self.attenuation_backprojection_parameters.N_samples = (
            self.scanner.attenuation_N_samples_backprojection_DEFAULT
        )
        self.attenuation_backprojection_parameters.sample_step = (
            self.scanner.attenuation_sample_step_backprojection_DEFAULT
        )

        self.set_attenuation_shape(self.scanner.attenuation_shape_DEFAULT)
        self.set_attenuation_size(self.scanner.attenuation_size_DEFAULT)

        self.attenuation_projection_parameters.gpu_acceleration = self._use_gpu
        self.attenuation_backprojection_parameters.gpu_acceleration = self._use_gpu

        binning = Binning()
        binning.size_u = self.scanner.size_u
        binning.size_v = self.scanner.size_v
        binning.N_u = self.scanner.N_u
        binning.N_v = self.scanner.N_v
        binning.N_axial = self.scanner.N_axial
        binning.N_azimuthal = self.scanner.N_azimuthal
        binning.angles_axial = self.scanner.angles_axial
        binning.angles_azimuthal = self.scanner.angles_azimuthal
        self.binning = binning

        self.set_activity_scale(self.scanner.scale_activity)

        self._subsets_generator = SubsetGenerator(
            self.binning.N_azimuthal, self.binning.N_axial
        )

    def use_gpu(self, use_it):
        self._use_gpu = use_it

    def use_compression(self, use_it):
        self._use_compression = use_it
        if not use_it:
            if self.prompts is not None:
                if self.prompts.is_compressed():
                    self.set_prompts(self.prompts.uncompress_self())
            if self.randoms is not None:
                if self.randoms.is_compressed():
                    self.set_randoms(self.randoms.uncompress_self())
            if self.sensitivity is not None:
                if self.sensitivity.is_compressed():
                    self.set_sensitivity(self.sensitivity.uncompress_self())
            if self.scatter is not None:
                if self.scatter.is_compressed():
                    self.set_scatter(self.scatter.uncompress_self())
        else:
            if hasattr(self, "_use_compression"):
                if self._use_compression is False and use_it is True:
                    # FIXME
                    # print "Not able to compress once uncompressed.
                    # Please implement PET_Projection.uncompress_self() to "
                    # print "enable this functionality. "
                    return
            if self.prompts is not None:
                if not self.prompts.is_compressed():
                    self.set_prompts(self.prompts.compress_self())
            if self.randoms is not None:
                if not self.randoms.is_compressed():
                    self.set_randoms(self.randoms.compress_self())
            if self.sensitivity is not None:
                if not self.sensitivity.is_compressed():
                    self.set_sensitivity(self.sensitivity.compress_self())
            if self.scatter is not None:
                if not self.scatter.is_compressed():
                    self.set_scatter(self.scatter.compress_self())

    ####################################################################################################################

    def import_prompts(self, filename, datafile=""):
        filetype = guess_file_type_by_name(filename)
        if filetype == "interfile_projection_header":
            projection = import_interfile_projection(
                filename,
                self.binning,
                self.scanner.michelogram,
                datafile,
                load_time=True,
            )
        elif filetype == "h5":
            projection = import_h5f_projection(filename)
        else:
            print("PET.import_prompts: file type unknown. ")
            return
        projection.data = np.float32(projection.data)
        if self._use_compression is False:
            projection = projection.uncompress_self()
        self.set_prompts(projection)

    def import_attenuation(
        self, filename, datafile="", filename_hardware="", datafile_hardware=""
    ):
        filetype = guess_file_type_by_name(filename)
        if filetype == "interfile_volume_header":
            volume = import_interfile_volume(filename, datafile)
        elif filetype == "nifti":
            print(
                "Nifti attenuation file not supported. Everything is ready to implement this, please implement it. "
            )
            # FIXME: if nifti files are used, sum the hardware image using resampling in the common space
        elif filetype == "h5":
            print(
                "H5 attenuation file not supported. Everything is ready to implement this, please implement it. "
            )
            # FIXME: if h5 files are used, sum the hardware image using resampling in the common space
        elif filetype == "mat":
            print(
                "Matlab attenuation file not supported. Everything is ready to implement this, please implement it. "
            )
        else:
            print(
                (
                    "PET.import_attenuation: file type of %s unknown. Unable to load_interfile attenuation tomogram. "
                    % filename
                )
            )
            return
        if filename_hardware != "":
            filetype = guess_file_type_by_name(filename_hardware)
            if filetype == "interfile_volume_header":
                volume_hardware = import_interfile_volume(
                    filename_hardware, datafile_hardware
                )
            else:
                print("File type of %s unknown. Unable to load_interfile hardware attenuation tomogram. "
                    % filename_hardware)
            volume.data = volume.data + volume_hardware.data
        volume.data = np.float32(volume.data)
        self.set_attenuation(volume)

    def import_attenuation_projection(self, filename, datafile=""):
        filetype = guess_file_type_by_name(filename)
        if filetype == "interfile_projection_header":
            projection = import_interfile_projection(
                filename,
                self.binning,
                self.scanner.michelogram,
                datafile,
                load_time=True,
            )
        elif filetype == "h5":
            projection = import_h5f_projection(filename)
        else:
            print("PET.import_attenuation_projection: file type unknown. ")
            return
        projection.data = np.float32(projection.data)
        if self._use_compression is False:
            projection = projection.uncompress_self()
        self.set_attenuation_projection(projection)

    # FIXME: when importing, compress if compression is enabled

    def import_sensitivity(self, filename, datafile="", vmin=0.00, vmax=1e10):
        filetype = guess_file_type_by_name(filename)
        if filetype == "h5":
            sensitivity = import_h5f_projection(filename)
        elif filetype == "interfile_projection_header":
            sensitivity = import_interfile_projection(
                filename,
                self.binning,
                self.scanner.michelogram,
                datafile,
                True,
                vmin,
                vmax,
            )
            if self.prompts is not None:
                # FIXME: sensitivity loaded from interfile with some
                # manufacturers has non-zero value
                # where there are no detectors - set to zero where data is zero
                # (good approx only for long acquisitions).
                # See if there is a better way to handle this.
                sensitivity.data[self.prompts.data == 0] = 0
            else:  # FIXME: see comment two lines up
                print(
                    "Warning: If loading real scanner data, please "
                    "load_interfile prompts before loading the sensitivity. \n"
                    "Ignore this message if this is a simulation. \n"
                    "See the source code for more info. ")
        elif filetype == "mat":
            print("Sensitivity from Matlab not yet implemented. "
                  "All is ready, please spend 15 minutes and implement. ")
            return
        else:
            print("File type unknown. ")
            return
        sensitivity.data = np.float32(sensitivity.data)
        if self._use_compression is False:
            sensitivity = sensitivity.uncompress_self()
        self.set_sensitivity(sensitivity)

    def import_scatter(self, filename, datafile="", duration_ms=None):
        filetype = guess_file_type_by_name(filename)
        if filetype == "interfile_projection_header":
            projection = import_interfile_projection(
                filename, self.binning, self.scanner.michelogram, datafile
            )
        elif filetype == "h5":
            projection = import_h5f_projection(filename)
        else:
            print("PET.import_scatter: file type unknown. ")
            return
        projection.data = np.float32(projection.data)
        if self._use_compression is False:
            projection = projection.uncompress_self()
        self.set_scatter(projection, duration_ms)

    def import_randoms(self, filename, datafile=""):
        filetype = guess_file_type_by_name(filename)
        if filetype == "interfile_projection_header":
            projection = import_interfile_projection(
                filename, self.binning, self.scanner.michelogram, datafile
            )
        elif filetype == "h5":
            projection = import_h5f_projection(filename)
        else:
            print("PET.import_randoms: file type unknown. ")
            return
        projection.data = np.float32(projection.data)
        if self._use_compression is False:
            projection = projection.uncompress_self()
        self.set_randoms(projection)

    ###########################################################################

    def set_prompts(self, prompts):
        if isinstance(prompts, PET_Projection):
            self.prompts = prompts
            self.sparsity = (
                self.prompts.sparsity
            )  # update self.sparsity (self.sparsity exists to store sparsity
            # information in case there is no prompts data)
        #            self.set_binning(prompts.get_binning()) #FIXME: check if it is compatible with the scanner
        elif self.prompts is not None:
            prompts = PET_Projection(
                self.prompts.get_binning(),
                prompts,
                self.prompts.sparsity.offsets,
                self.prompts.sparsity.locations,
                self.prompts.get_time_bins(),
            )
            self.prompts = prompts
        else:
            print(
                "Prompts data should be an instance of PET_Projection or an array whose dimension"
            )
            print("matches the sparsity pattern of the current projection data. ")
            # FIXME: raise input error and to a try-except when creating the instance of PET_Projection

    def set_scatter(self, scatter, duration_ms=None):
        self.scatter = scatter
        if duration_ms is not None:
            self.scatter.time_bins = np.int32([0, duration_ms])

    def set_randoms(self, randoms):
        if isinstance(randoms, PET_Projection):
            self.randoms = randoms
            self.sparsity_delay = (
                self.randoms.sparsity
            )  # update self.sparsity (self.sparsity exists to store
            # sparsity information in case there is not randoms data)
            # self.set_binning(randoms.get_binning())   #FIXME: make sure binning is consistent with randoms
        elif self.randoms is not None:
            randoms = PET_Projection(
                self.randoms.get_binning(),
                randoms,
                self.randoms.sparsity.offsets,
                self.randoms.sparsity.locations,
                self.randoms.get_time_bins(),
            )
            self.randoms = randoms
        else:
            print(
                "Delay randoms data should be an instance of PET_Projection or an array whose dimension"
            )
            print("matches the sparsity pattern of the current projection data. ")
            # FIXME: raise input error and to a try-except when creating the instance of PET_Projection

    def set_sensitivity(self, sensitivity):
        # FIXME: verify type: PET_projection or nd_array (the latter only in full sampling mode)
        self.sensitivity = sensitivity

    def set_attenuation_projection(self, attenuation_projection):
        self.attenuation_projection = attenuation_projection

    def set_attenuation(self, attenuation):
        self.attenuation = attenuation

    ####################################################################################################################

    def get_prompts(self):
        return self.prompts

    def get_scatter(self):
        return self.scatter

    def get_randoms(self):
        return self.randoms

    def get_sensitivity(self):
        return self.sensitivity

    def get_attenuation_projection(self):
        return self.attenuation_projection

    def get_attenuation(self):
        return self.attenuation

    def get_activity(self):
        return self.activity

    def get_normalization(
        self,
        attenuation_times_sensitivity=None,
        transformation=None,
        sparsity=None,
        duration_ms=None,
        subsets_matrix=None,
        epsilon=None,
    ):
        # FIXME: memoization
        if attenuation_times_sensitivity is None:
            attenuation_times_sensitivity = (
                self.sensitivity
            )  # FIXME: include attenuation here - memoization mumap proj
        if (
            np.isscalar(attenuation_times_sensitivity)
            or attenuation_times_sensitivity is None
        ):
            attenuation_times_sensitivity = np.ones(self.prompts.data.shape)
        if duration_ms is None:
            duration_ms = self.prompts.get_duration()
        duration_sec = duration_ms / 1000.0
        alpha = self.scale_activity
        normalization = self.backproject_activity(
            attenuation_times_sensitivity * duration_sec * alpha,
            transformation,
            subsets_matrix,
        )
        return normalization

    ####################################################################################################################

    def export_prompts(self, filename):
        self.get_prompts().save_to_file(filename)

    def export_sensitivity(self, filename):
        if self.sensitivity is None:
            print("Sensitivity has not been loaded")
        else:
            self.get_sensitivity().save_to_file(filename)

    def export_scatter(self, filename):
        self.get_randoms().save_to_file(filename)

    def export_randoms(self, filename):
        self.get_randoms().save_to_file(filename)

    def export_attenuation_projection(self, filename):
        self.get_attenuation_projection().save_to_file(filename)

    ####################################################################################################################

    def quick_inspect(self, index_axial=0, index_azimuthal=5, index_bin=60):
        if self.randoms is not None and not np.isscalar(self.randoms):
            randoms = self.randoms.to_nd_array()[
                index_axial, index_azimuthal, :, index_bin
            ]
        else:
            randoms = 0.0
        if self.prompts is not None and not np.isscalar(self.prompts):
            prompts = self.prompts.to_nd_array()[
                index_axial, index_azimuthal, :, index_bin
            ]
        else:
            prompts = 0.0
        if self.sensitivity is not None:
            if not np.isscalar(self.sensitivity):
                sensitivity = self.sensitivity.to_nd_array()[
                    index_axial, index_azimuthal, :, index_bin
                ]
            else:
                sensitivity = self.sensitivity
        else:
            sensitivity = 1.0
        if self.scatter is not None and not np.isscalar(self.scatter):
            scatter = self.scatter.to_nd_array()[
                index_axial, index_azimuthal, :, index_bin
            ]
            if self.scatter.get_duration() is not None:
                if self.scatter.get_duration() > 1e-6:
                    if self.prompts.get_duration() is not None:
                        if self.prompts.get_duration() > 1e-6:
                            scatter = (
                                scatter
                                * self.prompts.get_duration()
                                / self.scatter.get_duration()
                            )
        else:
            scatter = 0.0
        if has_pylab:
            pylab.plot(prompts - randoms)
            pylab.hold(1)
            pylab.plot(sensitivity * scatter, "g")
        else:
            print(
                "quick_inspect uses Pylab to display imaging data. Please install Pylab. "
            )

    def brain_crop(self, bin_range=(100, 240)):
        if self._use_compression is True:
            print("Projection cropping currently only works with uncompressed data. ")
            print(
                "In order to enable cropping, please complete the implementation of PET_Projection.get_subset()"
            )
            print("Now PET_Projection.get_subset() only works with uncompressed data. ")
            return
        if hasattr(self, "_cropped"):
            return
        A = bin_range[0]
        B = bin_range[1]
        self.binning.size_u = (1.0 * self.binning.size_u) / self.binning.N_u * (B - A)
        self.binning.N_u = B - A
        if self.prompts is not None:
            self.prompts = self.prompts.crop((A, B))
        if self.randoms is not None:
            self.randoms = self.randoms.crop((A, B))
        if self.scatter is not None:
            self.scatter = self.scatter.crop((A, B))
        if self.sensitivity is not None:
            self.sensitivity = self.sensitivity.crop((A, B))
        self._cropped = True

    '''
    def volume_render(self, volume, scale=1.0):
        # FIXME: use the VolumeRender object in occiput.Visualization (improve it), the following is a quick fix:
        [offsets, locations] = PET_initialize_compression_structure(180, 1, 256, 256)
        if isinstance(volume, np.ndarray):
            volume = np.float32(volume)
        else:
            volume = np.float32(volume.data)
        subsets_generator = SubsetGenerator(1, 180)
        subsets_matrix = subsets_generator.all_active()
        mask = uniform_cylinder(
            volume.shape,
            volume.shape,
            [0.5 * volume.shape[0], 0.5 * volume.shape[1], 0.5 * volume.shape[2]],
            0.5 * min(volume.shape[0] - 1, volume.shape[1]),
            volume.shape[2],
            2,
            1,
            0,
        )
        volume[np.where(mask.data == 0)] = 0.0
        direction = 7
        block_size = 512
        proj, timing = PET_project_compressed(
            volume,
            None,
            offsets,
            locations,
            subsets_matrix,
            180,
            1,
            np.pi / 180,
            0.0,
            256,
            256,
            256.0,
            256.0,
            256.0,
            256.0,
            256.0,
            256.0,
            256.0,
            256.0,
            128.0,
            128.0,
            128.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1,
            256,
            1.5,
            0.0,
            0.0,
            0,
            direction,
            block_size,
        )
        proj[np.where(proj > proj.max() / scale)] = proj.max() / scale
        binning = Binning()
        binning.N_axial = 180
        binning.N_azimuthal = 1
        binning.angles_axial = np.float32(np.linspace(0, np.pi - np.pi / 180.0, 180))
        binning.angles_azimuthal = np.float32(np.linspace(0, 0, 1))
        binning.size_u = 256.0
        binning.size_v = 256.0
        binning.N_u = 256
        binning.N_v = 256
        projection = PET_Projection(binning, proj, offsets, locations)
        return projection.uncompress_self()
    '''

    def display_geometry(self):
        return display_PET_Projection_geometry()

    def __repr__(self):
        s = "Static PET acquisition:  \n"
        s = s + " - Time_start:                   %s \n" % millisec_to_min_sec(
            self.prompts.get_time_start()
        )
        s = s + " - Time_end:                     %s \n" % millisec_to_min_sec(
            self.prompts.get_time_end()
        )
        s = s + " - Duration:                     %s \n" % millisec_to_min_sec(
            self.prompts.get_time_end() - self.prompts.get_time_start()
        )
        s = s + " - N_counts:                     %d \n" % self.prompts.get_integral()
        s = (
            s
            + " - N_locations:                  %d \n"
            % self.prompts.sparsity.get_N_locations()
        )
        # s = s+" - compression_ratio:            %d \n"%self.prompts.sparsity.compression_ratio
        # s = s+" - listmode_loss:                %d \n"%self.prompts.sparsity.listmode_loss
        s = s + " = Scanner: \n"
        s = s + "     - Name:                     %s \n" % self.scanner.model
        s = s + "     - Manufacturer:             %s \n" % self.scanner.manufacturer
        s = s + "     - Version:                  %s \n" % self.scanner.version
        s = s + " * Binning: \n"
        s = s + "     - N_axial bins:             %d \n" % self.binning.N_axial
        s = s + "     - N_azimuthal bins:         %d \n" % self.binning.N_azimuthal
        s = s + "     - Angles axial:             %s \n" % array_to_string(
            self.binning.angles_axial
        )
        s = s + "     - Angles azimuthal:         %s \n" % array_to_string(
            self.binning.angles_azimuthal
        )
        s = s + "     - Size_u:                   %f \n" % self.binning.size_u
        s = s + "     - Size_v:                   %f \n" % self.binning.size_v
        s = s + "     - N_u:                      %s \n" % self.binning.N_u
        s = s + "     - N_v:                      %s \n" % self.binning.N_v
        return s

    def _repr_html_(self):
        if not has_ipy_table:
            return "Please install ipy_table."
        if self.scanner is not None:
            table_data = [
                ["Time_start", millisec_to_min_sec(self.prompts.get_time_start())],
                ["Time_end", millisec_to_min_sec(self.prompts.get_time_end())],
                [
                    "Duration",
                    millisec_to_min_sec(
                        self.prompts.get_time_end() - self.prompts.get_time_start()
                    ),
                ],
                ["N_counts", pretty_print_large_number(self.prompts.get_integral())],
                [
                    "N_locations",
                    pretty_print_large_number(self.prompts.sparsity.get_N_locations),
                ],
                # ['compression_ratio',print_percentage(self.compression_ratio)],
                # ['listmode_loss',self.listmode_loss],
                ["Scanner Name", self.scanner.model],
                ["Scanner Manufacturer", self.scanner.manufacturer],
                ["Scanner Version", self.scanner.version],
            ]
        else:
            table_data = [
                ["Time_start", millisec_to_min_sec(self.prompts.get_time_start())],
                ["Time_end", millisec_to_min_sec(self.prompts.get_time_end())],
                [
                    "Duration",
                    millisec_to_min_sec(
                        self.prompts.get_time_end() - self.prompts.get_time_start()
                    ),
                ],
                ["N_counts", pretty_print_large_number(self.prompts.get_integral())],
                [
                    "N_locations",
                    pretty_print_large_number(self.prompts.sparsity.get_N_locations()),
                ],
            ]
            # ['compression_ratio',print_percentage(self.compression_ratio)],
            # ['listmode_loss',self.listmode_loss], ]
        table = ipy_table.make_table(table_data)
        table = ipy_table.apply_theme("basic_left")
        # table = ipy_table.set_column_style(0, color='lightBlue')
        table = ipy_table.set_global_style(float_format="%3.3f")
        return table._repr_html_()
