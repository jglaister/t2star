import os

import nipype.pipeline.engine as pe
import nipype.interfaces.ants as ants
import nipype.interfaces.utility as util
import nipype.interfaces.image as image

import os.path

import nipype.interfaces.base as base
import nipype.utils.filemanip as fip


from estimate_t2star import estimate_t2star


class EstimateT2StarInputSpec(base.BaseInterfaceInputSpec):
    t2star_files = base.traits.List(base.File(exists=True, desc='input image', mandatory=True))
    brainmask_file = base.File(exists=True, desc='input image', mandatory=True)
    te_list = base.traits.List(base.traits.Float(), mandatory=True)
    output_prefix = base.traits.Str(desc='Filename for output template', default_value='EstT2star_')


class EstimateT2StarOutputSpec(base.TraitedSpec):
    s0_file = base.File(exists=True, desc='output template')
    t2star_file = base.File(exists=True, desc='output template')
    r2_file = base.File(exists=True, desc='output template')


class EstimateT2Star(base.BaseInterface):
    input_spec = EstimateT2StarInputSpec
    output_spec = EstimateT2StarOutputSpec

    def _run_interface(self, runtime):
        estimate_t2star(self.inputs.t2star_files, self.inputs.brainmask_file, self.inputs.te_list, self.inputs.output_prefix, self.num_threads)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['s0_file'] = os.path.abspath(self.inputs.output_prefix + '_GRE_S0.nii.gz')
        outputs['t2star_file'] = os.path.abspath(self.inputs.output_prefix + '_GRE_T2star.nii.gz')
        outputs['r2_file'] = os.path.abspath(self.inputs.output_prefix + '_GRE_R2.nii.gz')

        return outputs


def create_t2star_workflow(scan_directory: str, te, patient_id: str = None, scan_id: str = None, reorient: str = 'RAS',
                           num_threads: int = 1) -> pe.Workflow:
    '''
    Registers and estimates t2star map
    :param scan_directory:
    :param te:
    :param patient_id:
    :param scan_id:
    :param reorient:
    :param num_threads:
    :return: A :class:'nipype.pipeline.engine.Workflow' object
    :rtype: nipype.pipeline.engine.Workflow
    '''

    name = 't2_star'

    if patient_id is not None and scan_id is not None:
        scan_directory = os.path.join(scan_directory, patient_id, 'pipeline')
        name += '_' + scan_id

    wf = pe.Workflow(name, scan_directory)

    input_node = pe.Node(util.IdentityInterface(['t2star_files', 'target_file', 'brainmask_file']), name='input_node')

    num_t2star_files = len(input_node.outputs.t2star_files)

    # Reorient
    if reorient is not None:
        reorient_to_target = pe.MapNode(image.Reorient(), iterfield=['in_file'], name='reorient_to_target')
        reorient_to_target.inputs.orientation = reorient
        wf.connect(input_node, 't2star_files', reorient_to_target, 'in_file')

    select_first_t2star = pe.Node(util.Split(), name='get_first_t2star')
    select_first_t2star.inputs.splits = [1, num_t2star_files - 1]
    select_first_t2star.inputs.squeeze = True

    if reorient is not None:
        wf.connect(reorient_to_target, 'out_file', select_first_t2star, 'inlist')
    else:
        wf.connect(input_node, 't2star_files', select_first_t2star, 'inlist')

    affine_reg_to_target = pe.Node(ants.Registration(), name='affine_reg_to_target')
    affine_reg_to_target.inputs.dimension = 3
    affine_reg_to_target.inputs.interpolation = 'Linear'
    affine_reg_to_target.inputs.metric = ['MI', 'MI']
    affine_reg_to_target.inputs.metric_weight = [1.0, 1.0]
    affine_reg_to_target.inputs.radius_or_number_of_bins = [32, 32]
    affine_reg_to_target.inputs.sampling_strategy = ['Regular', 'Regular']
    affine_reg_to_target.inputs.sampling_percentage = [0.25, 0.25]
    affine_reg_to_target.inputs.transforms = ['Rigid', 'Affine']
    affine_reg_to_target.inputs.transform_parameters = [(0.1,), (0.1,)]
    affine_reg_to_target.inputs.number_of_iterations = [[100, 50, 25], [100, 50, 25]]
    affine_reg_to_target.inputs.convergence_threshold = [1e-6, 1e-6]
    affine_reg_to_target.inputs.convergence_window_size = [10, 10]
    affine_reg_to_target.inputs.smoothing_sigmas = [[4, 2, 1], [4, 2, 1]]
    affine_reg_to_target.inputs.sigma_units = ['vox', 'vox']
    affine_reg_to_target.inputs.shrink_factors = [[4, 2, 1], [4, 2, 1]]
    affine_reg_to_target.inputs.write_composite_transform = True
    affine_reg_to_target.inputs.initial_moving_transform_com = 1
    affine_reg_to_target.inputs.output_warped_image = False
    wf.connect(select_first_t2star, 'out1', affine_reg_to_target, 'moving_image')
    wf.connect(input_node, 'target_file', affine_reg_to_target, 'fixed_image')

    # Register first t2star file
    transform_echoes = pe.MapNode(ants.ApplyTransforms(), iterfield=['input_image'], name='transform_echoes')
    transform_echoes.inputs.input_image_type = 3
    wf.connect(input_node, 'target_file', transform_echoes, 'reference_image')
    wf.connect(affine_reg_to_target, 'composite_transform', transform_echoes, 'transforms')
    wf.connect(input_node, 't2star_files', transform_echoes, 'input_image')

    estimate = pe.Node(EstimateT2Star(), name='estimate_t2star')
    estimate.interface.num_threads = num_threads
    estimate.inputs.te_list = te
    if patient_id is not None and scan_id is not None:
        estimate.inputs.output_prefix = patient_id + '_' + scan_id
    else:
        estimate.inputs.output_prefix = fip.split_filename(select_first_t2star.outputs.out1)[1]
    wf.connect(transform_echoes, 'output_image', estimate, 't2star_files')
    wf.connect(input_node, 'brainmask_file', estimate, 'brainmask_file')

    #TODO: Copy output to a final folder

    return wf

