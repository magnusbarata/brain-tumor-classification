import os
import re
import glob
import shutil
import subprocess

def get_first_dcm(dcm_dir):
    """Get the first image in a directory of dicom files"""
    return sorted(
        glob.glob(f'{dcm_dir}/*.dcm'),
        key=lambda path: int(re.sub('\D', '', os.path.basename(path)))
    )[0]

def captk_preproc(datadir, captk_dir='CaPTk/CaPTk/1.8.1', out_dir='data/captk-reg'):
    """Apply [CaPTk BraTS Pre-processing Pipeline](https://cbica.github.io/CaPTk/preprocessing_brats.html)
    to a dataset.
    
    Args:
      datadir: Root of data directory. Note that the directory structure must be the same
        as the [original competition directory structure](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/data).
      captk_dir: Location of captk binary file. `BraTSPipeline.cwl` must be available at the same
        location.
      out_dir: Output directory.
    """
    if os.path.exists(out_dir):
        usr_in = input(f'{out_dir} exist. Overwrite? (y/[n]): ')
        if usr_in.lower() == 'y':
            shutil.rmtree(out_dir)
        else:
            print('Canceled.')
            raise SystemExit
    os.makedirs(out_dir)
    subprocess.run(['cp', f'{datadir}/train_labels.csv', f'{out_dir}'])
    tempdir = f'{out_dir}/temp'

    cohorts = ['train', 'test']
    for cohort in cohorts:
        cases = sorted([f.name for f in os.scandir(f'{datadir}/{cohort}') if f.is_dir()])
        for i, case in enumerate(cases):
            print(f'{f"{i+1} / {len(cases)}" :>{12}} : {case}')
            casedir = f'{out_dir}/{cohort}/{case}'
            os.makedirs(casedir)

            try:
                subprocess.check_output([
                    f'{captk_dir}/captk',
                    f'{captk_dir}/BraTSPipeline.cwl',
                    '-s', '0',
                    '-b', '0',
                    '-d', '0',
                    '-i', '0',
                    '-fl', get_first_dcm(f'{datadir}/{cohort}/{case}/FLAIR'),
                    '-t1', get_first_dcm(f'{datadir}/{cohort}/{case}/T1w'),
                    '-t1c', get_first_dcm(f'{datadir}/{cohort}/{case}/T1wCE'),
                    '-t2', get_first_dcm(f'{datadir}/{cohort}/{case}/T2w'),
                    '-o', tempdir
                ], stderr=subprocess.STDOUT)
                subprocess.check_output([
                    'mv', f'{tempdir}/FL_to_SRI.nii.gz', f'{casedir}/FLAIR.nii.gz'])
                subprocess.check_output([
                    'mv', f'{tempdir}/T1_to_SRI.nii.gz', f'{casedir}/T1w.nii.gz'])
                subprocess.check_output([
                    'mv', f'{tempdir}/T1CE_to_SRI.nii.gz', f'{casedir}/T1wCE.nii.gz'])
                subprocess.check_output([
                    'mv', f'{tempdir}/T2_to_SRI.nii.gz', f'{casedir}/T2w.nii.gz'])
                shutil.rmtree(tempdir)
            except subprocess.CalledProcessError as e:
                print(f'Error: {e.stderr}')
        