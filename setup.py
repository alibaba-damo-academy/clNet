from setuptools import setup, find_namespace_packages

setup(name='clNet',
      packages=find_namespace_packages(include=["clnet", "clnet.*"]),
      version='2.0.0',
      description='Segment in the wild: life-long continual learning net (clNet)',
      install_requires=[
            "torch>=2.0.0",
            "torchvision",
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "medpy",
            "opencv-python",
            "albumentations",
            "scipy>=1.7.0",
            "pytorch-lightning",
            "batchgenerators>=0.21",
            "numpy",
            "scikit-learn",
            "SimpleITK",
            "pandas",
            "requests",
            "nibabel",
            "tifffile",
            "matplotlib",
            "einops",
            "line_profiler",
      ],
      entry_points={
            'console_scripts': [
                  'clNet_plan_and_preprocess = clnet.experiment_planning.clnet_plan_and_preprocess:main',
                  'clNet_train = clnet.run.run_training:main',
                  'clNet_predict = clnet.inference.predict_simple:main',
            ],
      }
      )
