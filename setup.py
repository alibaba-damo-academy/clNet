from setuptools import setup, find_namespace_packages

setup(name='clNet',
      packages=find_namespace_packages(include=["clnet", "clnet.*"]),
      version='2.0.0',
      description='Segment in the wild: life-long continual learning net (clNet)',
      install_requires=[
            "torch>=2.4.0",
            "spconv",
            "torchvision",
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "opencv-python",
            "albumentations",
            "scipy>=1.11.1",
            "pytorch-lightning>=2.4.0",
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
            "psutil",
      ],
      entry_points={
            'console_scripts': [
                  'clNet_predict = clnet.inference.predict_simple:main'
            ],
      }
      )
