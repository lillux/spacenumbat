from setuptools import setup, find_packages

setup(name = 'spacenumbat',
     version = '0.0.1',
     description = 'Haplotype aware CNAs caller with a spatial context',
     url = 'https://github.com/lillux/spacenumbat',
     author = 'Calogero Carlino, Valentina Giansanti, Davide Cittaro',
     author_email = 'calogero.carlino28@gmail.com',
     license = 'GPLv3',
     zip_safe=False,
     install_requires=['numpy',
                       'matplotlib',
                       'scipy',
                       'anndata',
                       'pyranges',
                       'joblib'],
                       
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 3.12',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
              
      packages=find_packages(),
      include_package_data=True,  # Include package data specified in MANIFEST.in
      package_data={
       'spacenumbat.data': ['*.tsv'],
       }
      )