import setuptools
__name__ == "__main__" and setuptools.setup(**dict(
    name="lsst_dashboard",
    version='v0.0.2a',
    author="quansight",
    author_email="tony.fast@quansight.com",
    description="Conventions for writing code in the notebook.",
    long_description_content_type='text/markdown',
    url="https://github.com/quanisght/lsst_dashboard",
    python_requires=">=3.6",
    license="BSD-3-Clause",
    install_requires=[],
    include_package_data=True,
    packages=setuptools.find_packages(),
    entry_points={
          'console_scripts': [
              'lsst_data_explorer = lsst_dashboard.__main__:main'
          ]
    })
)
