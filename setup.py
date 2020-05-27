import setuptools

__name__ == "__main__" and setuptools.setup(
    **dict(
        name="lsst_dashboard",
        version="v0.0.2a",
        author="quansight",
        author_email="dharhas@quansight.com",
        description="LSST Data Explorer",
        long_description_content_type="text/markdown",
        url="https://github.com/quansight/lsst_dashboard",
        python_requires=">=3.6",
        license="BSD-3-Clause",
        install_requires=["click"],
        include_package_data=True,
        packages=setuptools.find_packages(),
        entry_points={
            "console_scripts": [
                "lsst_data_explorer = lsst_dashboard.cli:start_dashboard",
                "lsst_data_repartition = lsst_dashboard.cli:repartition",
                "repartition_dataset = lsst_dashboard.cli:repartition_dataset",
            ]
        },
    )
)
