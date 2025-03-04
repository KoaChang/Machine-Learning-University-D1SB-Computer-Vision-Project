import os

from setuptools import find_packages, setup

# Declare your non-python data files:
# Files underneath configuration/ will be copied into the build preserving the
# subdirectory structure if they exist.
data_files = []
for root, dirs, files in os.walk("configuration"):
    data_files.append(
        (os.path.relpath(root, "configuration"), [os.path.join(root, f) for f in files])
    )

setup(
    name="ImageGuruModelTraining",
    version="1.0",
    # declare your packages
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    # include data files
    data_files=data_files,
    # defines files which should be bundled with the python code for redistribution
    package_data={"": ["py.typed"]},
    # declare your scripts
    # If you want to create any Python executables in bin/, define them here.
    # This is a three-step process:
    #
    # 1. Create the function you want to run on the CLI in src/image_guru_model_training/cli.py
    #    For convenience I usually recommend calling it main()
    #
    # 2. Uncomment this section of the setup.py arguments; this will create
    #    bin/ImageGuruModelTraining (which you can obviously change!) as a script
    #    that will call your main() function, above.
    #
    # entry_points="""\
    # [console_scripts]
    # ImageGuruModelTraining = image_guru_model_training.cli:main
    # """,
    #
    # 3. Uncomment the Python interpreter and Python-setuptools in the
    #   dependencies section of your Config. This is necessary to guarantee the
    #   presence of a runtime interpreter and for the script generated by
    #   setuptools to find its function.
    #
    # Control whether to install scripts to $ENVROOT/bin. The valid values are:
    # * "default-only": install scripts for the version corresponding to
    #   Python-default in your version set. If this package doesn't build for
    #   that version, you won't get root scripts.
    # * True: always install scripts for some version of python that the package
    #   builds for (in practice, this will be the last version that is built).
    #   Note that in this case, you also need to ensure that the appropriate
    #   runtime interpreter is in the dependency closure of your environment.
    # * <a specific python version, e.g. "python3.6" or "jython2.7">: only
    #   attempt to install root scripts for the specific interpreter version. If
    #   this package is in a version set where that interpreter is not enabled,
    #   you won't get root scripts. You almost certainly don't want this.
    root_script_source_version="default-only",
    # Use the pytest brazilpython runner. Provided by BrazilPython-Pytest.
    test_command="brazilpython_pytest",
    # Use custom sphinx command which adds an index.html that's compatible with
    # code.amazon.com links.
    doc_command="amazon_doc_utils_build_sphinx",
    check_format=False,  # Enable build-time format checking
    test_mypy=False,  # Enable type checking
    test_flake8=False,  # Enable linting at build time
)
