$PROJECT = "LSST Dashboard"
$ACTIVITIES = ['authors',
               'version_bump',  # Changes the version number in various source files (setup.py, __init__.py, etc)
               'tag', # Creates a tag for the new version number
               'ghrelease', # Creates a Github release entry for the new tag
               'push_tag',  # Pushes the tag up to the $TAG_REMOTE
               ]

$AUTHORS_FILENAME = "AUTHORS.md"
$AUTHORS_TEMPLATE = """
The $PROJECT project has some great contributors! They are:

{authors}

These have been sorted {sorting_text} based on github username.
"""
$AUTHORS_FORMAT= "* [{name}](https://github.com/{github})\n"
$AUTHORS_SORTBY = "alpha"

$VERSION_BUMP_PATTERNS = [  # These note where/how to find the version numbers
                         ('lsst_dashboard/__init__.py', '__version__\s*=.*', "__version__ = '$VERSION'"),
                         ('setup.py', 'version\s*=.*,', "version='$VERSION',")
                         ]

$PUSH_TAG_PROTOCOL = 'http'
$GITHUB_ORG = 'quansight'  # Github org for Github releases and conda-forge
$GITHUB_REPO = 'lsst_dashboard'  # Github repo for Github releases  and conda-forge