from rever.activity import activity

@activity
def run_tests():
    """Running the test suite."""
    cd tests
    pytest
    cd ..

$ACTIVITIES = ['run_tests']
$PROJECT = "LSST Dashboard"
$ACTIVITIES = ["authors"]

$AUTHORS_FILENAME = "AUTHORS.md"
$AUTHORS_TEMPLATE = """
The $PROJECT project has some great contributors! They are:

{authors}

These have been sorted {sorting_text}.
"""
$AUTHORS_FORMAT= "* [{name}](https://github.com/{github})\n"
$AUTHORS_SORTBY = "alpha"