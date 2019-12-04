#!/bin/sh
cd docs
make html
cd ..
yes | doctr deploy . --built-docs ./docs/_build/html/ --force
