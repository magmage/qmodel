qmodel
====

Documentation and API reference:
https://mage.uber.space/dokuwiki/material/qmodel - [PDF version](doc.pdf)

Installation
====

The qmodel package can be installed with pip from a local clone of the
repository. From within this directory, simply run
```sh
pip install .
```
Optionally one can install with the `-e` (`--editable`) flag to let changes in
the source code take effect without having to perform a fresh install. The
command for this would be
```sh
pip install -e .
```

To uninstall, run
```sh
pip uninstall qmodel
```
You may need to be in another directory than this one for the command to work.

To update after a new version has been 'released', pull the changes to your
local clone of the repository and run
```sh
pip install . --upgrade
```
