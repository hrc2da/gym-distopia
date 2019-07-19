# gym-distopia

## Installation

init the distopia submodule:
`git submodule update --init`
make sure that the 'agent' branch is checked out in distopia

make a virtualenv
`virtualenv venv -p python3`

activate and do a quick pip install on the requirements file. This will install kivy and cython in our virtualenv so that we can install distopia.

`pip install -r requirements.txt`

cd into the distopia directory and pip install an editable version of distopia (in case things change with distopia)

`pip install -e .`

check that you can now import distopia

cd up and install the gym environment

`pip install -e .` or `pip install -e gym-distopia`.
