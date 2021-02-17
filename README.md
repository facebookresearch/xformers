An empty repo, with everything set up to automate lint and correctness checks. You can start a new project by adding this project as a remote, and cloning the master branch.

## Config files:
### Code coverage
`.coveragerc`, set up the code coverage threshold that you would like to enforce for your project. This needs to be coupled with a unit test runner which produces a code coverage metric.

### CircleCI
`.circleci` folder, see [the doc](https://circleci.com/docs/2.0/language-python/) for more explanations. The attached CircleCI config only contains the foundations and can be considered as a starting point which will probably not cover your needs.

### EditorConfig
`.editorconfig`, a standard (works with VSCode, Atom, Vim, Emacs, ..) to preconfigure your favorite IDE

### Pre-commit
`.pre-commit-config.yaml`, makes sure that a set of rules are enforced on any code which lands on master. This file allows you to run the exact same checks on your machine at commit time. It is recommended to work in a dedicated virtual env (python virtualenv or conda) and to install [pre-commit](https://pre-commit.com/) locally as well

### Git ignore
`.gitignore`, signal all the files and folders which are not to be tracked by git.

### Pip requirements
`requirements.txt`. Install with `pip install -r requirements.txt`, and make sure that CI does the same, do not maintain a seperate dependency list. It is however possible to seperate your dependencies depending on the workload, so that somebody willing to use your library does not have to install everything required for benchmarking for instance. `requirements-test.txt` is an example of that, this installs all the dependencies required for linting, unit testing and formatting.


##  Documentation
This project proposes [Sphinx](https://www.sphinx-doc.org/en/master/) to build the documentation automatically given docstrings within the python code, and a basic hierarchy specified in the `/docs` subfolder. After the doc building requirements have been installed, the HTML doc can be built with `make html`