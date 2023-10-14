# Contributing to the xFormers Repository

We aim to make contributing to this project as easy and transparent as possible.

## Our Development Process

Minor changes and improvements will be released on an ongoing basis, while larger
changes (e.g., changesets implementing a new paper) will be released more periodically.

## Pull Requests

We enthusiastically welcome your pull requests.

1. Fork the repository and create your branch from the `main` branch.
2. If you've added code that needs testing, please include relevant tests.
3. If you've made changes to APIs, ensure that the documentation is updated.
4. Confirm that the test suite passes.
5. Make sure your code adheres to the coding style guidelines.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

To accept your pull request, we require you to submit a CLA.
This is a one-time requirement to work on any of Facebook's open source projects.

Complete your CLA [here](https://code.facebook.com/cla).

## Issues

We use GitHub issues to track public bugs. Please ensure your issue description is clear
and includes sufficient instructions for reproducing the problem.

Facebook offers a [bounty program](https://www.facebook.com/whitehat/) for the safe disclosure of security bugs. 
In such cases, please follow the process outlined on that page and refrain from filing a public issue.

## Environment Setup

```bash
~$ python3 -m venv venv2
~$ source venv2/bin/activate
(venv2) ~$ cd git/template/
(venv2) ~/git/template $ pip3 install -r requirements-test.txt
```

## Coding Style

In your editor, install the [editorconfig](https://editorconfig.org/) extension
to ensure that you are following the same standards as us.

There are two options to guarantee proper code formatting and linting:
* either you run black, mypy and isort before opening up your PR.

```bash
black .
isort . --profile black
flake8 --config .flake8
mypy --ignore-missing-imports --scripts-are-modules --pretty --exclude build/ --exclude stubs/ .
```

* Alternatively, you can just install [pre-commit](https://pre-commit.com/), which automatically runs these checks every time you commit.  
To set it up: 
```bash
pip install pre-commit 
```
then (in the xformers repository, just once)
```bash
pre-commit install 
```

After these steps each of your commits will run the same linting and formatting routines as the xformers continuous integration, which greatly helps getting your PRs all green !

_Read the [editorconfig](.editorconfig) file to understand the exact coding style preferences in detail._

## Testing

### Static analysis

```bash
mypy --ignore-missing-imports --scripts-are-modules --pretty --exclude stubs/ .
```

### Unit tests

```bash
pytest
```

or

``` bash
python -m pytest
```

### Check test coverage

``` bash
python -m pytest --cov-report term --cov=template  tests
```

### CircleCI status

From your PR page, you can review the CircleCI results. For GPU tests, you should see 
output like:

``` bash
...
----- generated xml file: /home/circleci/template/test-results/junit.xml ------
================== 217 passed, 2 xfailed in 218.74s (0:03:38) ==================
CircleCI received exit code 0
```

The number of passed and failed tests should give you an idea of whether your local tests
match with the CI results.

## Commit Guidelines

We follow the same guidelines as AngularJS. Each commit message consists of a **header**,
a **body** and a **footer**.  The header has a special format that includes a **type**,
and a **subject**:

```bash
[<type>] <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

Any line of the commit message cannot be longer 100 characters! This allows the message to be easier
to read on github as well as in various git tools.

### Type

Must be one of the following:

* **feat**: A new feature
* **fix**: A bug fix
* **cleanup**: Changes that do not affect the meaning of the code (white-space, formatting, missing
  semi-colons, dead code removal etc.)
* **refactor**: A code change that neither fixes a bug or adds a feature
* **perf**: A code change that improves performance
* **test**: Adding missing tests or fixing them
* **chore**: Changes to the build process or auxiliary tools and libraries such as documentation
generation
* **docs**: Documentation-only changes

## License

By contributing to *xFormers*, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
