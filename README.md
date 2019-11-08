<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Python Template Project](#python-template-project)
  - [About the project](#about-the-project)
    - [API docs](#api-docs)
    - [Design](#design)
    - [Status](#status)
    - [See also](#see-also)
  - [Getting started](#getting-started)
  - [Setting up your local development environment](#setting-up-your-local-development-environment)
    - [Layout](#layout)
  - [Notes](#notes)
  - [TODO](#todo)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Python Template Project

## About the project

The template is used to create Python project. All Python projects must follow the conventions in the
template. Calling for exceptions must be brought up in the engineering team.

### API docs

The template doesn't have API docs. For web service, please include API docs here, whether it's
auto-generated or hand-written. For auto-generated API docs, you can also give instructions on the
build process.

### Design

The template follows project convention doc.

* [Repository Conventions](https://github.com/caicloud/engineering/blob/master/guidelines/repo_conventions.md)

### Status

The template project is in alpha status.

### See also

* [nirvana project template](https://github.com/caicloud/nirvana-template-project)
* [golang project template](https://github.com/caicloud/golang-template-project)
* [common project template](https://github.com/caicloud/common-template-project)

## Getting started

Below we describe the conventions or tools specific to Python project.

## Setting up your local development environment

We use [pipenv](https://docs.pipenv.org) to manage local development environment, as well as syncing with other collaborators and GitHub. See https://github.com/caicloud/engineering/blob/master/guidelines/python.md for more details about Python versions as well as this tool.

To get started, install `pipenv` first, `cd` to the root of your project, and type:

```sh
pipenv install
```

And then run:

```sh
pipenv shell
```

to enable local dependency shell.

### Layout

```
.
├── .github
│   ├── ISSUE_TEMPLATE.md
│   └── PULL_REQUEST_TEMPLATE.md
├── .gitignore
├── CHANGELOG.md
├── CODEOWNERS
├── Dockerfile
├── Pipfile
├── Pipfile.lock
└── README.md
```

A brief description of the layout:

* `.github` has two template files for creating PR and issue. Please see the files for more details.
* `.gitignore` varies per project, but all projects need to ignore `bin` directory.
* `CHANGELOG.md` contains auto-generated changelog information.
* `CODEOWNERS` contains owners of the project.
* `Dockerfile` is the docker file for building image
* `Pipfile` is for specifying dependencies, managed and updated by `pipenv`
* `Pipfile.lock` is for locking down dependencies, managed and updated by `pipenv`
* `README.md` is a detailed description of the project.

## Notes

## TODO
