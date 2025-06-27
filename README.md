# Phenocam image processing

[![tests badge](https://github.com/ukceh-rse/image-explorer/actions/workflows/pipeline.yml/badge.svg)](https://github.com/ukceh-rse/image-explorer/actions)
[![docs badge](https://github.com/ukceh-rse/image-explorer/actions/workflows/deploy-docs.yml/badge.svg)](https://ukceh-rse.github.io/image-explorer/)

[Read the docs!](https://ukceh-rse.github.io/image-explorer/)

This repository started as image processing pipelines, search and navigation utilities.

It was originally created as a rapid prototype for COSMOS-UK Phenocam data, using the [UKCEH python project template](https://github.com/NERC-CEH/python-template) and repurposes some of the pipeline code in [plankton_ml](https://github.com/NERC-CEH/python-template).

### Note on dependency versions

We're using the [thingsvision](https://github.com/ViCCo-Group/thingsvision) package to simplify extracting features from different computer vision models. It currently requires python <3.11 and numpy <2. If the approach stays useful, it makes sense to remove `thingsvision` in favour of model-specific code.

## Getting Started

## Set up virtual environment

See the installation instructions for [uv](https://docs.astral.sh/uv/#tool-management).

```
uv python install 3.10
uv sync
source .venv/bin/activate
```

This should handle the `pip install` of dependencies 

### Using the Githook

From the root directory of the repo, run:

```
git config --local core.hooksPath .githooks/
```

This will set this repo up to use the git hooks in the `.githooks/` directory. The hook runs `ruff format --check` and `ruff check` to prevent commits that are not formatted correctly or have errors. The hook intentionally does not alter the files, but informs the user which command to run.

### Installing the package

The docs, tests, and linter packages can be installed together with:

```
pip install -e .[dev]
```

### Run the tests

`python -m pytest`

### Run the pipeline

This includes a Luigi pipeline which does the following work:

* Splits a set of dual-hemisphere images into left and right halves
* `defisheye` to flatten the perspective, and saves the results at 600x600px dimensions
* Extract and store image embeddings using a model from `thingsvision`
* Stores the embedding vectors, and metadata derived from the filename, in a sqlite database

Run with test data and dummy output locations:

`python src/phenocam/pipeline_luigi.py`

This should create a small image set inside `data/images/` and feature embeddings and the sqlite database inside `data/vectors/`

### Run the FastAPI API

This exposes an API around the vector search abstraction, it basically only has one query which is "find me all the URLs whose embeddings are closest to this one"

`fastapi run src/phenocam/data/api.py`

Visit http://localhost:8000/docs

Test the query with input like this:

```
{ 
    "url": "WADDN_20140101_0902_ID405_L.jpg",
    "n_results": 5
}
```

### Run the visualisation

There's a simple, self-contained visualisation of the N closest images done in p5js. It's a static HTML file with a Javascript file that calls the API above. It could do a lot more, depends what questions we now want to ask!

```
cd src/app
python -m http.server 8082
```

Then visit http://localhost:8082

### Building Docs Locally

The documentation is driven by [Sphinx](https://www.sphinx-doc.org/) an industry standard for documentation with a healthy userbase and lots of add-ons. It uses `sphinx-apidoc` to generate API documentation for the codebase from Python docstrings.

To run `sphinx-apidoc` run:

```
# Install your package with optional dependencies for docs
pip install -e .[docs]

cd docs
make apidoc
```

This will populate `./docs/sources/...` with `*.rst` files for each Python module, which may be included into the documentation.

Documentation can then be built locally by running `make html`, or found on the [GitHub Deployment](https://ukceh-rse.github.io/fdri-phenocam).

### Run the Tests

To run the tests run:

```
#Install package with optional dependencies for testing
pip install -e .[test]

pytest
```

### Automatic Versioning

This codebase is set up using [autosemver](https://autosemver.readthedocs.io/en/latest/usage.html#) a tool that uses git commit history to calculate the package version. Each time you make a commit, it increments the patch version by 1. You can increment by:

* Normal commit. Use for bugfixes and small updates
    * Increments patch version: `x.x.5 -> x.x.6`
* Commit starts with `* NEW:`. Use for new features
    * Increments minor version `x.1.x -> x.2.x`
* Commit starts with `* INCOMPATIBLE:`. Use for API breaking changes
    * Increments major version `2.x.x -> 3.x.x`


 
