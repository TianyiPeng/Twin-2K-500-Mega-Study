# .PHONY: notebook
# .DEFAULT: notebook

SHELL = /bin/bash


uname_m := $(shell uname -m)
uname_s := $(shell uname -s)
ifeq ($(uname_s),Darwin)
ifeq ($(uname_m),arm64)
ARCH_PREFIX := arch -x86_64
endif
endif

CYAN=\033[0;36m
RED=\033[0;31m
ORANGE=\033[38;5;208m
WHITE=\033[1;37m
RST=\033[0m

# Python Environment Setup
NAME := digitaltwin
PYMAJOR := 3
PYREV := 11
PYPATCH := 9
PYVERSION := ${PYMAJOR}.${PYREV}.${PYPATCH}
POETRY_VERSION := 1.8.3
PYENV := ~/.pyenv/versions/${PYVERSION}
VENV_NAME := ${NAME}-${PYVERSION}
VENV_DIR := ${PYENV}/envs/${VENV_NAME}
VENV := ${PYENV}/envs/${VENV_NAME}
EGGLINK := ${VENV}/lib/python${PYMAJOR}.${PYREV}/site-packages/${NAME}.egg-link
BREW_SSL := /usr/local/opt/openssl@1.1
BREW_READLINE := /usr/local/opt/readline
PY_BIN := ${ARCH_PREFIX} ${VENV}/bin/python
PYENV_BIN := ${ARCH_PREFIX} /usr/local/bin/pyenv
POETRY_BIN := PYENV_VERSION=${VENV_NAME} VIRTUAL_ENV=${VENV} ${ARCH_PREFIX} ${VENV}/bin/poetry

# delberately smash this so we keep arm64-homebrew out of our field of view
export PATH = ${VENV}/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/bin:/usr/sbin:/sbin

export LDFLAGS = -L${BREW_SSL}/lib -L${BREW_READLINE}/lib
export CFLAGS = -I${BREW_SSL}/include -I${BREW_READLINE}/include
export CPPFLAGS = -I${BREW_SSL}/include -I${BREW_READLINE}/include
export PYENV_VERSION := ${VENV_NAME}
export VIRTUAL_ENV := ${VENV}

# Python Management
${BREW_READLINE}:
	${ARCH_PREFIX} /usr/local/bin/brew install readline

${BREW_SSL}:
	${ARCH_PREFIX} /usr/local/bin/brew install openssl@1.1

${PYENV}: ${BREW_SSL} ${BREW_READLINE}
	@echo -e "${CYAN}*** Installing python runtime ${WHITE}${PYVERSION}${RST}"
	${PYENV_BIN} install -s ${PYVERSION}

${VENV}: ${PYENV}
	@echo -e "${CYAN}*** Creating a virtualenv for ${WHITE}${VENV_NAME}${RST}"
	${PYENV_BIN} virtualenv ${PYVERSION} ${VENV_NAME}
	${PY_BIN} -m pip install -U pip setuptools wheel
	${PY_BIN} -m pip install -U poetry==${POETRY_VERSION}

.python-version: ${VENV}
	@echo -e "${CYAN}*** Setting our default virtualenv to ${WHITE}${VENV_NAME}${RST}"
	echo ${VENV_NAME} >.python-version

${EGGLINK}: pyproject.toml 
	@echo -e "${CYAN}*** Installing the project into our virtualenv${RST}"
	${POETRY_BIN} install
	touch ${EGGLINK}

setup: .python-version ${EGGLINK}

clean:
	@echo -e "${ORANGE}*** Removing untracked files with git-clean --fdx!${RST}"
	@echo -e "${ORANGE}*** BEWARE! This is aggressive and will even ignore your .gitignore aside from anything in dotfiles. Will do this in interactive mode to avoid losing files.${RST}"
	git clean -fdx -i

nuke: clean
	@echo -e "${RED}*** Nuking your virtualenv: ${WHITE}${VENV_NAME}${RST}"
	rm -f .python-version
	${PYENV_BIN} uninstall -f ${VENV_NAME}
	rm -rf ${VENV_DIR}

lock:
	${POETRY_BIN} lock

# usually there's no reason to uninstall python itself, and reinstalling

tacnuke: nuke
	@echo -e "${RED}*** Nuking your python distribution to bedrock: ${WHITE}${PYVERSION}${RST}"
	${PYENV_BIN} uninstall -f ${PYVERSION}

update: ${EGGLINK}
	@echo -e "${ORANGE}Updating our poetry lockfile from pyproject.toml${RST}"
	${POETRY_BIN} update


format:
	@echo -e "${CYAN}*** formatting with ruff (black compatible)${RST}"
	${POETRY_BIN} run ruff format .
	@echo -e "${CYAN}*** formatting with black (full AST-based formatter)${RST}"
	${POETRY_BIN} run black .
	@echo -e "${CYAN}*** formatting with ruff for imports (iSort compatible)${RST}"
	${POETRY_BIN} run ruff check --select I --fix .
	# @echo -e "${CYAN}*** formatting notebooks with nbconvert (just removing output)${RST}"
	# ${POETRY_BIN} run find . -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \;
	@echo -e "${CYAN}*** running mypy type checks${RST}"
	${POETRY_BIN} run mypy .

lockcheck:
	@echo -e "${CYAN}*** Checking that poetry lock is up to date"
	${POETRY_BIN} lock --check

ruff:
	@echo -e "${CYAN}*** Running ruff${RST}"
	${POETRY_BIN} run ruff check .

ruff-fix:
	@echo -e "${CYAN}*** Running ruff in fix mode${RST}"
	${POETRY_BIN} run ruff check --fix .

mypy:
	@echo -e "${CYAN}*** Running mypy checks${RST}"
	${POETRY_BIN} run mypy .

# so we can identify local development
# notebook: export DEV_PLATFORM := LOCAL: will be useful when we have different environments
notebook:
	${POETRY_BIN} run jupyter notebook --NotebookApp.token= --NotebookApp.disable_check_xsrf=True
