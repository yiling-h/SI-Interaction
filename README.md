# Installation of the SI-Interaction Repository
## Installing Git Repo & Virtual Environment
The following tutorials requires a command line interface, e.g. Terminal on MacOS.
While the code should work in general for all python versions, python >= 3.10 is preferred.

Installing [homebrew](https://brew.sh/)
1. Homebrew is required for installation of `pipx`. This requires a `sudo` command, but oftentimes on UM-affiliated computing clusters, this is not allowed for individual users. However, this step is optional as `pipx` can be safely replaced by `pip`.

If you were able to install homebrew, then install `pipx`:
1. brew install `pipx`
2. ```pipx ensurepath```
3. open new terminal (so that path changes take effect)

Use `pipx` to install virtualenv: `pipx install virtualenv`
1. Alternatively, if installation of pipx failed / is not authorized on your computer or on Great Lakes, it turns out replacing `pipx` with vanilla `pip` also works. (For Great Lakes usage, one may install on their home directory by specifying the installation with `--user`)
2. To install with `pip` on a cluster on which you don't have global installation authorization, install on your home directory using
```pip install <package> --user```
Replace `<package>` with the name of the package to be installed, e.g. `virtualenv`, `numpy`, ...

Clone the repository pertaining to your project from git:
1. Go to the repo: [SI for Interaction](https://github.com/yiling-h/SI-Interaction)
2. Click on `Code`
3. Copy the https link, for the above project it would be: "https://github.com/yiling-h/SI-Interaction.git"
4. Go to terminal and type 
```git clone https://github.com/yiling-h/SI-Interaction.git```

cd into `selective-inference` repo

Init and populate submodules: `git submodule update --init`

Create a new python3 virtual environment: e.g., `virtualenv env3 -p python3.10`
or in general `virtualenv env3 -p python3`

Active the environment: `source env3/bin/activate`

pip will likely fail to install `regreg` from `pypi` (where it looks for packages by default) because it's missing some wheels and doesn't have a source package there). Instead, install it from git
1. ```pip install git+https://github.com/regreg/regreg.git```
2. It will look like it fail but probably didn't: it tries to build a wheel for regreg which fails, it then falls back on setup.py install but there's a deprecation warning that starting with pip 21 it won't fall back anymore.
#%% md
Use pip to install the rest. 
1. "requirements.txt" is a file that contains a list of package names to be installed in one-shot. It turns out sometimes this list is not comprehensive enough for some test/simulation files. Please install additional packages as needed if you have trouble running certain codes.
2. Again, on a cluster, you may need the `--user` option whenever pip is used.