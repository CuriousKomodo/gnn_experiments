#!/bin/bash
set -e

usage()
{
    echo "usage: setup_env.sh [-r] [-h]
       -r (--reset) Wipe the existing python virtual environment
       -h (--help)  Display this help message"
}

# Default values
RESET=false
ENV_FOLDER=".venv" # This is fixed as poetry installs only in .venv

##### Main
while [[ "$1" != "" ]]; do
    case $1 in
        -r | --reset )          RESET=true
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

if [ "$RESET" = true ] ; then
    echo "Removing virtual environment ${ENV_FOLDER}.."
    rm -rf ./${ENV_FOLDER}
fi

##### cd to current script directory as we expect the build and run scripts here too
cd "${0%/*}"  # https://stackoverflow.com/questions/3349105/how-to-set-current-working-directory-to-the-directory-of-the-script
ROOT_DIR_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Create virtualenv and install dependencies
poetry install

# Add modules to python path
source ./${ENV_FOLDER}/bin/activate

SITE_PKG_FOLDER=$(python -c "from distutils.sysconfig import get_python_lib;print(get_python_lib())")
echo "../../../.." > "${SITE_PKG_FOLDER}"/nate_configure_module_path.pth

deactivate

# allow remote interpreter to access remote system variables
echo "#!/bin/bash -l
${ROOT_DIR_REPO}/${ENV_FOLDER}/bin/python \"\$@\"" > ./${ENV_FOLDER}/bin/python_sys_env_wrapper.sh
chmod 775 ./${ENV_FOLDER}/bin/python_sys_env_wrapper.sh
ln -sf python_sys_env_wrapper.sh ./${ENV_FOLDER}/bin/python_pycharm

printf "\nVirtual environment ready.\n"
printf "Enter 'source %s/bin/activate' to activate.\n" ${ENV_FOLDER}
printf "Enter 'deactivate' to exit.\n"
