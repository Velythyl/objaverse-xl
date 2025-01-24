import os
import subprocess
import sys

import glob
import json
import multiprocessing
import os
import platform
import random
import subprocess
import tempfile
import time
import zipfile
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import fire
import fsspec
import GPUtil
import pandas as pd
from loguru import logger

import objaverse.xl as oxl
from objaverse.utils import get_uid_from_str

def filedir():
    # Get the directory of the current file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    return current_file_directory

BLENDER_TOPLEVEL_DIR_NAME = "blender-3.2.2-linux-x64"

def blender_path():
    return f"{filedir()}/{BLENDER_TOPLEVEL_DIR_NAME}"

def ensure_blender():
    if os.path.exists(f"{filedir()}/{BLENDER_TOPLEVEL_DIR_NAME}"):
        return

    curdir = os.getcwd()
    os.chdir(filedir())

    subprocess.run(
        "wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz && tar -xf blender-3.2.2-linux-x64.tar.xz && rm blender-3.2.2-linux-x64.tar.xz",
        check=False,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stdout
    )

    os.chdir(curdir)




def render(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[str, Any],
    num_renders: int,
    render_dir: str,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
    successful_log_file: Optional[str] = "handle-found-object-successful.csv",
    failed_log_file: Optional[str] = "handle-found-object-failed.csv",
) -> bool:
    """Called when an object is successfully found and downloaded.

    Here, the object has the same sha256 as the one that was downloaded with
    Objaverse-XL. If None, the object will be downloaded, but nothing will be done with
    it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): File identifier of the 3D object.
        sha256 (str): SHA256 of the contents of the 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        num_renders (int): Number of renders to save of the object.
        render_dir (str): Directory where the objects will be rendered.
        only_northern_hemisphere (bool): Only render the northern hemisphere of the
            object.
        gpu_devices (Union[int, List[int]]): GPU device(s) to use for rendering. If
            an int, the GPU device will be randomly selected from 0 to gpu_devices - 1.
            If a list, the GPU device will be randomly selected from the list.
            If 0, the CPU will be used for rendering.
        render_timeout (int): Number of seconds to wait for the rendering job to
            complete.
        successful_log_file (str): Name of the log file to save successful renders to.
        failed_log_file (str): Name of the log file to save failed renders to.

    Returns: True if the object was rendered successfully, False otherwise.
    """

    def log_processed_object(csv_filename: str, *args) -> None:
        """Log when an object is done being used.

        Args:
            csv_filename (str): Name of the CSV file to save the logs to.
            *args: Arguments to save to the CSV file.

        Returns:
            None
        """
        args = ",".join([str(arg) for arg in args])
        # log that this object was rendered successfully
        # saving locally to avoid excessive writes to the cloud
        dirname = os.path.expanduser(f"{render_dir}/logs/")
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, csv_filename), "a", encoding="utf-8") as f:
            f.write(f"{time.time()},{args}\n")

    def zipdir(path: str, ziph: zipfile.ZipFile) -> None:
        """Zip up a directory with an arcname structure.

        Args:
            path (str): Path to the directory to zip.
            ziph (zipfile.ZipFile): ZipFile handler object to write to.

        Returns:
            None
        """
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                # this ensures the structure inside the zip starts at folder/
                arcname = os.path.join(os.path.basename(root), file)
                ziph.write(os.path.join(root, file), arcname=arcname)

    save_uid = get_uid_from_str(file_identifier)
    args = f"--object_path '{local_path}' --num_renders {num_renders}"

    # get the GPU to use for rendering
    using_gpu: bool = True
    gpu_i = 0
    if isinstance(gpu_devices, int) and gpu_devices > 0:
        num_gpus = gpu_devices
        gpu_i = random.randint(0, num_gpus - 1)
    elif isinstance(gpu_devices, list):
        gpu_i = random.choice(gpu_devices)
    elif isinstance(gpu_devices, int) and gpu_devices == 0:
        using_gpu = False
    else:
        raise ValueError(
            f"gpu_devices must be an int > 0, 0, or a list of ints. Got {gpu_devices}."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        # get the target directory for the rendering job
        target_directory = os.path.join(temp_dir, save_uid)
        os.makedirs(target_directory, exist_ok=True)
        args += f" --output_dir {target_directory}"

        # check for Linux / Ubuntu or MacOS
        if platform.system() == "Linux" and using_gpu:
            args += " --engine BLENDER_EEVEE"
        elif platform.system() == "Darwin" or (
            platform.system() == "Linux" and not using_gpu
        ):
            # As far as I know, MacOS does not support BLENER_EEVEE, which uses GPU
            # rendering. Generally, I'd only recommend using MacOS for debugging and
            # small rendering jobs, since CYCLES is much slower than BLENDER_EEVEE.
            args += " --engine CYCLES"
        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

        # check if we should only render the northern hemisphere
        if only_northern_hemisphere:
            args += " --only_northern_hemisphere"

        # get the command to run
        command = f"{blender_path()}/blender --background --python blender_script.py -- {args}"
        if using_gpu:
            command = f"export DISPLAY=:0.{gpu_i} && {command}"

        # render the object (put in dev null)
        subprocess.run(
            ["bash", "-c", command],
            timeout=render_timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # check that the renders were saved successfully
        png_files = glob.glob(os.path.join(target_directory, "*.png"))
        metadata_files = glob.glob(os.path.join(target_directory, "*.json"))
        npy_files = glob.glob(os.path.join(target_directory, "*.npy"))
        if (
            (len(png_files) != num_renders)
            or (len(npy_files) != num_renders)
            or (len(metadata_files) != 1)
        ):
            logger.error(
                f"Found object {file_identifier} was not rendered successfully!"
            )
            if failed_log_file is not None:
                log_processed_object(
                    failed_log_file,
                    file_identifier,
                    sha256,
                )
            return False

        # update the metadata
        metadata_path = os.path.join(target_directory, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_file = json.load(f)
        metadata_file["sha256"] = sha256
        metadata_file["file_identifier"] = file_identifier
        metadata_file["save_uid"] = save_uid
        metadata_file["metadata"] = metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_file, f, indent=2, sort_keys=True)

        # Make a zip of the target_directory.
        # Keeps the {save_uid} directory structure when unzipped
        with zipfile.ZipFile(
            f"{target_directory}.zip", "w", zipfile.ZIP_DEFLATED
        ) as ziph:
            zipdir(target_directory, ziph)

        # move the zip to the render_dir
        fs, path = fsspec.core.url_to_fs(render_dir)

        # move the zip to the render_dir
        fs.makedirs(os.path.join(path, "renders"), exist_ok=True)
        fs.put(
            os.path.join(f"{target_directory}.zip"),
            os.path.join(path, "renders", f"{save_uid}.zip"),
        )

        # log that this object was rendered successfully
        if successful_log_file is not None:
            log_processed_object(successful_log_file, file_identifier, sha256)

        return True



if __name__ == "__main__":
    ensure_blender()