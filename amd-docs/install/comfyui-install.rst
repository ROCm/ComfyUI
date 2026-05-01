.. meta::
  :description: installing ComfyUI for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, ComfyUI

.. _comfyui-on-rocm-installation:

********************************************************************
ComfyUI on ROCm installation
********************************************************************

This topic covers setup and installation instructions to help you get started running ComfyUI.

System requirements
====================================================================

To use ComfyUI `0.18.2 <https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.18.2>`__, you need the following prerequisites:

- **ROCm version:** `7.2.0 <https://rocm.docs.amd.com/en/docs-7.2.0/>`__, `7.1.0 <https://rocm.docs.amd.com/en/docs-7.1.0/>`__
- **Operating system:** Ubuntu 24.04, 22.04
- **GPU platform:** AMD Instinct™ MI355X, MI325X, MI300X
- **PyTorch:** `2.10.0a0+git449b176 <https://hub.docker.com/r/rocm/pytorch/tags?name=rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0>`__
- **Python:** `3.12 <https://www.python.org/downloads/release/python-3123/>`__

Install ComfyUI
================================================================================

To install ComfyUI on ROCm, you have the following options:

* :ref:`using-docker-with-comfyui-pre-installed` **(recommended)**
* :ref:`build-comfyui-rocm-docker-image`

After setting up the container with either option, follow the common step to launch the ComfyUI server.

.. _using-docker-with-comfyui-pre-installed:

Use a prebuilt Docker image with ComfyUI pre-installed
--------------------------------------------------------------------------------------

The prebuilt image contains a fully configured ComfyUI installation and all required dependencies pre-installed.

1. Pull the Docker image.

   .. tab-set::

      .. tab-item:: ROCm 7.2.0 + Ubuntu 24.04

         .. code-block:: bash

            docker pull rocm/comfyui:comfyui-0.18.2.amd0_rocm7.2.0_ubuntu24.04

      .. tab-item:: ROCm 7.1.0 + Ubuntu 22.04

         .. code-block:: bash

            docker pull rocm/comfyui:comfyui-0.18.2.amd0_rocm7.1.0_ubuntu22.04

2. Start a Docker container using the image.

   .. tab-set::

      .. tab-item:: ROCm 7.2.0 + Ubuntu 24.04

         .. code-block:: bash

            docker run -it --privileged \
            --rm \
            --device=/dev/kfd \
            --device=/dev/dri \
            --group-add video \
            --cap-add=SYS_PTRACE \
            --security-opt seccomp=unconfined \
            --ipc=host \
            -p 8188:8188 \
            rocm/comfyui:comfyui-0.18.2.amd0_rocm7.2.0_ubuntu24.04

      .. tab-item:: ROCm 7.1.0 + Ubuntu 22.04

         .. code-block:: bash

            docker run -it --privileged \
            --rm \
            --device=/dev/kfd \
            --device=/dev/dri \
            --group-add video \
            --cap-add=SYS_PTRACE \
            --security-opt seccomp=unconfined \
            --ipc=host \
            -p 8188:8188 \
            rocm/comfyui:comfyui-0.18.2.amd0_rocm7.1.0_ubuntu22.04

.. _build-comfyui-rocm-docker-image:

Build from source
--------------------------------------------------------------------------------------

ComfyUI on ROCm can be run directly by setting up a Docker container from scratch.
A Dockerfile is provided in the `https://github.com/ROCm/ComfyUI/blob/amd-integration/docker/Dockerfile.rocm <https://github.com/ROCm/ComfyUI/blob/amd-integration/docker/Dockerfile.rocm>`__ repository to help you get started.

1. Clone the `https://github.com/ROCm/ComfyUI/tree/release/0.18.2.amd0 <https://github.com/ROCm/ComfyUI/tree/release/0.18.2.amd0>`__ repository.

   .. code-block:: bash

      git clone https://github.com/rocm/ComfyUI.git -b amd-integration
      cd ComfyUI

2. Build the Docker image.

   .. code-block:: bash

      docker build --file docker/Dockerfile.rocm --tag comfyui-rocm .

   This will pull the ``rocm/pytorch-training:v25.2`` image and install ComfyUI, the ComfyUI Node Manager, 
   the `https://github.com/rgthree/rgthree-comfy <https://github.com/rgthree/rgthree-comfy>`__ custom nodes, and the required dependencies.

3. Launch a container based on the image.

   .. code-block:: bash

      docker run -it --privileged \
      --rm \
      --device=/dev/kfd \
      --device=/dev/dri \
      --group-add video \
      --cap-add=SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --ipc=host \
      -p 8188:8188 \
      comfyui-rocm


Run ComfyUI
=========================================================================================

To run ComfyUI, you can choose to launch the server remotely or from the command line.

Launch the remote ComfyUI server
--------------------------------------------------------------------------------------

After starting a Docker container with either option above, you can launch the ComfyUI server:

.. code-block:: bash

   python $COMFYUI_PATH/main.py --port 8188 --listen

This starts the server on the default port ``8188``. The port can be changed by setting
the environment variable ``COMFYUI_PORT_HOST`` or by using the ``--port`` flag.

Server options:

- ``--listen``: Allow connections from any network interface (needed for remote or container access).
- ``--port <PORT>``: Change the default port (default: ``8188``). Can also be set via the ``COMFYUI_PORT_HOST`` environment variable.
- ``--gpu-only``: Force all operations to run on the GPU.

Launch ComfyUI from the command line
--------------------------------------------------------------------------------------

1. Start the ComfyUI server from the command line.

   .. code-block:: bash

      python ComfyUI/main.py

   This starts the server and displays a prompt like:

   .. code-block:: text

      To see the GUI go to: http://127.0.0.1:8188

2. Navigate to ``http://127.0.0.1:8188`` in your web browser. You might need to
   replace ``8188`` with the appropriate port number.

   .. image:: ../images/comfyui-main.png
      :align: center
      :alt: Example output with host name and port number
      :width: 600px

3. Search for one of the following templates and download any missing models. See :ref:`comfyui-download-models`.

   .. tab-set::

      .. tab-item:: SD3.5 Simple

         Select **Template** → **Model Filter** → **SD3.5** → **SD3.5 Simple**

         .. image:: ../images/sd3_5-simple-card.png
            :align: center

         Download required models, if missing.

         .. image:: ../images/sd3_5-missing-models.png
            :align: center

      .. tab-item:: Chroma1 Radiance text to image

         Select **Template** → **Model Filter** → **Chroma** → **Chroma1 Radiance text to image**

         .. image:: ../images/chroma1-radiance-tti-card.png
            :align: center

         Download required models, if missing.

         .. image:: ../images/chroma1-radiance-tti-missing-models.png
            :align: center

4. Click **Run**.

The application will use your AMD GPU to convert the prompted text to an image.


Test the ComfyUI installation
=========================================================================================

To verify that ComfyUI was installed correctly, test ROCm PyTorch support.
Inside the running container, confirm that PyTorch detects the GPU:

.. code-block:: bash

   python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

The expected output is ``True`` followed by your AMD GPU device name, for example:

.. code-block:: text

   True
   AMD Instinct MI355X

If you see the version string above, ComfyUI ``0.18.2`` has been installed successfully. You can now use ComfyUI in your projects.


Next Steps
--------------------------------------------------------------------------------------

Now that you have ComfyUI running on your AMD Instinct GPU, you can:

* Explore additional workflow templates
* Create custom workflows from scratch
* Install community-created custom nodes
* Experiment with different models and parameters
* Build your own custom nodes for specialized tasks
