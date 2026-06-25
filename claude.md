This respository will house a locally hosted agentic coding system designed specifically to integrate with Jupyter AI 3.

Design Constraints:
1. This software will be written in Python using pip-installable libraries. UV should be used for ease of use
2. Use a "deps" directory within the repo. There should be an install script that automatically installs the dependencies for the software; any and all pip packages needed should be downloaded to this directory. The install script should check to see if the dependencies for the current platform are present in the deps directory and use them first (in offline install mode) before trying to download them.
3. The default mode should be to use a "builtin" install of llama-cpp-python. However, this should be swappable for any generic IP-based endpoint via configuration

