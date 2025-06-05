# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Original author: Reza Hosseini


from typing import Optional

import pkg_resources
from pyhocon import ConfigFactory, ConfigTree


def get_hocon_conf(file: str, path: Optional[str] = None) -> ConfigTree:
    """
    Returns the configuration tree for HOCON files.

    Args:
        file (str): The name of the HOCON file to be read.
        path (Optional[str], optional): The path to the directory containing the file.
            If provided, the full file path will be constructed. Defaults to None.

    Returns:
        ConfigTree: The configuration tree object containing the parsed HOCON file data.

    Raises:
        Exception: If the HOCON file cannot be found or read.
    """
    if path:
        file = f"{path}/{file}"

    # Read the HOCON file
    config = ConfigFactory.parse_file(file)

    return config


def get_metric_conf(metric_family_name: str, path: Optional[str] = None) -> ConfigTree:
    """
    Gets metric names using the `dataset.conf` file.
    This currently requires having the conf files in the same repo or specifying a path.

    Args:
        metric_family_name (str): The name of the metric family to fetch.
        path (Optional[str], optional): The path to the directory containing the configuration files.
            If not provided, a default path will be used. Defaults to None.

    Returns:
        ConfigTree: The configuration tree object containing the parsed HOCON file data for the given metric family.

    Raises:
        Exception: If the HOCON configuration cannot be loaded or if the required keys are missing.
    """
    if not path:
        path = "data/conf/"
    file = pkg_resources.resource_filename(
        "abvelocity", f"{path}{metric_family_name}/dataset.conf"
    )

    return get_hocon_conf(file=file, path=None)


def get_metric_names_via_conf(metric_family_name: str, path: Optional[str] = None) -> list[str]:
    """
    Retrieves the list of metric names from the HOCON configuration.

    Args:
        metric_family_name (str): The name of the metric family to fetch the metrics for.
        path (Optional[str], optional): The path to the directory containing the configuration files.
            If not provided, a default path will be used. Defaults to None.

    Returns:
        list[str]: A list of metric names extracted from the configuration.

    Raises:
        KeyError: If the required keys ('datafiles' or 'metrics') are missing in the configuration.
    """
    config = get_metric_conf(metric_family_name, path)
    metric_names = []
    for metric_info in config["datafiles"][0]["metrics"]:
        metric_names.append(metric_info["name"])

    return metric_names
