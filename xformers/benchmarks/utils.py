# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def pretty_print(results, title, units):
    """ Printout the contents of a dict as a human-readable and Markdown compatible array"""
    print(title)
    header = " Units: {:<40}".format(units)
    print("|" + header + "|" + "".join("{0:<20}|".format(k) for k in results.keys()))

    offset = len(header)
    print(
        "|{}|".format("-" * offset)
        + "".join("{}|".format("-" * 20) for _ in results.keys())
    )

    workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
    for v in results.values():
        for k in v.keys():
            workloads[k].append(v[k])

    for k, w in workloads.items():
        print(
            "|{0:<{offset}}|".format(k, offset=offset)
            + "".join("{:<20}|".format(v) for v in w)
        )

    print("")


def pretty_plot(results, title, units: str, filename=None, dash_key=""):
    """Graph out the contents of a dict.
    Dash key means that if the result label has this key, then it will be displayed with a dash"""

    if not filename:
        filename = title + ".png"

    # Sanitize the filename
    filename = filename.replace(" ", "_").replace("/", "_").replace("-", "_")

    # Gather all the results in "collumns"
    workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
    for v in results.values():
        for k in v.keys():
            workloads[k].append(float(v[k]))

    # Make sure that the plot is big enough
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(10)

    # Display the collections
    for k, v in workloads.items():
        if dash_key and dash_key in k:
            plt.plot(list(results.keys()), v, "--")
        else:
            plt.plot(list(results.keys()), v)

    plt.title(title)
    plt.legend(list(workloads.keys()), loc="lower right")
    plt.ylabel(units)
    plt.xticks(rotation=45)

    plt.savefig(filename, bbox_inches="tight")
    plt.clf()
