# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from hydra.core.config_store import ConfigStore

from xformers.factory.hydra_helper import import_xformer_config_schema


def test_import_schema():
    import_xformer_config_schema()
    cs = ConfigStore.instance()
    groups = cs.list("xformers")
    # check all groups registered
    assert groups == ["attention", "ff", "pe"]
    # check the attention is registered
    attentions = cs.list("xformers/attention")
    assert "favor_schema.yaml" in attentions
