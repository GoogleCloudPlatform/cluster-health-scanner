# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for diffing two NodeConfigs."""

import config

DIFF_EXPLANATION_TEMPLATE = "(Current) %s -> (Golden) %s"
CONFIG_DIFF_EXPLANATION_TEMPLATE = "%s -> %s"


def diff_configs(
    experiment: config.NodeConfig, golden: config.NodeConfig
) -> config.NodeDiff:
  """Diffs an experiment config against a golden config.

  Args:
    experiment: The experiment config to diff.
    golden: The golden config to diff against.

  Returns:
    A NodeConfig named after the experiment config, populated with differences
    between the experiment and golden configs. If the configs are the same, a
    NodeConfig with no dependencies is returned.
  """
  diffs = {}
  for golden_dep in golden.dependencies.values():
    experiment_dep = experiment.dependencies.get(
        golden_dep.name, config.DependencyConfig(name="", version="None")
    )
    dependency_diff = config.DependencyDiff(name=golden_dep.name)
    should_diff_version = (
        not experiment_dep.config_settings
        and not golden_dep.config_settings
        and golden_dep.version != experiment_dep.version
    )
    if should_diff_version:
      dependency_diff.diff_explanation = DIFF_EXPLANATION_TEMPLATE % (
          experiment_dep.version,
          golden_dep.version,
      )
      dependency_diff.experiment_dependency = experiment_dep
      dependency_diff.golden_dependency = golden_dep
    elif experiment_dep.config_settings and golden_dep.config_settings:
      diff_explanation = ""
      for golden_key, golden_value in golden_dep.config_settings.items():
        if golden_value != experiment_dep.config_settings.get(golden_key):
          diff_explanation += (
              f"{golden_key}: "
              + CONFIG_DIFF_EXPLANATION_TEMPLATE
              % (
                  experiment_dep.config_settings.get(golden_key),
                  golden_value,
              )
              + "\n"
          )
      if diff_explanation:
        dependency_diff.diff_explanation = diff_explanation
        dependency_diff.experiment_dependency = experiment_dep
        dependency_diff.golden_dependency = golden_dep
    elif experiment_dep.version == golden_dep.version:
      dependency_diff.diff_explanation = (
          f"{experiment_dep.version} ({config.DependencyDiff.DEFAULT_DIFF_EXPLANATION})"
      )
    else:
      dependency_diff.diff_explanation = experiment_dep.version
    diffs[golden_dep.name] = dependency_diff
  skipped_deps = experiment.dependencies.keys() - golden.dependencies.keys()
  for skipped_dep in skipped_deps:
    dep = experiment.dependencies[skipped_dep]
    diffs[skipped_dep] = config.DependencyDiff(
        name=skipped_dep,
        diff_explanation=dep.version,
        experiment_dependency=dep,
        golden_dependency=None,
    )
  return config.NodeDiff(
      name=experiment.name,
      dependency_diffs=diffs,
  )
