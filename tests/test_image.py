# Copyright 2026 Google LLC
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

"""Unit tests for chs.image."""

import unittest
from unittest import mock

from chs import image
from chs.exceptions import ChsValidationError
from chs.version import UNKNOWN_VERSION


class GetImageTest(unittest.TestCase):
    def test_image_of_the_installed_version(self):
        with mock.patch.object(image, "get_version", return_value="0.4.0"):
            self.assertEqual(
                image.get_image(), f"{image.IMAGE_REPOSITORY}:0.4.0"
            )

    def test_unknown_version_raises(self):
        with mock.patch.object(
            image, "get_version", return_value=UNKNOWN_VERSION
        ):
            with self.assertRaisesRegex(ChsValidationError, "Install CHS"):
                image.get_image()


if __name__ == "__main__":
    unittest.main()
