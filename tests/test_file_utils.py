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

"""Unit tests for the file_utils module."""

import fcntl
import os
import pathlib
import tempfile
import unittest

from chs.utils import file_utils


class TestFileUtils(unittest.TestCase):
    """Test suite for file_utils."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = pathlib.Path(self.temp_dir) / "test_file.txt"

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.temp_dir)

    def test_atomic_open_writes_successfully(self):
        """Test atomic_open correctly writes to a file."""
        with file_utils.atomic_open(self.test_file, "w") as f:
            f.write("hello world")

        with open(self.test_file, "r") as f:
            content = f.read()

        self.assertEqual(content, "hello world")

    def test_atomic_open_exception_cleans_up(self):
        """Test atomic_open does not overwrite the file on exception."""
        # Create an initial file
        with open(self.test_file, "w") as f:
            f.write("initial state")

        # Try to atomic_write but fail halfway
        try:
            with file_utils.atomic_open(self.test_file, "w") as f:
                f.write("new state")
                raise ValueError("Something went wrong")
        except ValueError:
            pass

        # Verify the original file was completely untouched
        with open(self.test_file, "r") as f:
            content = f.read()

        self.assertEqual(content, "initial state")

        # Verify temp files were cleaned up
        files = os.listdir(self.temp_dir)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], "test_file.txt")

    def test_atomic_open_rejects_non_write_modes(self):
        """Test atomic_open raises ValueError when non-write mode used."""
        with self.assertRaisesRegex(ValueError, "requires a write mode"):
            with file_utils.atomic_open(self.test_file, "a"):
                pass

        with self.assertRaisesRegex(ValueError, "requires a write mode"):
            with file_utils.atomic_open(self.test_file, "r"):
                pass

    def _try_lock(self, lock_path: pathlib.Path) -> bool:
        fd = os.open(lock_path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
        finally:
            os.close(fd)

    def test_exclusive_lock_excludes_other_holders(self):
        """Test the lock is held for the block and released afterwards."""
        lock_path = pathlib.Path(self.temp_dir) / "app.lock"

        with file_utils.exclusive_lock(lock_path):
            self.assertFalse(self._try_lock(lock_path))

        self.assertTrue(self._try_lock(lock_path))

    def test_exclusive_lock_is_released_on_error(self):
        """Test an exception inside the block still releases the lock."""
        lock_path = pathlib.Path(self.temp_dir) / "app.lock"

        with self.assertRaisesRegex(ValueError, "boom"):
            with file_utils.exclusive_lock(lock_path):
                raise ValueError("boom")

        self.assertTrue(self._try_lock(lock_path))

    def test_exclusive_lock_on_read_only_lock_file(self):
        """Test a lock file created read-only by another user still locks."""
        lock_path = pathlib.Path(self.temp_dir) / "app.lock"
        lock_path.touch(mode=0o444)

        with file_utils.exclusive_lock(lock_path):
            self.assertFalse(self._try_lock(lock_path))

        self.assertTrue(self._try_lock(lock_path))


if __name__ == "__main__":
    unittest.main()
