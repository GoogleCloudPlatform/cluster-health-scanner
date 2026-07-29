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

import unittest
import click

from chs.main import _normalize_argv
from chs.validators.utils import validate_non_empty_string


class TestMain(unittest.TestCase):
    def test_normalize_argv_empty(self):
        self.assertEqual(_normalize_argv([]), [])

    def test_normalize_argv_no_commas(self):
        argv = ["chs", "run", "--suite", "network"]
        self.assertEqual(_normalize_argv(argv), argv)

    def test_normalize_argv_commas_between_args(self):
        argv = ["chs", "run", "--tests", "a,", "b,", "c"]
        expected = ["chs", "run", "--tests", "a,b,c"]
        self.assertEqual(_normalize_argv(argv), expected)

    def test_normalize_argv_comma_starts_token(self):
        argv = ["chs", "run", "--tests", "a", ",b", ",c"]
        expected = ["chs", "run", "--tests", "a,b,c"]
        self.assertEqual(_normalize_argv(argv), expected)

    def test_normalize_argv_standalone_comma(self):
        argv = ["chs", "run", "--tests", "a", ",", "b"]
        expected = ["chs", "run", "--tests", "a,b"]
        self.assertEqual(_normalize_argv(argv), expected)

    def test_normalize_argv_comma_with_dash_flag(self):
        # Even if previous ends with comma, a token starting with '-'
        # shouldn't be merged
        argv = ["chs", "run", "--tests", "a,", "--other-flag"]
        expected = ["chs", "run", "--tests", "a,", "--other-flag"]
        self.assertEqual(_normalize_argv(argv), expected)

    def test_normalize_argv_commas_mixed(self):
        argv = [
            "chs",
            "run",
            "--nodes",
            "node1,",
            "node2",
            "--tests",
            "a",
            ",",
            "b,",
            "c",
            "-v",
        ]
        expected = [
            "chs",
            "run",
            "--nodes",
            "node1,node2",
            "--tests",
            "a,b,c",
            "-v",
        ]
        self.assertEqual(_normalize_argv(argv), expected)

    def test_normalize_argv_starts_with_dash_value(self):
        argv = ["chs", "run", "--nodes", "-node1"]
        expected = ["chs", "run", "--nodes", "-node1"]
        self.assertEqual(_normalize_argv(argv), expected)

    def test_normalize_argv_ignores_unrelated_positionals(self):
        argv = ["chs", "results", "run-id-1234", "other-positional"]
        expected = ["chs", "results", "run-id-1234", "other-positional"]
        self.assertEqual(_normalize_argv(argv), expected)


class TestValidatorsUtils(unittest.TestCase):
    def setUp(self):
        self.ctx = click.Context(click.Command("test"))
        self.param = click.Option(["--test"])

    def test_validate_non_empty_string_none(self):
        self.assertIsNone(validate_non_empty_string(self.ctx, self.param, None))

    def test_validate_non_empty_string_empty(self):
        with self.assertRaises(click.BadParameter):
            validate_non_empty_string(self.ctx, self.param, "")
        with self.assertRaises(click.BadParameter):
            validate_non_empty_string(self.ctx, self.param, "   ")

    def test_validate_non_empty_string_valid(self):
        self.assertEqual(
            validate_non_empty_string(self.ctx, self.param, " abc "), "abc"
        )

    def test_validate_non_empty_string_tuple_empty(self):
        self.assertIsNone(validate_non_empty_string(self.ctx, self.param, ()))

    def test_validate_non_empty_string_tuple_one(self):
        self.assertEqual(
            validate_non_empty_string(self.ctx, self.param, ("abc",)), "abc"
        )

    def test_validate_non_empty_string_tuple_multiple(self):
        self.assertEqual(
            validate_non_empty_string(self.ctx, self.param, ("abc", "def")),
            "abc,def",
        )

    def test_validate_non_empty_string_tuple_mixed_whitespace(self):
        self.assertEqual(
            validate_non_empty_string(
                self.ctx, self.param, ("  abc  ", " def")
            ),
            "abc,def",
        )

    def test_validate_non_empty_string_tuple_empty_strings(self):
        self.assertEqual(
            validate_non_empty_string(
                self.ctx, self.param, ("", "  ", "abc", "")
            ),
            "abc",
        )

    def test_validate_non_empty_string_tuple_only_empty_strings(self):
        with self.assertRaises(click.BadParameter):
            validate_non_empty_string(self.ctx, self.param, ("", "  "))
