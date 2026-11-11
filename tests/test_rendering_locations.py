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

"""Unit tests for artifact location helpers in chs.core.utils.rendering."""

import io
import unittest

from rich.console import Console

from chs.core.utils import rendering


class TestGetWebHref(unittest.TestCase):
    def test_gcs_uri_maps_to_console_url(self):
        self.assertEqual(
            rendering.get_web_href("gs://bucket/prefix/run.html"),
            "https://storage.cloud.google.com/bucket/prefix/run.html",
        )

    def test_http_urls_are_returned_unchanged(self):
        for url in ("https://example.com/a.html", "http://example.com/a"):
            self.assertEqual(rendering.get_web_href(url), url)

    def test_local_paths_have_no_href(self):
        for path in (
            "./chs-output/run1/bill_of_health.html",
            "/tmp/run1/bill_of_health.json",
            "relative/dir",
        ):
            self.assertIsNone(rendering.get_web_href(path))

    def test_empty_values_have_no_href(self):
        self.assertIsNone(rendering.get_web_href(None))
        self.assertIsNone(rendering.get_web_href(""))

    def test_url_unsafe_characters_are_percent_encoded(self):
        self.assertEqual(
            rendering.get_web_href("gs://bucket/my prefix/p]x[y/run.html"),
            "https://storage.cloud.google.com/bucket/my%20prefix/p%5Dx%5By/"
            "run.html",
        )

    def test_already_encoded_and_query_urls_are_unchanged(self):
        url = "https://example.com/a%20b/run.html?x=1&y=2#top"
        self.assertEqual(rendering.get_web_href(url), url)


class TestFormatLocation(unittest.TestCase):
    def _render(self, markup: str) -> str:
        buf = io.StringIO()
        Console(file=buf, width=200).print(markup)
        return buf.getvalue().rstrip("\n")

    def test_gcs_location_is_hyperlinked(self):
        markup = rendering.format_location("gs://bucket/run.html")
        self.assertEqual(
            markup,
            "[link=https://storage.cloud.google.com/bucket/run.html]"
            "gs://bucket/run.html[/link]",
        )

    def test_local_location_is_not_hyperlinked(self):
        path = "./chs-output/run1/bill_of_health.html"
        markup = rendering.format_location(path)
        self.assertNotIn("[link=", markup)
        self.assertEqual(self._render(markup), path)

    def test_brackets_in_path_render_verbatim(self):
        path = "/tmp/out/[run]/bill_of_health.html"
        self.assertEqual(self._render(rendering.format_location(path)), path)

    def test_local_location_emits_no_osc8_hyperlink(self):
        buf = io.StringIO()
        Console(file=buf, force_terminal=True, color_system="truecolor").print(
            rendering.format_location("/tmp/run1/bill_of_health.html")
        )
        self.assertNotIn("\x1b]8;", buf.getvalue())

    def test_brackets_in_gcs_path_render_without_markup_error(self):
        uri = "gs://bucket/[run]/bill_of_health.html"
        # Must not raise rich.errors.MarkupError
        self.assertEqual(self._render(rendering.format_location(uri)), uri)

    def test_gcs_location_with_brackets_encodes_href(self):
        uri = "gs://bucket/[run]/bill_of_health.html"
        markup = rendering.format_location(uri)
        self.assertIn("%5Brun%5D", markup)
        self.assertIn(r"\[run]", markup)


if __name__ == "__main__":
    unittest.main()
