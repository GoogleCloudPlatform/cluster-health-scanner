# Security Policy

To report a security issue, please use [g.co/vulnz](https://g.co/vulnz).

The Google Security Team will respond within 5 working days of your report on g.co/vulnz.

We use g.co/vulnz for our intake, and do coordination and disclosure here using GitHub Security Advisory to privately discuss and fix the issue.

## Sensitive data in CHS output

Bill of Health reports, workspace archives, and logs may contain sensitive
information such as cluster and node names, Compute Engine instance IDs,
bucket names, and the `chs run` command line. Do not attach workspace
archives or unredacted logs to public GitHub issues.
