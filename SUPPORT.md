# Support

## Getting help

Use [GitHub Issues](https://github.com/GoogleCloudPlatform/cluster-health-scanner/issues) for bug reports, feature requests, and questions about using CHS. Search existing issues before opening a new one.

For security vulnerabilities, do not open an issue. Follow [SECURITY.md](SECURITY.md).

## Before filing

- Check [Troubleshooting](docs/troubleshooting.md) for known errors.
- Run `chs <command> --help` to confirm the options you are using.

## Filing a bug report

Include:

- A detailed description of the issue and steps to reproduce it
- The `chs` command that was run, the scheduler (`kubernetes`, `slurm`, or `standalone`), and the accelerator (machine type)
- Relevant error messages from the terminal output

Do not attach workspace archives or unredacted logs to public GitHub issues, as they contain environment details such as cluster and node names, instance IDs, and bucket names.
