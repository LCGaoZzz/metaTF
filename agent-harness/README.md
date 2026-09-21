# metaTF agent harness

An additive Omicos integration, following
[iobrx's Agent/Skill layout](https://github.com/LCGaoZzz/iobrx/tree/5ca23a985d722ce346e79846a90e6e1b9161e179/agent-harness/omicos).
It does not change metaTF algorithms, package dependencies or native CLI contracts.

See [Omicos installation and verification](omicos/README.md). The bundle contains
one Agent, one Skill, on-demand references and a standard-library launcher for
the installed `metatf` CLI. It does not duplicate the scientific implementation
or introduce an MCP server, workflow engine, job scheduler or state database.

```bash
python agent-harness/install_omicos.py --destination /path/to/analysis-workspace --dry-run
python agent-harness/install_omicos.py --destination /path/to/analysis-workspace
python -m pytest tests/test_omicos_harness.py -q
```
