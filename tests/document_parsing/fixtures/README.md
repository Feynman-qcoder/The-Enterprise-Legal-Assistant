# Parser fixtures

- `policy.md` is a UTF-8 Markdown/front-matter fixture.
- DOCX fixtures are generated with `python-docx` in `tmp_path` so they remain valid Office packages.
- PDF coverage reuses `data/中华人民共和国劳动法.pdf`.
- Legacy DOC coverage uses the repository's real OLE binary
  `项目用到的软件及工具/mysql软件/mac系统/mysql安装步骤(mac版本).doc`.
  The current machine has no LibreOffice, so that test asserts the explicit
  `PARSER_UNAVAILABLE` capability rather than claiming conversion success.
