# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Converter docs/ATTR_RTG_RCMZ_PREREGISTRATION.md para PDF compartilhável.

## Summary

O preregistration ATTR-RTG-RCMZ foi convertido para PDF A4 com estilo legível para compartilhamento. O documento possui 8 páginas, título e status `LOCAL PROTOCOL LOCK — LOCAL-ONLY`, tabelas formatadas, blocos de código e seções paginadas.

- PDF: `docs/ATTR_RTG_RCMZ_PREREGISTRATION.pdf`
- SHA-256: `3ded013d3e7e96115190f6d0e08bfbee73e4910ec085836fc904a474c1686a11`
- Sidecar: `docs/ATTR_RTG_RCMZ_PREREGISTRATION.pdf.sha256`

O Markdown original e o lock não foram alterados.

## Modified files

- [docs/ATTR_RTG_RCMZ_PREREGISTRATION.pdf](../ATTR_RTG_RCMZ_PREREGISTRATION.pdf)
- [docs/ATTR_RTG_RCMZ_PREREGISTRATION.pdf.sha256](../ATTR_RTG_RCMZ_PREREGISTRATION.pdf.sha256)
- [docs/report/0057_pdf-compartilhavel-attr-rtg-rcmz_2026-09-02.md](0057_pdf-compartilhavel-attr-rtg-rcmz_2026-09-02.md)

## Changes

- Gerado PDF A4 estilizado a partir do Markdown selado.
- Criado sidecar SHA-256 para validação do arquivo compartilhado.

## Validation

- Chromium headless PDF generation — passed
- pdfinfo — valid PDF 1.4, A4, 8 pages, tagged, unencrypted
- pdftotext — title/status/content extraction passed
- sha256sum -c PDF sidecar — passed
