# Legal notes

Plain-language status of the legal questions that matter for this project. Not legal advice.

## Emulators

Emulators themselves are legal in essentially every jurisdiction (*Sony v. Connectix*, *Sony v. Bleem* in the US). Xenia is open source (BSD-3). This project is a derivative work of xenia-canary under BSD-3; see `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## Game dumps

- **Do legal backups.** In the US, making one archival copy of software you own is defensible fair use; distributing, uploading, or downloading copies you don't own is not. This project never hosts dumps and neither can you through it.
- **Microsoft copyrighted material** — the Xbox 360 NAND flash, CPU key, and system executables are Microsoft's property. They are required to decrypt and boot retail titles and are **never** distributed with this project. Extract them from your own console into your own files.
- **Homebrew** is generally free to distribute and is the right way to test the emulator end-to-end.

## "BIOS" / firmware

The Xbox 360 does not boot like a PC BIOS machine — it loads signed executables from NAND. Same rules: yours to extract, nobody else's to give you.

## Content protection

This project includes no DRM circumvention. XEX/ELF parsing is format inspection for interoperability (unambiguously fine under fair use / interoperability doctrine). Decryption of retail images requires keys that are never included here; nothing in this repository performs or enables unlawful circumvention of access controls.

## Trademarks

Xbox 360, Xbox, Xenon, Xenos are trademarks of Microsoft Corporation. Xenia is a trademark of the xenia contributors. This is an independent, unaffiliated project.

## Your responsibility

If you live somewhere with different rules, follow your local law. If you found a dump for download somewhere, don't; use it at your own risk, and don't ask this project to launder it for you.
