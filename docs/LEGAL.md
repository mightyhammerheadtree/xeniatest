# Legal notes

Plain-language status of the legal questions that matter for this project. Not legal advice.

## Emulators

Emulators themselves are legal in essentially every jurisdiction (*Sony v. Connectix*, *Sony v. Bleem* in the US). Xenia is open source (BSD-3). This project is a derivative work of xenia-canary under BSD-3; see `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## Game dumps

- **Do legal backups.** In the US, making one archival copy of software you own is defensible fair use; distributing, uploading, or downloading copies you don't own is not. This project never hosts dumps and neither can you through it.
- **Microsoft copyrighted material** — the Xbox 360 NAND flash, CPU key, and system executables are Microsoft's property. They are **never** distributed with this project, and this project's HLE kernel (see [HLE.md](HLE.md)) means they are not *needed* to run recompiled binaries either. The only key involved is the public XEX2 container key, discussed below.
- **Homebrew** is generally free to distribute and is the right way to test the emulator end-to-end.

## "BIOS" / firmware

The Xbox 360 does not boot like a PC BIOS machine — it loads signed executables from NAND. In this project's architecture there is **nothing to extract**: the OS surface is reimplemented as host functions (pure HLE, the xenia model), so no BIOS, kernel or firmware file is required, read or distributed. The retail Xbox 360 needs no console-specific secret for the XEX2 *container* format — the class constant that decrypts retail executables' outer layer is public (it ships in xenia's source), the way the CSS DVD key ships in libdvdcss and desktop DVD players. Retail *content* encryption (CPU key) is a different, untouched layer; nobody needs it here because nothing decrypts or executes retail game code beyond what you recompile yourself for personal use.

## Content protection

This project includes no DRM circumvention. XEX/ELF parsing is format inspection for interoperability (unambiguously fine under fair use / interoperability doctrine). Decrypting a retail executable's **XEX2 container** uses the public container key (see the note above) — the same constant every major open-source Xbox 360 project uses; it is a format constant, not an access-control bypass, and it does not decrypt game content beyond the executable's own load image. This project never reads, requests or requires the CPU key, keyvault or NAND. Nothing in this repository performs or enables unlawful circumvention of access controls.

## Trademarks

Xbox 360, Xbox, Xenon, Xenos are trademarks of Microsoft Corporation. Xenia is a trademark of the xenia contributors. This is an independent, unaffiliated project.

## Your responsibility

If you live somewhere with different rules, follow your local law. If you found a dump for download somewhere, don't; use it at your own risk, and don't ask this project to launder it for you.
