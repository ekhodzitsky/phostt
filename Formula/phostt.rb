# Homebrew formula for phostt.
#
# Install with:
#   brew tap ekhodzitsky/phostt https://github.com/ekhodzitsky/phostt
#   brew install phostt
#
# The `sha256` values below are pinned to the v<version> release tarballs.
# They are refreshed automatically by the `.github/workflows/homebrew.yml`
# workflow after every successful `release.yml` run — do not hand-edit
# unless you are backfilling a release that predated that automation.

class Phostt < Formula
  desc "On-device Vietnamese speech recognition server powered by Zipformer-vi RNN-T"
  homepage "https://github.com/ekhodzitsky/phostt"
  version "0.4.2"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/phostt/releases/download/v0.4.2/phostt-0.4.2-aarch64-apple-darwin.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000" # placeholder: filled by homebrew.yml on first release
    else
      odie "Intel macOS builds are no longer distributed via Homebrew. Build from source with: cargo install phostt"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/phostt/releases/download/v0.4.2/phostt-0.4.2-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000" # placeholder: filled by homebrew.yml on first release
    elsif Hardware::CPU.arm? && Hardware::CPU.is_64_bit?
      url "https://github.com/ekhodzitsky/phostt/releases/download/v0.4.2/phostt-0.4.2-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000" # placeholder: filled by homebrew.yml on first release
    end
  end

  def install
    bin.install "phostt"
  end

  def caveats
    <<~EOS
      The Zipformer-vi RNN-T ONNX bundle (~75 MB INT8) is downloaded on first
      run into ~/.phostt/models. Weights are pre-quantized upstream — no
      separate quantization step is required.

      Quick start:
        phostt download         # fetches Zipformer-vi ONNX bundle
        phostt serve            # starts STT server on 127.0.0.1:9876

      Homepage: https://github.com/ekhodzitsky/phostt
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/phostt --version")
  end
end
