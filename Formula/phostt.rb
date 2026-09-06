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
  version "0.5.0"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/ekhodzitsky/phostt/releases/download/v0.5.0/phostt-0.5.0-aarch64-apple-darwin.tar.gz"
      sha256 "8562ac3460510c7ec9b21bda495ca060d6f46b09af59ca38c98f692fd44ccb00" # placeholder: filled by homebrew.yml on first release
    else
      odie "Intel macOS builds are no longer distributed via Homebrew. Build from source with: cargo install phostt"
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/ekhodzitsky/phostt/releases/download/v0.5.0/phostt-0.5.0-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "ee333f43b5056374a22dee7611ef7b2a935e993acfbe924dff8bbc8862869f88" # placeholder: filled by homebrew.yml on first release
    elsif Hardware::CPU.arm? && Hardware::CPU.is_64_bit?
      url "https://github.com/ekhodzitsky/phostt/releases/download/v0.5.0/phostt-0.5.0-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "42ab229f1eac7ee663a2385d4993c59c0945215ceca75fd55eee65fa2c5e737b" # placeholder: filled by homebrew.yml on first release
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
