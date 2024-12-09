{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

# let
# faustSrc = pkgs.fetchFromGitHub {
# owner = "crop2000";
# repo = "faust";
# rev = "b6dd1e0d86a15846641a857f8943978f58413411";
# fetchSubmodules = true;
# sha256 = "sha256-vHl1FWrnU2wvsXIOijKotIqLSbHRLpm7P6HdHyP0Po8="; # Placeholder - replace with the actual hash
# };

# faust =  pkgs.faust.overrideAttrs (oldAttrs: {
# version = "sized_io";
# src = faustSrc;
# });

# in
{
  # https://devenv.sh/basics/
  env.GREET = "magnetophon";

  # https://devenv.sh/packages/
  packages = with pkgs; [
    git
    # lldb
    cargo
    rustc
    rustfmt
    cargo-flamegraph
    # rust-analyzer
    # clippy
    # cargo-watch
    # cargo-nextest
    # cargo-expand # expand macros and inspect the output
    # cargo-llvm-lines # count number of lines of LLVM IR of a generic function
    # cargo-inspect
    # cargo-criterion
    # evcxr # make sure repl is in a gc-root
    # cargo-play # quickly run a rust file that has a maint function

    pkg-config

    libjack2
    alsa-lib

    libGL
    xorg.libXcursor
    xorg.libX11 # libX11-xcb.so
    xorg.xcbutilwm # libxcb-icccm.so

    fontconfig
    egl-wayland
    wayland
  ];

  # https://devenv.sh/scripts/
  scripts.hello.exec = "echo hello from $GREET";

  enterShell = ''
    hello
    # git --version
    # fish
  '';

  # https://devenv.sh/tests/
  # enterTest = ''
  # echo "Running tests"
  # git --version | grep "2.42.0"
  # '';

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/languages/
  # https://devenv.sh/reference/options/#languagesrustchannel
  languages.rust = {
    enable = true;
    channel = "nightly";
  };

  # https://devenv.sh/pre-commit-hooks/
  pre-commit.hooks = {
    shellcheck.enable = true;
    # clippy.enable = true;
    # hunspell.enable = true;
    # alejandra.enable = true;
    rustfmt.enable = true;
    typos.enable = true;
  };

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
