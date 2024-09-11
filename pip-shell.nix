{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  environment = {
    sessionVariables = {
      LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    };
  };
  name = "pipzone";
  targetPkgs = pkgs: (with pkgs; [
    python39
    python39Packages.pip
    python39Packages.virtualenv
    gcc
    cmake
  ]);
  runScript = "bash";
}).env
