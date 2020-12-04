with import ./nix/nixpkgs.nix {
  overlays = [ (import ./nix/overlay.nix) ];
};

let
  py = python3;
in
mkShell {
  buildInputs = [

    entr

    (py.withPackages (ps: with ps; [

      (dialog.override ( { pytorch = pytorch-bin; } ))

      # 2020-08-07: wandb not yet available in nixpkgs
      pip

      # dev deps
      pudb  # debugger
      ipython
      pyls-isort
      pyls-mypy
      python-language-server
    ]))
  ];

  shellHook = ''
    export PIP_PREFIX="$(pwd)/.build/pip_packages"
    export PATH="$PIP_PREFIX/bin:$PATH"
    export PYTHONPATH="$PIP_PREFIX/${py.sitePackages}:$PYTHONPATH"
    unset SOURCE_DATE_EPOCH
  '';
}
