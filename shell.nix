with import ./nix/nixpkgs.nix {};

let
  py = python3;
in
mkShell {
  buildInputs = [

    entr

    (py.withPackages (ps: with ps; [

      jupyter
      pytorch-bin
      pandas
      (transformers.overridePythonAttrs (old: {
        version = "3.5.1-dev";
        src = fetchFromGitHub {
          owner = "huggingface";
          repo = "transformers";
          rev = "2819da02f7e3d0c0328daef12115d7a0cc78fc12";
          sha256 = "1xm02jmsaj9i5nj80ryml83fycr080dfw0crzzc06bzcpy5szfra";
          fetchSubmodules = true;
        };
        doCheck = false;
      }))

      tqdm
      scikitlearn

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
